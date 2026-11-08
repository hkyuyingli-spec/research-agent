import os
import json
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient
from google import genai
from google.genai import types

# Load local .env file if present
load_dotenv()

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="wide")

MODEL = "gemini-3.5-flash"
OFFICIAL_DOMAINS = [
    "pajak.go.id",
    "jdih.kemenkeu.go.id",
    "peraturan.go.id",
    "peraturan.bpk.go.id",
    "kemenkeu.go.id",
]

SOCIAL_DOMAINS = [
    "youtube.com",
    "tiktok.com",
    "facebook.com",
    "reddit.com",
    "x.com",
]


# ---------- Keys & Secrets Helper ----------
def get_secret(key_name):
    """Read from environment (.env) first, then fall back to st.secrets wrapped in try/except."""
    val = os.getenv(key_name)
    if val and val.strip():
        return val.strip()
    try:
        if key_name in st.secrets:
            sec_val = str(st.secrets[key_name]).strip()
            if sec_val:
                return sec_val
    except Exception:
        pass
    return None


TAVILY_API_KEY = get_secret("TAVILY_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")


# ---------- Verify API Keys ----------
if not TAVILY_API_KEY or not GEMINI_API_KEY:
    st.error(
        "API key missing. Please verify TAVILY_API_KEY and GEMINI_API_KEY are configured in your "
        ".env file or Streamlit Community Cloud Secrets."
    )
    st.stop()

tavily = TavilyClient(api_key=TAVILY_API_KEY)
gemini = genai.Client(api_key=GEMINI_API_KEY)


# ---------- Helpers ----------
def ask_gemini(prompt, as_json=False):
    config = types.GenerateContentConfig(response_mime_type="application/json") if as_json else None
    response = gemini.models.generate_content(model=MODEL, contents=prompt, config=config)
    return response.text


def is_official(url):
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in OFFICIAL_DOMAINS)


def is_social(url):
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in SOCIAL_DOMAINS)


def plan_queries(question):
    prompt = f"""Turn this research question into 3 different web search queries
that together cover the topic (for example: the law itself, recent changes, official guidance).
Question: {question}
Return JSON only: a list of 3 strings."""
    try:
        queries = json.loads(ask_gemini(prompt, as_json=True))
        queries = [q for q in queries if isinstance(q, str)][:3]
        return queries or [question]
    except Exception:
        return [question]


def search_all(queries, official_only=False, include_social=False):
    options = {"include_domains": OFFICIAL_DOMAINS} if official_only else {}
    found = {}
    for q in queries:
        # 1. Normal search runs first
        try:
            results = tavily.search(query=q, search_depth="basic", max_results=5, **options).get("results", [])
            for r in results:
                u = r["url"]
                if u not in found:
                    found[u] = {
                        "title": r.get("title", "Untitled"),
                        "url": u,
                        "content": r.get("content", ""),
                        "social": is_social(u),
                    }
        except Exception as e:
            st.warning(f"Standard search encountered an issue for query '{q}': {e}")

        # 2. Extra social search if enabled
        if include_social:
            try:
                soc_results = tavily.search(
                    query=q,
                    search_depth="basic",
                    max_results=5,
                    include_domains=SOCIAL_DOMAINS,
                ).get("results", [])
                for r in soc_results:
                    u = r["url"]
                    if u not in found:
                        found[u] = {
                            "title": r.get("title", "Untitled"),
                            "url": u,
                            "content": r.get("content", ""),
                            "social": True,
                        }
            except Exception as e:
                st.warning(f"Social media search failed for query '{q}': {e}. Continuing with standard results.")

    return list(found.values())


def review_sources(question, sources):
    listing = "\n".join(
        f"{i}. [{('Social' if s.get('social') else 'Web')}] {s['title']} | {s['url']}\n   {s['content'][:300]}"
        for i, s in enumerate(sources)
    )
    prompt = f"""You are reviewing search results for this research question: {question}
Rate each result from 1 to 5 for how useful and trustworthy it is (5 = best).
Prefer official regulations and recent authoritative material.
Be critical of social media posts (YouTube, TikTok, Reddit, X, Facebook): do not inflate their score, as they represent unverified public sentiment or personal commentary rather than verified facts.
Give a short reason.
Return JSON only: a list like [{{"id": 0, "score": 4, "note": "short reason"}}], one entry per result.

Results:
{listing}"""
    try:
        ratings = json.loads(ask_gemini(prompt, as_json=True))
    except Exception:
        ratings = []
    by_id = {r.get("id"): r for r in ratings if isinstance(r, dict)}
    for i, s in enumerate(sources):
        r = by_id.get(i, {})
        s["score"] = r.get("score", 3)
        s["note"] = r.get("note", "")
        s["social"] = s.get("social", False) or is_social(s["url"])
        # A social result must NEVER get the official label even if domain check behaves unexpectedly
        s["official"] = is_official(s["url"]) and not s["social"]
    return sources


def pick_best(sources, top_n=4):
    for s in sources:
        base_score = s.get("score", 3)
        if s.get("official"):
            # Official government domains get a +1 ranking bonus
            s["effective_score"] = base_score + 1
        elif s.get("social"):
            # Social media sources get a -1 penalty (no official bonus, verified sources preferred)
            s["effective_score"] = base_score - 1
        else:
            s["effective_score"] = base_score
    sorted_sources = sorted(sources, key=lambda x: x["effective_score"], reverse=True)
    return sorted_sources[:top_n]


def fetch_full_text(sources):
    urls = [s["url"] for s in sources]
    extracted_map = {}
    try:
        res = tavily.extract(urls=urls)
        for item in res.get("results", []):
            extracted_map[item["url"]] = item.get("raw_content", "")
    except Exception:
        pass

    for s in sources:
        raw = extracted_map.get(s["url"], "")
        if raw and raw.strip():
            s["full_text"] = raw.strip()[:12000]
            s["extracted"] = True
        else:
            s["full_text"] = s.get("content", "") + " (snippet only)"
            s["extracted"] = False
    return sources


def summarize(question, sources, language="English", include_social=False):
    sources_text = ""
    for idx, s in enumerate(sources, start=1):
        source_type = "Social/unverified" if s.get("social") else ("Official Government" if s.get("official") else "Standard Web")
        sources_text += f"\n--- Source [{idx}] ({source_type}) ---\nTitle: {s['title']}\nURL: {s['url']}\nNote: {s.get('note', '')}\nContent:\n{s.get('full_text', s.get('content', ''))}\n"

    social_instructions = ""
    social_section_structure = ""
    if include_social:
        social_instructions = """
5. Social Media Rules:
   - Any claims originating from social media sources (YouTube, TikTok, Reddit, X, Facebook) MUST be placed strictly in a dedicated section titled '### 5. Social Media Signals'.
   - State clearly and explicitly that these are unverified public opinions, discussions, or anecdotal reports, NOT official facts or legal authority.
   - NEVER use or cite a social media source to support a regulation number, legal rate, tax threshold, or official enactment date.
"""
        social_section_structure = """
   ### 5. Social Media Signals
   (Summarize public sentiment, community discussions, or unverified reports from social sources with citations [1], [2]. Explicitly state that these are unverified public opinions. If no social media sources are present in the provided sources, state 'No social media sources identified.')

   ### 6. References
   (Numbered list of sources used with title, URL, and category: Official, Web, or Social/unverified)
"""
    else:
        social_section_structure = """
   ### 5. References
   (Numbered list of sources used with title and URL)
"""

    prompt = f"""You are a specialized legal and web research assistant.
Answer this research question strictly using ONLY the provided sources below.

Question: {question}

Sources:
{sources_text}

Instructions:
1. Rely ONLY on the information present in the sources. Do not extrapolate or guess.
2. Cite all claims with numbered citations matching the sources, e.g. [1], [2].
3. Answer in {language}.
4. Follow this EXACT report structure:
   ### 1. Short Answer
   (2-4 sentences summarizing the core conclusion)
   
   ### 2. Key Findings
   (Bullet points with citations like [1], [2])
   
   ### 3. Applicable Regulations
   (Explicitly state regulation numbers, dates, and titles, e.g., Law No. 7/2021, PMK No. ...)
   
   ### 4. Conflicts or Gaps
   (Clearly note any conflicting rules between sources, or information that is missing or insufficient)
{social_section_structure}{social_instructions}
"""
    return ask_gemini(prompt)


# ---------- Streamlit UI ----------
st.title("🔎 Research Agent")
st.caption("AI-powered web research with Tavily and Gemini")

with st.sidebar:
    st.header("Search Settings")
    official_only = st.checkbox("Restrict to official government domains", value=False)
    include_social = st.checkbox("Include social media", value=False)
    st.caption("Adds YouTube, TikTok, Facebook, Reddit and X. Results are unverified.")
    top_n = st.slider("Number of top sources to read in full", min_value=2, max_value=8, value=4)
    language = st.selectbox("Answer Language", ["English", "Chinese (Simplified)"])

question = st.text_input("Enter your research question:", placeholder="e.g. What is the VAT rate on digital goods in Indonesia?")
run_btn = st.button("Start Research", type="primary", disabled=not question.strip())

if run_btn and question.strip():
    # 1. PLAN
    st.subheader("1. Plan: Generating Search Queries")
    with st.spinner("Generating 3 search queries with Gemini..."):
        queries = plan_queries(question)
    st.write("Generated search queries:")
    for q in queries:
        st.markdown(f"- `{q}`")

    # 2. SEARCH
    st.subheader("2. Search: Gathering Web Results")
    with st.spinner("Searching via Tavily..."):
        raw_sources = search_all(queries, official_only=official_only, include_social=include_social)
    st.write(f"Found **{len(raw_sources)}** unique sources across queries.")
    with st.expander("View all gathered sources"):
        for s in raw_sources:
            if s.get("social"):
                badge = "📱 Social/unverified"
            elif s.get("official"):
                badge = "🏛️ Official"
            else:
                badge = "🌐 Web"
            st.markdown(f"- **[{s['title']}]({s['url']})** ({badge})")

    if not raw_sources:
        st.warning("No search results found. Try adjusting your query or disabling official domains restriction.")
        st.stop()

    # 3. REVIEW
    st.subheader("3. Review: Scoring & Ranking Sources")
    with st.spinner("Scoring sources with Gemini..."):
        reviewed = review_sources(question, raw_sources)
        best_sources = pick_best(reviewed, top_n=top_n)

    with st.expander("View source ratings and ranking"):
        table_data = []
        for s in reviewed:
            adj = "None"
            if s.get("official"):
                adj = "+1 (Official)"
            elif s.get("social"):
                adj = "-1 (Social)"

            table_data.append({
                "Title": s.get("title", "Untitled")[:60],
                "Social": "Yes" if s.get("social") else "No",
                "Official": "Yes" if s.get("official") else "No",
                "Base Score": s.get("score", 3),
                "Adjustment": adj,
                "Effective": s.get("effective_score", s.get("score", 3)),
                "Reason": s.get("note", ""),
            })
        st.dataframe(table_data, use_container_width=True)

    # 4. READ
    st.subheader("4. Read: Extracting Full Page Text")
    with st.spinner(f"Extracting full text for top {len(best_sources)} sources (up to 12,000 chars each)..."):
        best_sources = fetch_full_text(best_sources)

    with st.expander(f"View extracted content for top {len(best_sources)} sources"):
        for idx, s in enumerate(best_sources, start=1):
            status = "Full page extracted" if s.get("extracted") else "Snippet only"
            type_label = "Social/unverified" if s.get("social") else ("Official" if s.get("official") else "Web")
            st.markdown(f"**[{idx}] {s['title']}** (`{type_label}` | `{status}` - {len(s.get('full_text', ''))} characters)")
            st.caption(s["url"])

    # 5. SUMMARIZE
    st.subheader("5. Summarize: Final Research Report")
    with st.spinner("Synthesizing cited report with Gemini..."):
        report = summarize(question, best_sources, language=language, include_social=include_social)

    st.markdown("---")
    st.markdown(report)
