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
APP_PASSWORD = get_secret("APP_PASSWORD")


# ---------- Password Protection Gate ----------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 Indonesian Tax & Regulation Research Agent")
    st.subheader("Login Required")

    if not APP_PASSWORD:
        st.warning(
            "APP_PASSWORD is not set yet. Please add `APP_PASSWORD=your_choice` to your .env file "
            "(or to Streamlit Secrets when deployed) to protect this app."
        )
        st.stop()

    with st.form("login_form"):
        pwd_input = st.text_input("Enter password:", type="password")
        submitted = st.form_submit_button("Enter App", type="primary")

    if submitted:
        if pwd_input == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

    # Stop rendering anything else until authenticated
    st.stop()


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


def search_all(queries, official_only):
    options = {"include_domains": OFFICIAL_DOMAINS} if official_only else {}
    found = {}
    for q in queries:
        results = tavily.search(query=q, search_depth="basic", max_results=5, **options)["results"]
        for r in results:
            if r["url"] not in found:
                found[r["url"]] = {
                    "title": r["title"],
                    "url": r["url"],
                    "content": r.get("content", ""),
                }
    return list(found.values())


def review_sources(question, sources):
    listing = "\n".join(
        f"{i}. {s['title']} | {s['url']}\n   {s['content'][:300]}" for i, s in enumerate(sources)
    )
    prompt = f"""You are reviewing search results for this research question: {question}
Rate each result from 1 to 5 for how useful and trustworthy it is (5 = best).
Prefer official regulations and recent material. Give a short reason.
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
        s["official"] = is_official(s["url"])
    return sources


def pick_best(sources, top_n=4):
    for s in sources:
        # Official government domains get a +1 ranking bonus
        s["effective_score"] = s.get("score", 3) + (1 if s.get("official") else 0)
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


def summarize(question, sources, language="English"):
    sources_text = ""
    for idx, s in enumerate(sources, start=1):
        sources_text += f"\n--- Source [{idx}] ---\nTitle: {s['title']}\nURL: {s['url']}\nNote: {s.get('note', '')}\nContent:\n{s.get('full_text', s.get('content', ''))}\n"

    prompt = f"""You are a specialized legal and tax research assistant focusing on Indonesian regulations.
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
   
   ### 5. References
   (Numbered list of sources used with title and URL)
"""
    return ask_gemini(prompt)


# ---------- Streamlit UI ----------
st.title("🔎 Indonesian Tax & Regulation Research Agent")
st.caption("AI-powered legal research with Tavily and Gemini 3.5 Flash")

with st.sidebar:
    st.header("Search Settings")
    official_only = st.checkbox("Restrict to official government domains", value=False)
    top_n = st.slider("Number of top sources to read in full", min_value=2, max_value=8, value=4)
    language = st.selectbox("Answer Language", ["English", "Chinese (Simplified)"])

    st.markdown("---")
    if st.button("Log Out"):
        st.session_state["authenticated"] = False
        st.rerun()

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
        raw_sources = search_all(queries, official_only=official_only)
    st.write(f"Found **{len(raw_sources)}** unique sources across queries.")
    with st.expander("View all gathered sources"):
        for s in raw_sources:
            badge = "🏛️ Official" if is_official(s["url"]) else "🌐 Web"
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
        for s in reviewed:
            bonus = " (+1 official bonus)" if s.get("official") else ""
            st.markdown(f"- **{s['title']}** — Score: `{s.get('score', 3)}/5`{bonus}")
            if s.get("note"):
                st.caption(f"Reason: {s['note']}")
            st.caption(f"URL: {s['url']}")

    # 4. READ
    st.subheader("4. Read: Extracting Full Page Text")
    with st.spinner(f"Extracting full text for top {len(best_sources)} sources (up to 12,000 chars each)..."):
        best_sources = fetch_full_text(best_sources)

    with st.expander(f"View extracted content for top {len(best_sources)} sources"):
        for idx, s in enumerate(best_sources, start=1):
            status = "Full page extracted" if s.get("extracted") else "Snippet only"
            st.markdown(f"**[{idx}] {s['title']}** (`{status}` - {len(s.get('full_text', ''))} characters)")
            st.caption(s["url"])

    # 5. SUMMARIZE
    st.subheader("5. Summarize: Final Research Report")
    with st.spinner("Synthesizing cited report with Gemini..."):
        report = summarize(question, best_sources, language=language)

    st.markdown("---")
    st.markdown(report)
