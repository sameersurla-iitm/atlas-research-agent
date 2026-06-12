"""
Atlas — Autonomous Research Agent (Streamlit UI).

Run locally:
    streamlit run app.py

The UI drives the four-agent pipeline and renders each stage live: the Planner's
decomposition, the Researcher's evidence gathering, the Critic's review, and the
streamed final report with citations, sources, and run telemetry.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

import config
from src.models import Source, UsageStats
from src.pipeline import ResearchPipeline

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Atlas — Autonomous Research Agent",
    page_icon="🛰️",
    layout="wide",
)


def load_css() -> None:
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


load_css()

# Session state
st.session_state.setdefault("query_input", "")
st.session_state.setdefault("history", [])       # completed runs for THIS session
st.session_state.setdefault("active_id", None)    # id of the run currently displayed
st.session_state.setdefault("pending", None)      # {"query","provider"} to run next pass
st.session_state.setdefault("run_counter", 0)     # monotonic id source
st.session_state.setdefault("authed", False)


def check_password() -> bool:
    """Optional gate for public deployments.

    Active ONLY when an APP_PASSWORD secret/env var is set. When unset (the
    default, e.g. local dev), this is a no-op and the app is open — so it never
    gets in the way of a recruiter trying the live demo. When set, it protects
    your free-tier API quota from anonymous abuse on a public URL.
    """
    import os

    expected = os.getenv("APP_PASSWORD", "")
    if not expected:
        return True
    if st.session_state.authed:
        return True

    st.markdown(
        '<div class="hero"><div class="hero-badge">PROTECTED DEMO</div>'
        '<h1 class="hero-title">Atlas</h1>'
        '<p class="hero-sub">Enter the access password to continue.</p></div>',
        unsafe_allow_html=True,
    )
    pw = st.text_input("Access password", type="password", label_visibility="collapsed")
    if pw:
        if pw == expected:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()


EXAMPLE_QUERIES = [
    "Compare LangGraph, AutoGPT, and CrewAI for multi-agent systems",
    "What techniques reduce LLM inference cost in production?",
    "How does retrieval-augmented generation reduce hallucinations?",
    "Explain the ReAct prompting pattern and when to use it",
]


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------
def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-badge">AUTONOMOUS RESEARCH AGENT</div>
          <h1 class="hero-title">Atlas</h1>
          <p class="hero-sub">A multi-agent system that plans, researches the live web,
          self-critiques for gaps, and writes a fully-cited report.</p>
          <div class="pipeline-pills">
            <span class="pill">🧭 Plan</span><span class="arrow">→</span>
            <span class="pill">🔎 Research</span><span class="arrow">→</span>
            <span class="pill">🧪 Critique</span><span class="arrow">→</span>
            <span class="pill">📄 Synthesize</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def key_badge(label: str, ok: bool, note: str = "") -> None:
    state = "ok" if ok else "off"
    text = note or ("configured" if ok else "not set")
    st.markdown(
        f'<div class="keyrow"><span class="dot {state}"></span>'
        f"<b>{escape(label)}</b><span class='keynote'>{escape(text)}</span></div>",
        unsafe_allow_html=True,
    )


def render_sources(sources: list[Source]) -> None:
    if not sources:
        return
    st.markdown('<div class="section-label">🔗 Sources</div>', unsafe_allow_html=True)
    cards = ['<div class="source-grid">']
    for i, s in enumerate(sources, start=1):
        cred = int(round(s.credibility * 100))
        title = escape(s.title or s.domain or s.url)
        cards.append(
            f'<a class="source-card" href="{escape(s.url)}" target="_blank" rel="noopener">'
            f'<div class="source-index">{i}</div>'
            f'<div class="source-body"><div class="source-title">{title}</div>'
            f'<div class="source-domain">{escape(s.domain)}</div></div>'
            f'<div class="source-cred" title="credibility score">{cred}</div></a>'
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_metrics(stats: UsageStats) -> None:
    st.markdown('<div class="section-label">📊 Run telemetry</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("⏱ Time", f"{stats.elapsed_seconds}s")
    c2.metric("🧠 LLM calls", stats.llm_calls)
    c3.metric("🔎 Searches", stats.search_calls)
    c4.metric("📄 Pages read", stats.pages_read)
    c5.metric("🔢 Tokens", f"{stats.total_tokens:,}")
    c6.metric("💵 Est. cost", f"${stats.estimated_cost_usd:.4f}")
    st.caption(
        "Cost is the *production-equivalent* spend at public API prices — "
        "actual cost on free tiers is $0."
    )


# ----------------------------------------------------------------------------
# Live run
# ----------------------------------------------------------------------------
def _save_run(query: str, provider: str, markdown: str, sources, stats) -> int:
    """Persist a completed run to session history and make it the active one."""
    st.session_state.run_counter += 1
    rid = st.session_state.run_counter
    st.session_state.history.append(
        {
            "id": rid,
            "query": query,
            "provider": provider,
            "markdown": markdown,
            "sources": [s.model_dump() for s in sources],
            "stats": stats.model_dump(),
        }
    )
    st.session_state.active_id = rid
    return rid


def _get_run(rid: int | None) -> dict | None:
    for r in st.session_state.history:
        if r["id"] == rid:
            return r
    return None


def run_and_render(query: str, provider: str) -> bool:
    """Execute the pipeline, rendering each stage live. Returns True on success.

    On success the run is persisted to session history (so it survives any later
    rerun — e.g. switching provider or opening another thread — instead of
    vanishing and wasting the tokens already spent).
    """
    st.markdown('<div class="section-label">🧠 Agent pipeline</div>', unsafe_allow_html=True)
    planner_status = st.status("🧭 **Planner** — decomposing your query…", expanded=True)
    researcher_status = st.status("🔎 **Researcher** — waiting…", expanded=False)
    critic_status = st.status("🧪 **Critic** — waiting…", expanded=False)

    def progress(stage: str, event: str, payload: object) -> None:
        if stage == "planner" and event == "done":
            plan = payload
            planner_status.update(
                label=f"🧭 **Planner** — {len(plan.sub_questions)} sub-questions",
                state="complete",
            )
            with planner_status:
                if plan.interpretation:
                    st.caption(plan.interpretation)
                for sq in plan.sub_questions:
                    st.markdown(f"**{sq.id}. {sq.question}**")
                    if sq.search_queries:
                        st.caption("🔍 " + "  ·  ".join(sq.search_queries))
            researcher_status.update(
                label="🔎 **Researcher** — gathering evidence…",
                state="running",
                expanded=True,
            )

        elif stage == "researcher" and event == "subquestion":
            idx, total = payload["index"], payload["total"]
            sq = payload["sub_question"]
            researcher_status.update(
                label=f"🔎 **Researcher** — sub-question {idx}/{total}",
                state="running",
            )
            with researcher_status:
                st.markdown(f"**{idx}. {escape(sq.question)}**")

        elif stage == "researcher" and event == "searching":
            with researcher_status:
                qs = payload.get("queries", [])
                st.caption("🔍 searching — " + "  ·  ".join(escape(q) for q in qs))

        elif stage == "researcher" and event == "reading":
            with researcher_status:
                src = payload.get("source")
                if src is not None:
                    st.caption(f"📖 reading — {escape(src.domain)}")

        elif stage == "researcher" and event == "extracting":
            with researcher_status:
                st.caption(f"🧠 extracting facts from {payload.get('count', 0)} sources")

        elif stage == "researcher" and event == "update":
            facts = payload["facts"]
            sq = payload["sub_question"]
            with researcher_status:
                if sq is not None:
                    st.markdown(f"✅ *{escape(sq.question)}* — **{len(facts)}** facts")
                else:
                    st.markdown(f"➕ follow-up — **{len(facts)}** facts")

        elif stage == "researcher" and event == "done":
            kb = payload
            researcher_status.update(
                label=(
                    f"🔎 **Researcher** — {len(kb.facts)} facts "
                    f"from {len(kb.unique_sources())} sources"
                ),
                state="complete",
            )
            critic_status.update(label="🧪 **Critic** — checking coverage…", state="running")

        elif stage == "critic" and event == "update":
            crit = payload
            with critic_status:
                verdict = "sufficient ✅" if crit.is_sufficient else "needs more 🔁"
                st.markdown(f"**Assessment:** {verdict}")
                if crit.reasoning:
                    st.caption(crit.reasoning)
                if crit.gaps:
                    st.markdown("**Gaps:** " + "; ".join(escape(g) for g in crit.gaps))
                if crit.conflicts:
                    st.markdown("**Conflicts:** " + "; ".join(escape(c) for c in crit.conflicts))
                if crit.follow_up_queries:
                    st.caption("Follow-ups: " + "  ·  ".join(crit.follow_up_queries))

        elif stage == "critic" and event == "done":
            critic_status.update(label="🧪 **Critic** — review complete", state="complete")

    try:
        pipeline = ResearchPipeline(provider)
        plan, kb, _ = pipeline.run_research(query, progress)

        st.markdown('<div class="section-label">📄 Research report</div>', unsafe_allow_html=True)
        with st.container(border=True):
            markdown = st.write_stream(pipeline.synthesize_stream(query, plan, kb))

        report = pipeline.report_from_markdown(query, markdown, kb)
        stats = pipeline.stats()

        render_sources(report.sources)
        render_metrics(stats)
        _save_run(query, provider, markdown, report.sources, stats)
        return True
    except Exception as exc:  # noqa: BLE001 - surface a friendly error in the UI
        msg = str(exc)
        low = msg.lower()
        blocked = (
            "zscaler" in low
            or "permissiondenied" in type(exc).__name__.lower()
            or "403" in msg
        )
        invalid_key = (
            "api key" in low
            or "api_key" in low
            or "invalid_argument" in low
            or "unauthenticated" in low
            or "401" in msg
            or ("400" in msg and "key" in low)
        )
        if invalid_key:
            st.error("Invalid or missing API key for the selected provider.")
            st.info(
                "**If this is the deployed app** (Streamlit Cloud): open your app → "
                "**⋮ → Settings → Secrets** and make sure the key matches your current "
                "key, with **no surrounding quotes or spaces**. If you regenerated the "
                "key recently, update it here too.\n\n"
                "```toml\nGEMINI_API_KEY = \"your_current_key\"\nLLM_PROVIDER = \"gemini\"\n```\n\n"
                "**If this is local**: check the same key in your `.env` file."
            )
        elif blocked:
            st.error(
                "This provider's API endpoint appears to be **blocked by your "
                "corporate network** (e.g. Zscaler). Switch the **LLM provider** "
                "in the sidebar — Gemini is reachable on this network."
            )
        else:
            st.error(f"Research run failed: {exc}")
            st.info("Check that your selected provider's API key is valid in `.env` (or Streamlit secrets).")
        return False


def render_stored_run(res: dict) -> None:
    """Render a previously-completed run from session history."""
    prov = res.get("provider", "")
    prov_label = config.PROVIDERS.get(prov, {}).get("label", prov)
    st.markdown(
        f'<div class="section-label">📄 Report · {escape(res["query"])}</div>',
        unsafe_allow_html=True,
    )
    if prov_label:
        st.caption(f"Generated with {escape(prov_label)}")
    with st.container(border=True):
        st.markdown(res["markdown"])
    render_sources([Source(**d) for d in res["sources"]])
    render_metrics(UsageStats(**res["stats"]))


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
# A run is in flight whenever a query is pending. While it is, we lock the
# controls so a stray click (e.g. switching provider) can't interrupt the
# generation and throw away tokens already spent.
running = st.session_state.pending is not None

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    provider_keys = list(config.PROVIDERS.keys())
    default_idx = (
        provider_keys.index(config.LLM_PROVIDER)
        if config.LLM_PROVIDER in provider_keys
        else 0
    )
    provider = st.selectbox(
        "LLM provider",
        options=provider_keys,
        index=default_idx,
        format_func=lambda k: config.PROVIDERS[k]["label"],
        disabled=running,
        help="Locked while a research run is in progress.",
    )

    st.markdown("#### 🔑 Credentials")
    has_llm = config.has_llm_key(provider)
    key_badge(config.PROVIDERS[provider]["label"], has_llm)
    key_badge(
        "Brave Search",
        bool(config.BRAVE_API_KEY),
        "" if config.BRAVE_API_KEY else "using DuckDuckGo",
    )
    key_badge("Jina Reader", True, "free, no key needed")

    if not has_llm:
        st.warning(
            "Add a free API key for the selected provider in your `.env` file "
            "to start researching.",
            icon="⚠️",
        )

    st.divider()
    st.markdown("#### 🧵 This session")
    if st.button(
        "＋ New research",
        use_container_width=True,
        disabled=running,
        help="Start a fresh question. Your previous results stay saved below.",
    ):
        st.session_state.active_id = None
        st.session_state.query_input = ""
        st.rerun()

    if st.session_state.history:
        for item in reversed(st.session_state.history):
            is_active = item["id"] == st.session_state.active_id
            short = item["query"][:38] + ("…" if len(item["query"]) > 38 else "")
            label = ("● " if is_active else "○ ") + short
            if st.button(
                label,
                key=f"hist_{item['id']}",
                use_container_width=True,
                disabled=running,
            ):
                st.session_state.active_id = item["id"]
                st.rerun()
    else:
        st.caption("Questions you research will be saved here for this session.")

    st.divider()
    st.markdown("#### 💡 Example queries")

    def _set_query(q: str) -> None:
        st.session_state.query_input = q

    for i, ex in enumerate(EXAMPLE_QUERIES):
        st.button(
            ex,
            key=f"ex_{i}",
            on_click=_set_query,
            args=(ex,),
            use_container_width=True,
            disabled=running,
        )

    st.divider()
    st.caption(
        "**Atlas** runs a Planner → Researcher → Critic → Synthesizer pipeline over "
        "live web search. Built with Streamlit + OpenAI-compatible LLMs."
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
render_header()

col_q, col_btn = st.columns([5, 1])
with col_q:
    query = st.text_input(
        "Research query",
        key="query_input",
        placeholder="Ask anything — e.g. 'How does speculative decoding speed up LLMs?'",
        label_visibility="collapsed",
        disabled=running,
    )
with col_btn:
    run = st.button(
        "🚀 Research",
        type="primary",
        use_container_width=True,
        disabled=(not has_llm) or running,
    )

# Submit: capture the query + provider, then rerun so the controls lock before
# the (blocking) run begins.
if run and query.strip():
    st.session_state.pending = {"query": query.strip(), "provider": provider}
    st.rerun()

if st.session_state.pending:
    p = st.session_state.pending
    ok = run_and_render(p["query"], p["provider"])
    st.session_state.pending = None
    if ok:
        # Re-render from saved history (re-enables the controls cleanly).
        st.rerun()
elif st.session_state.active_id is not None and _get_run(st.session_state.active_id):
    render_stored_run(_get_run(st.session_state.active_id))
else:
    st.markdown(
        '<div class="empty">Enter a question above and Atlas will research it '
        "end-to-end — planning, searching, fact-checking, and citing its sources.</div>",
        unsafe_allow_html=True,
    )
