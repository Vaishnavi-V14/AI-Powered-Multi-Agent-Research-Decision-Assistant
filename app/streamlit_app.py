"""
Streamlit frontend — Production Multi-Agent Research Assistant
Tabs: Research | Evaluation Dashboard | Knowledge Base | History
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.graph import stream_research
from src.database import get_all_sessions, get_eval_trends
from src.document_pipeline import ingest_file, ingest_text, get_ingestion_stats
from src.config import get_settings

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

cfg = get_settings()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Research Assistant")
    st.markdown("---")

    st.subheader("⚙️ Pipeline Config")
    st.code(
        f"Model:  {cfg.ollama_model}\n"
        f"Ollama: {cfg.ollama_base_url}\n"
        f"Retries: {cfg.max_retry}",
        language=None,
    )

    st.markdown("---")
    st.subheader("📄 Upload Documents")
    uploaded = st.file_uploader(
        "PDF, DOCX, TXT, MD",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "md"],
        help="Documents are chunked and stored in ChromaDB for retrieval.",
    )
    if uploaded and st.button("📥 Ingest into Knowledge Base", use_container_width=True):
        total = 0
        with st.spinner("Ingesting..."):
            for f in uploaded:
                raw = f.read()
                # Save to temp file for format-aware extraction
                tmp = Path(f"/tmp/{f.name}")
                tmp.write_bytes(raw)
                total += ingest_file(str(tmp))
        st.success(f"✅ Ingested {total} chunks from {len(uploaded)} file(s)!")
        st.rerun()

    # KB Stats
    stats = get_ingestion_stats()
    if stats["total_chunks"] > 0:
        st.markdown("**Knowledge Base:**")
        st.metric("Chunks", stats["total_chunks"])
        st.metric("Documents", stats["unique_sources"])

    st.markdown("---")
    st.caption("LangChain · LangGraph · Ollama · ChromaDB · BM25 · RAGAS")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_research, tab_eval, tab_kb, tab_history = st.tabs(
    ["🔍 Research", "📊 Evaluation", "📚 Knowledge Base", "🕒 History"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: RESEARCH
# ═══════════════════════════════════════════════════════════════════════════════
with tab_research:
    st.title("🔍 Multi-Agent Research")
    st.markdown(
        "Eight specialized agents collaborate: **Planner → Searcher → Source Verifier → "
        "Retriever (BM25+Vector+Rerank) → Summarizer → Critic (claim-level) → "
        "Finalizer → Evaluator**"
    )

    with st.form("research_form"):
        question = st.text_area(
            "Research Question",
            placeholder='e.g. "What are the latest advances in AI-driven drug discovery?"',
            height=90,
        )
        submitted = st.form_submit_button("🚀 Run Research Pipeline", use_container_width=True)

    if submitted and question.strip():
        st.markdown("---")

        AGENT_INFO = {
            "planner":         ("🗺️",  "Planner",         "Decomposing into sub-queries..."),
            "searcher":        ("🌐",  "Searcher",        "Parallel search across sub-queries..."),
            "source_verifier": ("🔐",  "Source Verifier", "Verifying URLs & cross-referencing claims..."),
            "retriever":       ("🔍",  "Retriever",       "BM25 + Vector + Rerank..."),
            "summarizer":      ("📝",  "Summarizer",      "Compressing key points..."),
            "critic":          ("🧐",  "Critic",          "Claim-level fact checking..."),
            "finalizer":       ("✅",  "Finalizer",       "Synthesizing with citations..."),
            "evaluator":       ("📊",  "Evaluator",       "Computing RAGAS metrics..."),
        }

        col_pipe, col_live = st.columns([1, 2])

        with col_pipe:
            st.subheader("Agent Pipeline")
            agent_containers = {name: st.empty() for name in AGENT_INFO}
            for name, (icon, label, desc) in AGENT_INFO.items():
                agent_containers[name].markdown(f"⬜ {icon} **{label}** — _{desc}_")

        with col_live:
            st.subheader("Live Output")
            live_box = st.empty()

        # Stream pipeline
        final_state: dict = {}
        try:
            for node_name, node_output in stream_research(question):
                # Mark agent complete
                if node_name in AGENT_INFO:
                    icon, label, _ = AGENT_INFO[node_name]
                    agent_containers[node_name].markdown(f"✅ {icon} **{label}**")

                final_state.update(node_output)

                # Live previews per node
                if node_name == "planner":
                    sq = node_output.get("sub_queries", [])
                    live_box.info(
                        f"**Sub-queries ({len(sq)}):**\n"
                        + "\n".join(f"- {q}" for q in sq)
                    )

                elif node_name == "searcher":
                    n = len(node_output.get("raw_data", []))
                    live_box.info(f"🌐 Found **{n}** raw results across all sub-queries.")

                elif node_name == "source_verifier":
                    vs = node_output.get("verified_sources", [])
                    accessible = sum(1 for s in vs if s.get("accessible", True))
                    live_box.info(
                        f"🔐 **{accessible} / {len(vs)}** sources verified accessible."
                    )

                elif node_name == "retriever":
                    chunks = node_output.get("ranked_chunks", [])
                    facts = node_output.get("facts", [])
                    live_box.info(
                        f"🔍 **{len(chunks)}** chunks after hybrid retrieval + rerank.\n"
                        f"**{len(facts)}** facts extracted."
                    )

                elif node_name == "summarizer":
                    s = node_output.get("summary", "")
                    live_box.info(f"📝 Summary drafted ({len(s)} chars).")

                elif node_name == "critic":
                    checks = node_output.get("claim_checks", [])
                    status = node_output.get("validation_status", "")
                    supported = sum(1 for c in checks if c.get("verdict") == "SUPPORTED")
                    if status == "PASS":
                        live_box.success(
                            f"🧐 Critic **PASS** — {supported}/{len(checks)} claims supported."
                        )
                    else:
                        retries = node_output.get("retry_count", 0)
                        live_box.warning(
                            f"🧐 Critic **FAIL** (retry {retries}) — "
                            f"{supported}/{len(checks)} claims supported."
                        )

                elif node_name == "finalizer":
                    live_box.success("✅ Final answer synthesized.")

                elif node_name == "evaluator":
                    scores = node_output.get("eval_scores", {})
                    overall = scores.get("overall_score", 0)
                    live_box.success(f"📊 Evaluation complete — Overall score: **{overall:.2f}**")

        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

        # ── Results ───────────────────────────────────────────────────────────
        st.markdown("---")
        col_ans, col_meta = st.columns([3, 1])

        with col_ans:
            st.subheader("📋 Final Answer")
            final_answer = final_state.get("final_answer", "")
            st.markdown(final_answer or "_No answer generated._")

        with col_meta:
            st.subheader("📊 Scores")
            scores = final_state.get("eval_scores", {})
            if scores:
                for metric, label in [
                    ("faithfulness", "Faithfulness"),
                    ("answer_relevancy", "Relevancy"),
                    ("context_precision", "Precision"),
                    ("context_recall", "Recall"),
                    ("source_coverage", "Coverage"),
                ]:
                    val = scores.get(metric, 0)
                    color = "🟢" if val >= 0.8 else "🟡" if val >= 0.5 else "🔴"
                    st.metric(f"{color} {label}", f"{val:.2f}")
                st.metric("⭐ Overall", f"{scores.get('overall_score', 0):.2f}")

        # Claim checks expander
        claim_checks = final_state.get("claim_checks", [])
        if claim_checks:
            with st.expander(f"🧐 Claim-Level Fact Checks ({len(claim_checks)} claims)"):
                for c in claim_checks:
                    verdict = c.get("verdict", "")
                    icon = "✅" if verdict == "SUPPORTED" else "⚠️" if verdict == "UNSUPPORTED" else "❌"
                    conf = c.get("confidence", 0)
                    st.markdown(
                        f"{icon} **[{verdict}]** _{c.get('claim', '')}_  \n"
                        f"Evidence: {c.get('evidence', 'none')[:150]} _(conf: {conf:.2f})_"
                    )

        # Sources
        verified_sources = final_state.get("verified_sources", [])
        if verified_sources:
            with st.expander(f"🔗 Verified Sources ({len(verified_sources)})"):
                for s in verified_sources:
                    url = s.get("url", "")
                    ok = s.get("accessible", True)
                    conf = s.get("confidence", 0)
                    icon = "✅" if ok else "❌"
                    if url.startswith("http"):
                        st.markdown(f"{icon} [{url}]({url}) _(confidence: {conf:.2f})_")
                    else:
                        st.markdown(f"📄 `{url}` _(local doc, confidence: {conf:.2f})_")

        # Sub-queries
        sub_queries = final_state.get("sub_queries", [])
        if sub_queries:
            with st.expander("🗺️ Planner Sub-Queries"):
                for i, q in enumerate(sub_queries, 1):
                    st.markdown(f"{i}. {q}")

    elif submitted and not question.strip():
        st.warning("Please enter a research question.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: EVALUATION DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.title("📊 Evaluation Dashboard")
    st.markdown(
        "RAGAS-style metrics tracked across all research sessions. "
        "Higher scores = more faithful, relevant, and well-sourced answers."
    )

    if st.button("🔄 Refresh"):
        st.rerun()

    trends = get_eval_trends(limit=30)

    if not trends:
        st.info("No evaluation data yet. Run a research query to generate scores.")
    else:
        # Latest session metrics
        latest = trends[0]
        st.subheader("Latest Session Scores")
        cols = st.columns(6)
        metrics = [
            ("Faithfulness", "eval_faithfulness"),
            ("Relevancy", "eval_relevancy"),
            ("Precision", "eval_precision"),
            ("Recall", "eval_recall"),
            ("Coverage", "eval_coverage"),
            ("Overall", "eval_overall"),
        ]
        for col, (label, key) in zip(cols, metrics):
            val = latest.get(key) or 0
            col.metric(label, f"{val:.2f}", delta=None)

        st.markdown("---")
        st.subheader("Score Trends")

        # Build trend chart
        dates = [t["created_at"][:19] for t in reversed(trends)]
        fig = go.Figure()
        for label, key in metrics[:-1]:  # exclude overall from lines
            vals = [t.get(key) or 0 for t in reversed(trends)]
            fig.add_trace(go.Scatter(x=dates, y=vals, mode="lines+markers", name=label))

        # Overall as bar
        overall_vals = [t.get("eval_overall") or 0 for t in reversed(trends)]
        fig.add_trace(
            go.Bar(x=dates, y=overall_vals, name="Overall", opacity=0.3, yaxis="y")
        )

        fig.update_layout(
            yaxis=dict(range=[0, 1.05], title="Score"),
            xaxis_title="Session",
            legend=dict(orientation="h"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Score distribution radar for latest
        st.subheader("Latest Session — Metric Radar")
        radar_vals = [latest.get(k) or 0 for _, k in metrics[:-1]]
        radar_labels = [l for l, _ in metrics[:-1]]
        fig_r = go.Figure(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            line_color="royalblue",
        ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(range=[0, 1])),
            height=350,
        )
        st.plotly_chart(fig_r, use_container_width=True)

        # Score table
        st.subheader("All Sessions")
        table_data = []
        for t in trends:
            table_data.append({
                "Date": t["created_at"][:19],
                "Faithfulness": f"{t.get('eval_faithfulness') or 0:.2f}",
                "Relevancy": f"{t.get('eval_relevancy') or 0:.2f}",
                "Precision": f"{t.get('eval_precision') or 0:.2f}",
                "Recall": f"{t.get('eval_recall') or 0:.2f}",
                "Coverage": f"{t.get('eval_coverage') or 0:.2f}",
                "Overall ⭐": f"{t.get('eval_overall') or 0:.2f}",
            })
        st.dataframe(table_data, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_kb:
    st.title("📚 Knowledge Base")
    st.markdown("Documents stored in ChromaDB for local retrieval.")

    stats = get_ingestion_stats()
    col1, col2 = st.columns(2)
    col1.metric("Total Chunks", stats["total_chunks"])
    col2.metric("Unique Documents", stats["unique_sources"])

    if stats["file_types"]:
        st.subheader("File Types")
        fig_ft = px.pie(
            names=list(stats["file_types"].keys()),
            values=list(stats["file_types"].values()),
            title="Chunks by File Type",
        )
        st.plotly_chart(fig_ft, use_container_width=True)

    if stats["source_names"]:
        st.subheader("Indexed Documents")
        for src in stats["source_names"]:
            st.markdown(f"📄 `{src}`")
    else:
        st.info("No documents ingested yet. Upload files from the sidebar.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.title("🕒 Research History")

    if st.button("🔄 Refresh History"):
        st.rerun()

    try:
        sessions = get_all_sessions(limit=20)
    except Exception as exc:
        sessions = []
        st.error(f"Could not load sessions: {exc}")

    if not sessions:
        st.info("No sessions yet. Run a research query to get started!")
    else:
        for s in sessions:
            overall = s.get("eval_overall") or 0
            status = s.get("validation_status", "")
            status_icon = "✅" if status == "PASS" else "⚠️"
            score_color = "🟢" if overall >= 0.8 else "🟡" if overall >= 0.5 else "🔴"

            with st.expander(
                f"{status_icon} {s['question'][:70]}... | {s['created_at'][:19]} | {score_color} {overall:.2f}",
                expanded=False,
            ):
                # Sub-queries
                sq = s.get("sub_queries", [])
                if sq:
                    st.markdown("**Sub-queries:** " + " · ".join(f"`{q}`" for q in sq[:3]))

                # Scores
                score_cols = st.columns(5)
                for col, (label, key) in zip(
                    score_cols,
                    [("Faith.", "eval_faithfulness"), ("Relev.", "eval_relevancy"),
                     ("Prec.", "eval_precision"), ("Recall", "eval_recall"), ("Cover.", "eval_coverage")]
                ):
                    col.metric(label, f"{s.get(key) or 0:.2f}")

                # Answer
                st.markdown("**Final Answer:**")
                st.markdown(s.get("final_answer") or s.get("summary") or "_No answer stored._")

                # Claim checks summary
                checks = s.get("claim_checks", [])
                if checks:
                    supported = sum(1 for c in checks if c.get("verdict") == "SUPPORTED")
                    st.markdown(f"**Claims:** {supported}/{len(checks)} supported")

                # Sources
                sources = s.get("sources", [])
                if sources:
                    st.markdown("**Sources:** " + " · ".join(
                        f"[link]({src})" if src.startswith("http") else f"`{src}`"
                        for src in sources[:4]
                    ))
