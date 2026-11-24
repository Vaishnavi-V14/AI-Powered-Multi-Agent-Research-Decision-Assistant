"""
Streamlit frontend for the AI-Powered Multi-Agent Research Assistant.
"""

import sys
import os
from pathlib import Path

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from langchain_core.documents import Document

from src.graph import stream_research
from src.database import get_all_sessions
from src.vectorstore import ingest_documents
from src.config import get_settings

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

cfg = get_settings()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    st.subheader("🤖 LLM")
    st.info(f"Model: `{cfg.ollama_model}`\nOllama: `{cfg.ollama_base_url}`")

    st.subheader("📄 Upload Local Docs")
    uploaded_files = st.file_uploader(
        "Upload .txt or .md files to add to the knowledge base",
        accept_multiple_files=True,
        type=["txt", "md"],
    )

    if uploaded_files and st.button("📥 Ingest Documents"):
        docs = []
        for f in uploaded_files:
            content = f.read().decode("utf-8", errors="ignore")
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": f.name},
                )
            )
        with st.spinner("Ingesting documents into ChromaDB..."):
            count = ingest_documents(docs)
        st.success(f"✅ Ingested {count} document(s)!")

    st.markdown("---")
    st.caption("Built with LangChain · LangGraph · Ollama · ChromaDB")

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_research, tab_history = st.tabs(["🔍 Research", "📚 History"])

# ── Research Tab ─────────────────────────────────────────────────────────────
with tab_research:
    st.title("🧠 AI-Powered Multi-Agent Research Assistant")
    st.markdown(
        "Ask any research question. Five specialized AI agents will search the web, "
        "filter facts, summarize, fact-check, and deliver a verified answer."
    )

    with st.form("research_form"):
        question = st.text_area(
            "🔎 Your Research Question",
            placeholder='e.g. "How is AI used in healthcare?"',
            height=100,
        )
        submitted = st.form_submit_button("🚀 Research", use_container_width=True)

    if submitted and question.strip():
        st.markdown("---")

        # Agent status tracker
        agent_status = {
            "searcher": "⏳",
            "retriever": "⏳",
            "summarizer": "⏳",
            "critic": "⏳",
            "finalizer": "⏳",
        }

        AGENT_LABELS = {
            "searcher": "🌐 Searcher — Fetching web & local docs",
            "retriever": "🔍 Retriever — Filtering relevant facts",
            "summarizer": "📝 Summarizer — Compressing key points",
            "critic": "🧐 Critic — Fact-checking for hallucinations",
            "finalizer": "✅ Finalizer — Synthesizing final answer",
        }

        col_status, col_result = st.columns([1, 2])

        with col_status:
            st.subheader("🤖 Agent Pipeline")
            status_containers = {
                name: st.empty() for name in agent_status
            }
            for name, container in status_containers.items():
                container.markdown(f"⏳ {AGENT_LABELS[name]}")

        with col_result:
            st.subheader("📊 Live Updates")
            live_area = st.empty()

        # Stream the research pipeline
        final_state = {}
        retry_info = st.empty()

        try:
            for node_name, node_output in stream_research(question):
                # Update agent status
                for name in agent_status:
                    if name == node_name:
                        agent_status[name] = "✅"
                        status_containers[name].markdown(
                            f"✅ **{AGENT_LABELS[name]}**"
                        )
                    elif agent_status[name] == "⏳":
                        status_containers[name].markdown(
                            f"⏳ {AGENT_LABELS[name]}"
                        )

                # Show live partial results
                final_state.update(node_output)

                if node_name == "searcher":
                    n = len(node_output.get("raw_data", []))
                    live_area.info(f"🌐 Found **{n}** raw results.")

                elif node_name == "retriever":
                    facts = node_output.get("facts", [])
                    live_area.info(
                        f"🔍 Filtered to **{len(facts)}** relevant facts.\n\n"
                        + "\n".join(f"- {f}" for f in facts[:5])
                        + ("\n..." if len(facts) > 5 else "")
                    )

                elif node_name == "summarizer":
                    summary = node_output.get("summary", "")
                    live_area.info(f"📝 **Summary drafted** ({len(summary)} chars)")

                elif node_name == "critic":
                    status = node_output.get("validation_status", "")
                    notes = node_output.get("critique_notes", "")
                    retries = node_output.get("retry_count", 0)
                    if status == "PASS":
                        live_area.success("🧐 Critic: **PASS** ✅")
                    else:
                        live_area.warning(
                            f"🧐 Critic: **FAIL** — Retrying (attempt {retries})\n\n{notes}"
                        )

                elif node_name == "finalizer":
                    live_area.success("✅ Research complete!")

        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

        # ── Final Answer ──────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Final Answer")

        final_answer = final_state.get("final_answer", "")
        if final_answer:
            st.markdown(final_answer)
        else:
            st.warning("No answer was generated.")

        # ── Sources ───────────────────────────────────────────────────────────
        sources = final_state.get("sources", [])
        if sources:
            with st.expander("🔗 Sources Used", expanded=False):
                for i, s in enumerate(sources, 1):
                    if s.startswith("http"):
                        st.markdown(f"{i}. [{s}]({s})")
                    else:
                        st.markdown(f"{i}. 📄 `{s}`")

        # ── Session info ──────────────────────────────────────────────────────
        session_id = final_state.get("session_id", "")
        retry_count = final_state.get("retry_count", 0)
        validation = final_state.get("validation_status", "")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Session ID", session_id[:8] + "..." if session_id else "N/A")
        col_b.metric("Critic Retries", retry_count)
        col_c.metric("Validation", validation)

    elif submitted and not question.strip():
        st.warning("Please enter a research question.")

# ── History Tab ───────────────────────────────────────────────────────────────
with tab_history:
    st.title("📚 Research History")
    st.markdown("Past research sessions stored in SQLite.")

    if st.button("🔄 Refresh"):
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
            with st.expander(
                f"🔎 {s['question'][:80]}... | {s['created_at'][:19]}", expanded=False
            ):
                st.markdown(f"**Status:** `{s['validation_status']}` | **Retries:** {s['retry_count']}")
                st.markdown("**Final Answer:**")
                st.markdown(s.get("final_answer") or s.get("summary") or "_No answer stored._")
                srcs = s.get("sources", [])
                if srcs:
                    st.markdown("**Sources:** " + " | ".join(srcs[:3]))
