"""
FastAPI REST API for the AI Research Assistant.
Provides programmatic access alongside the Streamlit UI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.graph import run_research
from src.database import get_all_sessions, get_session

logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Research Assistant API",
    description=(
        "Multi-agent research pipeline built with LangChain, LangGraph, and Ollama. "
        "Submit a question and receive a verified, source-cited answer."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=5, description="The research question.")


class ResearchResponse(BaseModel):
    session_id: str
    question: str
    final_answer: str
    summary: str
    sources: list[str]
    validation_status: str
    retry_count: int


class SessionListResponse(BaseModel):
    sessions: list[dict[str, Any]]
    total: int


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "AI Research Assistant"}


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/research", response_model=ResearchResponse, tags=["Research"])
async def research(request: ResearchRequest):
    """
    Run the full multi-agent research pipeline for the given question.

    This is a synchronous endpoint — it blocks until all agents complete.
    For large questions this may take 30–90 seconds.
    """
    try:
        logger.info("API: Starting research for '%s'", request.question)
        result = run_research(request.question)
        return ResearchResponse(
            session_id=result.get("session_id", ""),
            question=result.get("question", request.question),
            final_answer=result.get("final_answer", ""),
            summary=result.get("summary", ""),
            sources=result.get("sources", []),
            validation_status=result.get("validation_status", ""),
            retry_count=result.get("retry_count", 0),
        )
    except Exception as exc:
        logger.error("API research error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/sessions", response_model=SessionListResponse, tags=["Sessions"])
async def list_sessions(limit: int = 20):
    """Return the most recent research sessions."""
    try:
        sessions = get_all_sessions(limit=limit)
        return SessionListResponse(sessions=sessions, total=len(sessions))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/sessions/{session_id}", tags=["Sessions"])
async def get_session_detail(session_id: str):
    """Return details of a single research session."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session
