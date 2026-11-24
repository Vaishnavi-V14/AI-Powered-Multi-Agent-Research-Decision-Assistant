"""SQLite session storage — upgraded to persist eval_scores and claim_checks."""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)


def _get_connection() -> sqlite3.Connection:
    settings = get_settings()
    db_path = Path(settings.database_url)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _create_schema(conn)
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id                TEXT PRIMARY KEY,
            question          TEXT NOT NULL,
            sub_queries       TEXT,           -- JSON array
            summary           TEXT,
            final_answer      TEXT,
            sources           TEXT,           -- JSON array
            verified_sources  TEXT,           -- JSON array of SourceVerification
            claim_checks      TEXT,           -- JSON array of ClaimCheck
            retry_count       INTEGER DEFAULT 0,
            validation_status TEXT,
            eval_faithfulness REAL,
            eval_relevancy    REAL,
            eval_precision    REAL,
            eval_recall       REAL,
            eval_coverage     REAL,
            eval_overall      REAL,
            created_at        TEXT NOT NULL
        )
        """
    )
    conn.commit()


def save_session(state: dict[str, Any]) -> str:
    """Persist a completed research session. Returns UUID session ID."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    eval_scores = state.get("eval_scores") or {}

    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO sessions (
                id, question, sub_queries, summary, final_answer,
                sources, verified_sources, claim_checks,
                retry_count, validation_status,
                eval_faithfulness, eval_relevancy, eval_precision,
                eval_recall, eval_coverage, eval_overall,
                created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                session_id,
                state.get("question", ""),
                json.dumps(state.get("sub_queries", [])),
                state.get("summary", ""),
                state.get("final_answer", ""),
                json.dumps(state.get("sources", [])),
                json.dumps(state.get("verified_sources", [])),
                json.dumps(state.get("claim_checks", [])),
                state.get("retry_count", 0),
                state.get("validation_status", ""),
                eval_scores.get("faithfulness"),
                eval_scores.get("answer_relevancy"),
                eval_scores.get("context_precision"),
                eval_scores.get("context_recall"),
                eval_scores.get("source_coverage"),
                eval_scores.get("overall_score"),
                now,
            ),
        )
        conn.commit()
        logger.info("Session %s saved.", session_id)
    except sqlite3.Error as exc:
        logger.error("Failed to save session: %s", exc)
        raise
    finally:
        conn.close()

    return session_id


def get_all_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """Return most recent sessions with parsed JSON fields."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            for field in ("sub_queries", "sources", "verified_sources", "claim_checks"):
                raw = d.get(field)
                d[field] = json.loads(raw) if raw else []
            result.append(d)
        return result
    finally:
        conn.close()


def get_session(session_id: str) -> dict[str, Any] | None:
    """Return single session by ID with parsed JSON fields."""
    conn = _get_connection()
    try:
        cursor = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        d = dict(row)
        for field in ("sub_queries", "sources", "verified_sources", "claim_checks"):
            raw = d.get(field)
            d[field] = json.loads(raw) if raw else []
        return d
    finally:
        conn.close()


def get_eval_trends(limit: int = 30) -> list[dict[str, Any]]:
    """Return evaluation score trends for charting."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT created_at, eval_faithfulness, eval_relevancy,
                   eval_precision, eval_recall, eval_coverage, eval_overall
            FROM sessions
            WHERE eval_overall IS NOT NULL
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
