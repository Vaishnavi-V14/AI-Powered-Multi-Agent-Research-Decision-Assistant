"""SQLite session storage — schema, inserts, and history queries."""

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)


def _get_connection() -> sqlite3.Connection:
    """Return a SQLite connection, creating the DB file and schema if needed."""
    settings = get_settings()
    db_path = Path(settings.database_url)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _create_schema(conn)
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    """Ensure the sessions table exists."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            question    TEXT NOT NULL,
            summary     TEXT,
            final_answer TEXT,
            sources     TEXT,       -- JSON array
            retry_count INTEGER DEFAULT 0,
            validation_status TEXT,
            created_at  TEXT NOT NULL
        )
        """
    )
    conn.commit()


def save_session(state: dict[str, Any]) -> str:
    """
    Persist a completed research session to SQLite.

    Args:
        state: The final LangGraph ResearchState dict.

    Returns:
        The UUID session ID.
    """
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    sources_json = json.dumps(state.get("sources", []))

    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO sessions
                (id, question, summary, final_answer, sources, retry_count, validation_status, created_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                state.get("question", ""),
                state.get("summary", ""),
                state.get("final_answer", ""),
                sources_json,
                state.get("retry_count", 0),
                state.get("validation_status", ""),
                now,
            ),
        )
        conn.commit()
        logger.info("Session %s saved to database.", session_id)
    except sqlite3.Error as exc:
        logger.error("Failed to save session: %s", exc)
        raise
    finally:
        conn.close()

    return session_id


def get_all_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent research sessions."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["sources"] = json.loads(d.get("sources") or "[]")
            result.append(d)
        return result
    finally:
        conn.close()


def get_session(session_id: str) -> dict[str, Any] | None:
    """Return a single session by ID, or None if not found."""
    conn = _get_connection()
    try:
        cursor = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["sources"] = json.loads(d.get("sources") or "[]")
        return d
    finally:
        conn.close()
