"""SQLite-backed memory store for conversation history, summaries, and facts."""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from companion.config import MemoryConfig

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant", "system"]


@dataclass
class Message:
    """A single conversation message."""

    id: int
    role: Role
    content: str
    timestamp: str


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    role      TEXT    NOT NULL CHECK(role IN ('user','assistant','system')),
    content   TEXT    NOT NULL,
    timestamp TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS summaries (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    content   TEXT    NOT NULL,
    timestamp TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS facts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    key       TEXT    NOT NULL UNIQUE,
    value     TEXT    NOT NULL,
    updated   TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


class MemoryStore:
    """Persistent conversation memory backed by SQLite.

    Stores messages, rolling summaries, and user facts.  Thread-safe for
    concurrent reads/writes from the summarisation background thread.
    """

    def __init__(self, config: MemoryConfig) -> None:
        db_path = Path(config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(db_path), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)
        self._max_context = config.max_context_messages
        self._summary_threshold = config.summary_threshold
        self._lock = threading.Lock()
        logger.info("Memory store opened at %s", db_path)

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def add_message(self, role: Role, content: str) -> None:
        """Persist a conversation message."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (role, content) VALUES (?, ?)",
                (role, content),
            )
            self._conn.commit()

    def get_recent_messages(self, n: int | None = None) -> list[Message]:
        """Return the most recent *n* messages (oldest first).

        If *n* is ``None``, defaults to ``max_context_messages``.
        """
        limit = n if n is not None else self._max_context
        rows = self._conn.execute(
            "SELECT id, role, content, timestamp FROM messages "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            Message(id=r["id"], role=r["role"], content=r["content"], timestamp=r["timestamp"])
            for r in reversed(rows)
        ]

    def get_total_message_count(self) -> int:
        """Return the total number of stored messages."""
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM messages").fetchone()
        return row["cnt"]

    def get_messages_for_summary(self, keep_recent: int) -> list[Message]:
        """Return all messages EXCEPT the most recent *keep_recent*.

        These are the older messages that should be summarised and then deleted.
        Returns oldest-first ordering.
        """
        rows = self._conn.execute(
            "SELECT id, role, content, timestamp FROM messages "
            "WHERE id NOT IN ("
            "  SELECT id FROM messages ORDER BY id DESC LIMIT ?"
            ") ORDER BY id ASC",
            (keep_recent,),
        ).fetchall()
        return [
            Message(id=r["id"], role=r["role"], content=r["content"], timestamp=r["timestamp"])
            for r in rows
        ]

    def delete_messages_up_to(self, max_id: int) -> None:
        """Delete messages with ``id <= max_id`` (only the summarised ones)."""
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE id <= ?", (max_id,))
            self._conn.commit()
        logger.info("Deleted messages up to id=%d after summarisation.", max_id)

    def clear_messages(self) -> None:
        """Delete all messages (typically after summarisation)."""
        with self._lock:
            self._conn.execute("DELETE FROM messages")
            self._conn.commit()
        logger.info("Message history cleared after summarisation.")

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def add_summary(self, content: str) -> None:
        """Store a new conversation summary and remove older ones."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO summaries (content) VALUES (?)", (content,)
            )
            # Keep only the latest summary to avoid unbounded growth.
            self._conn.execute(
                "DELETE FROM summaries WHERE id NOT IN ("
                "  SELECT id FROM summaries ORDER BY id DESC LIMIT 1"
                ")"
            )
            self._conn.commit()
        logger.info("New conversation summary saved.")

    def get_latest_summary(self) -> str | None:
        """Return the most recent summary text, or ``None``."""
        row = self._conn.execute(
            "SELECT content FROM summaries ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["content"] if row else None

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    def set_fact(self, key: str, value: str) -> None:
        """Insert or update a user fact."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO facts (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated = datetime('now')",
                (key, value),
            )
            self._conn.commit()

    def get_fact(self, key: str) -> str | None:
        """Look up a single fact by key."""
        row = self._conn.execute(
            "SELECT value FROM facts WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def get_all_facts(self) -> dict[str, str]:
        """Return every stored fact as a ``{key: value}`` dict."""
        rows = self._conn.execute("SELECT key, value FROM facts").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def delete_fact(self, key: str) -> None:
        """Remove a fact by key."""
        with self._lock:
            self._conn.execute("DELETE FROM facts WHERE key = ?", (key,))
            self._conn.commit()

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def build_context(self, system_prompt: str) -> list[dict]:
        """Assemble the message list sent to the LLM.

        Order:
        1. System prompt
        2. Previous conversation summary (if any)
        3. Stored user facts (if any)
        4. Recent messages (oldest first)
        """
        context: list[dict] = [{"role": "system", "content": system_prompt}]

        summary = self.get_latest_summary()
        if summary:
            context.append({
                "role": "system",
                "content": f"[Summary of earlier conversation]: {summary}",
            })

        facts = self.get_all_facts()
        if facts:
            lines = [f"- {k}: {v}" for k, v in facts.items()]
            context.append({
                "role": "system",
                "content": "[Things to remember about the user]:\n" + "\n".join(lines),
            })

        for msg in self.get_recent_messages():
            context.append({"role": msg.role, "content": msg.content})

        return context

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    @property
    def should_summarise(self) -> bool:
        """Return ``True`` when the message count exceeds the summary threshold."""
        return self.get_total_message_count() >= self._summary_threshold

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.info("Memory store closed.")
