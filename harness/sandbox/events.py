from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .models import SandboxRecord, format_time


@dataclass(frozen=True)
class SandboxEvent:
    event_id: int
    sandbox_id: str
    kind: str
    state: str
    detail: str | None
    created_at: str


class SandboxEventStore:
    """Append-only SQLite lifecycle evidence for sandbox operations."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            raise ValueError("sandbox event database path must be absolute")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def append(self, kind: str, record: SandboxRecord) -> SandboxEvent:
        if not kind or len(kind) > 64:
            raise ValueError("sandbox event kind is invalid")
        created_at = format_time(record.updated_at)
        assert created_at is not None
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO sandbox_events(
                    sandbox_id, kind, state, detail, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.sandbox_id,
                    kind,
                    record.state.value,
                    record.detail,
                    created_at,
                ),
            )
            event_id = int(cursor.lastrowid)
        return SandboxEvent(
            event_id=event_id,
            sandbox_id=record.sandbox_id,
            kind=kind,
            state=record.state.value,
            detail=record.detail,
            created_at=created_at,
        )

    def list(
        self,
        *,
        sandbox_id: str | None = None,
        limit: int = 200,
    ) -> tuple[SandboxEvent, ...]:
        if not 1 <= limit <= 10_000:
            raise ValueError("event limit must be between 1 and 10000")
        query = (
            "SELECT id, sandbox_id, kind, state, detail, created_at "
            "FROM sandbox_events"
        )
        parameters: tuple[object, ...]
        if sandbox_id is None:
            query += " ORDER BY id DESC LIMIT ?"
            parameters = (limit,)
        else:
            query += " WHERE sandbox_id=? ORDER BY id DESC LIMIT ?"
            parameters = (sandbox_id, limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(
            SandboxEvent(
                event_id=int(row[0]),
                sandbox_id=str(row[1]),
                kind=str(row[2]),
                state=str(row[3]),
                detail=str(row[4]) if row[4] is not None else None,
                created_at=str(row[5]),
            )
            for row in rows
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sandbox_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sandbox_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    state TEXT NOT NULL,
                    detail TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sandbox_events_id
                ON sandbox_events(sandbox_id, id)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection
