from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from harness.shadow.models import (
    HookRecord,
    ShadowAttempt,
    ShadowTask,
    canonical_json,
)
from harness.training.security import assert_no_secrets


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class ShadowLease:
    task: ShadowTask
    token: str
    attempt: int


class ShadowSpool:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.database = self.root / "spool.sqlite3"
        if self.database.is_symlink():
            raise ValueError("shadow spool database cannot be a symlink")
        if not self.database.exists():
            descriptor = os.open(
                self.database,
                os.O_RDWR | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            os.close(descriptor)
        info = self.database.stat()
        if (
            info.st_uid != os.geteuid()
            or not stat.S_ISREG(info.st_mode)
        ):
            raise PermissionError("shadow spool must be an owned regular file")
        os.chmod(self.database, 0o600)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS shadow_tasks (
                    task_id TEXT PRIMARY KEY,
                    correlation_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    generation_id TEXT,
                    repository_id TEXT NOT NULL,
                    task_json TEXT NOT NULL,
                    task_sha256 TEXT NOT NULL UNIQUE,
                    state_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL,
                    available_at TEXT NOT NULL,
                    lease_token TEXT,
                    lease_expires_at TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_shadow_tasks_queue
                    ON shadow_tasks(status, available_at, created_at);
                CREATE INDEX IF NOT EXISTS idx_shadow_tasks_correlation
                    ON shadow_tasks(repository_id, correlation_id, created_at);

                CREATE TABLE IF NOT EXISTS shadow_hook_events (
                    event_id TEXT PRIMARY KEY,
                    task_id TEXT,
                    correlation_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES shadow_tasks(task_id)
                );

                CREATE INDEX IF NOT EXISTS idx_shadow_events_task
                    ON shadow_hook_events(task_id, created_at);

                CREATE TABLE IF NOT EXISTS shadow_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL UNIQUE,
                    attempt_json TEXT NOT NULL,
                    attempt_sha256 TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES shadow_tasks(task_id)
                );

                CREATE TABLE IF NOT EXISTS shadow_replays (
                    replay_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    candidate_kind TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    report_sha256 TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    UNIQUE (task_id, candidate_kind, report_sha256),
                    FOREIGN KEY (task_id) REFERENCES shadow_tasks(task_id)
                );

                CREATE TABLE IF NOT EXISTS shadow_comparisons (
                    comparison_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL UNIQUE,
                    report_json TEXT NOT NULL,
                    report_sha256 TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES shadow_tasks(task_id)
                );

                CREATE TABLE IF NOT EXISTS shadow_processing (
                    task_id TEXT PRIMARY KEY,
                    comparison_id TEXT NOT NULL UNIQUE,
                    outcome TEXT NOT NULL
                        CHECK (outcome IN ('admitted', 'rejected')),
                    learning_event_id TEXT UNIQUE,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES shadow_tasks(task_id),
                    FOREIGN KEY (comparison_id)
                        REFERENCES shadow_comparisons(comparison_id)
                );

                CREATE TRIGGER IF NOT EXISTS shadow_hook_events_no_update
                BEFORE UPDATE ON shadow_hook_events
                BEGIN
                    SELECT RAISE(ABORT, 'shadow hook events are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS shadow_hook_events_no_delete
                BEFORE DELETE ON shadow_hook_events
                BEGIN
                    SELECT RAISE(ABORT, 'shadow hook events are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS shadow_attempts_no_update
                BEFORE UPDATE ON shadow_attempts
                BEGIN
                    SELECT RAISE(ABORT, 'shadow attempts are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS shadow_attempts_no_delete
                BEFORE DELETE ON shadow_attempts
                BEGIN
                    SELECT RAISE(ABORT, 'shadow attempts are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS shadow_replays_no_update
                BEFORE UPDATE ON shadow_replays
                BEGIN
                    SELECT RAISE(ABORT, 'shadow replays are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS shadow_replays_no_delete
                BEFORE DELETE ON shadow_replays
                BEGIN
                    SELECT RAISE(ABORT, 'shadow replays are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS shadow_comparisons_no_update
                BEFORE UPDATE ON shadow_comparisons
                BEGIN
                    SELECT RAISE(ABORT, 'shadow comparisons are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS shadow_comparisons_no_delete
                BEFORE DELETE ON shadow_comparisons
                BEGIN
                    SELECT RAISE(ABORT, 'shadow comparisons are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS shadow_processing_no_update
                BEFORE UPDATE ON shadow_processing
                BEGIN
                    SELECT RAISE(ABORT, 'shadow processing records are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS shadow_processing_no_delete
                BEFORE DELETE ON shadow_processing
                BEGIN
                    SELECT RAISE(ABORT, 'shadow processing records are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS shadow_tasks_no_delete
                BEFORE DELETE ON shadow_tasks
                BEGIN
                    SELECT RAISE(ABORT, 'shadow tasks are durable history');
                END;
                """
            )
        for sidecar in (
            self.database.with_name(self.database.name + "-wal"),
            self.database.with_name(self.database.name + "-shm"),
        ):
            if sidecar.exists():
                os.chmod(sidecar, 0o600)

    def enqueue(self, task: ShadowTask) -> bool:
        payload = canonical_json(task)
        digest = hashlib.sha256(payload).hexdigest()
        now = _timestamp(_now())
        with self.connect() as connection:
            current = connection.execute(
                "SELECT task_sha256 FROM shadow_tasks WHERE task_id = ?",
                (task.task_id,),
            ).fetchone()
            if current is not None:
                if current["task_sha256"] != digest:
                    raise ValueError("shadow task ID already has different content")
                return False
            connection.execute(
                """
                INSERT INTO shadow_tasks (
                    task_id, correlation_id, session_id, generation_id,
                    repository_id, task_json, task_sha256, state_sha256,
                    status, max_attempts, available_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.correlation_id,
                    task.session_id,
                    task.generation_id,
                    task.policy.repository_id,
                    payload.decode(),
                    digest,
                    task.snapshot.state_sha256,
                    task.policy.max_attempts,
                    now,
                    _timestamp(task.created_at),
                    now,
                ),
            )
        return True

    def append_hook(self, record: HookRecord) -> bool:
        payload = canonical_json(record.payload)
        if hashlib.sha256(payload).hexdigest() != record.payload_sha256:
            raise ValueError("hook payload digest mismatch")
        with self.connect() as connection:
            current = connection.execute(
                "SELECT payload_sha256 FROM shadow_hook_events WHERE event_id = ?",
                (record.event_id,),
            ).fetchone()
            if current is not None:
                if current["payload_sha256"] != record.payload_sha256:
                    raise ValueError("shadow hook event ID conflict")
                return False
            connection.execute(
                """
                INSERT INTO shadow_hook_events (
                    event_id, task_id, correlation_id, event_type,
                    payload_json, payload_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.event_id,
                    record.task_id,
                    record.correlation_id,
                    record.event_type,
                    payload.decode(),
                    record.payload_sha256,
                    _timestamp(record.created_at),
                ),
            )
        return True

    def task_for_correlation(
        self,
        *,
        repository_id: str,
        correlation_id: str,
    ) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT task_id FROM shadow_tasks
                WHERE repository_id = ? AND correlation_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (repository_id, correlation_id),
            ).fetchone()
        return str(row["task_id"]) if row is not None else None

    def claim(self, *, lease_seconds: int = 1800) -> ShadowLease | None:
        now_value = _now()
        now = _timestamp(now_value)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE shadow_tasks
                SET
                    status = CASE
                        WHEN attempt >= max_attempts THEN 'failed'
                        ELSE 'queued'
                    END,
                    lease_token = NULL,
                    lease_expires_at = NULL,
                    error = CASE
                        WHEN attempt >= max_attempts
                        THEN 'shadow lease expired at maximum attempts'
                        ELSE error
                    END,
                    updated_at = ?
                WHERE status = 'running' AND lease_expires_at <= ?
                """,
                (now, now),
            )
            row = connection.execute(
                """
                SELECT * FROM shadow_tasks
                WHERE status = 'queued' AND available_at <= ?
                ORDER BY created_at, task_id
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            expires = _timestamp(now_value + timedelta(seconds=lease_seconds))
            updated = connection.execute(
                """
                UPDATE shadow_tasks
                SET status = 'running', attempt = attempt + 1,
                    lease_token = ?, lease_expires_at = ?, updated_at = ?
                WHERE task_id = ? AND status = 'queued'
                """,
                (token, expires, now, row["task_id"]),
            )
            if updated.rowcount != 1:
                return None
            task = ShadowTask.model_validate_json(row["task_json"])
            return ShadowLease(task=task, token=token, attempt=int(row["attempt"]) + 1)

    def complete(self, lease: ShadowLease, attempt: ShadowAttempt) -> None:
        if attempt.task_id != lease.task.task_id:
            raise ValueError("attempt does not belong to claimed shadow task")
        payload = canonical_json(attempt)
        digest = hashlib.sha256(payload).hexdigest()
        now = _timestamp(_now())
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT status, lease_token FROM shadow_tasks WHERE task_id = ?
                """,
                (lease.task.task_id,),
            ).fetchone()
            if (
                row is None
                or row["status"] != "running"
                or row["lease_token"] != lease.token
            ):
                raise ValueError("shadow lease is no longer valid")
            connection.execute(
                """
                INSERT INTO shadow_attempts (
                    attempt_id, task_id, attempt_json,
                    attempt_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    attempt.attempt_id,
                    attempt.task_id,
                    payload.decode(),
                    digest,
                    _timestamp(attempt.created_at),
                ),
            )
            connection.execute(
                """
                UPDATE shadow_tasks
                SET status = ?, lease_token = NULL, lease_expires_at = NULL,
                    error = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (
                    attempt.status,
                    attempt.error,
                    now,
                    attempt.task_id,
                ),
            )

    def fail(
        self,
        lease: ShadowLease,
        error: str,
        *,
        retry_delay_seconds: int = 60,
        retryable: bool = True,
    ) -> str:
        assert_no_secrets(error, field="shadow failure")
        now_value = _now()
        now = _timestamp(now_value)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT status, lease_token, attempt, max_attempts
                FROM shadow_tasks WHERE task_id = ?
                """,
                (lease.task.task_id,),
            ).fetchone()
            if (
                row is None
                or row["status"] != "running"
                or row["lease_token"] != lease.token
            ):
                raise ValueError("shadow lease is no longer valid")
            should_retry = retryable and int(row["attempt"]) < int(row["max_attempts"])
            status = "queued" if should_retry else "failed"
            available = _timestamp(
                now_value + timedelta(seconds=max(0, retry_delay_seconds))
            )
            connection.execute(
                """
                UPDATE shadow_tasks
                SET status = ?, available_at = ?, lease_token = NULL,
                    lease_expires_at = NULL, error = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (status, available, error, now, lease.task.task_id),
            )
        return status

    def get_task(self, task_id: str) -> tuple[ShadowTask, dict[str, Any]]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM shadow_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return ShadowTask.model_validate_json(row["task_json"]), dict(row)

    def get_attempt(self, task_id: str) -> ShadowAttempt | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT attempt_json FROM shadow_attempts WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return (
            ShadowAttempt.model_validate_json(row["attempt_json"])
            if row is not None
            else None
        )

    def events(self, task_id: str) -> tuple[HookRecord, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM shadow_hook_events
                WHERE task_id = ? ORDER BY created_at, event_id
                """,
                (task_id,),
            ).fetchall()
        return tuple(
            HookRecord(
                event_id=row["event_id"],
                task_id=row["task_id"],
                correlation_id=row["correlation_id"],
                event_type=row["event_type"],
                payload=json.loads(row["payload_json"]),
                payload_sha256=row["payload_sha256"],
                created_at=row["created_at"],
            )
            for row in rows
        )

    def record_replay(
        self,
        *,
        replay_id: str,
        task_id: str,
        candidate_kind: str,
        report: dict[str, Any],
        report_sha256: str,
        created_at: datetime,
    ) -> bool:
        payload = canonical_json(report)
        if hashlib.sha256(payload).hexdigest() != report_sha256:
            raise ValueError("shadow replay report digest mismatch")
        with self.connect() as connection:
            current = connection.execute(
                "SELECT report_sha256 FROM shadow_replays WHERE replay_id = ?",
                (replay_id,),
            ).fetchone()
            if current is not None:
                if current["report_sha256"] != report_sha256:
                    raise ValueError("shadow replay ID already has different content")
                return False
            connection.execute(
                """
                INSERT INTO shadow_replays (
                    replay_id, task_id, candidate_kind,
                    report_json, report_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    replay_id,
                    task_id,
                    candidate_kind,
                    payload.decode(),
                    report_sha256,
                    _timestamp(created_at),
                ),
            )
        return True

    def replays(self, task_id: str) -> tuple[dict[str, Any], ...]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT report_json FROM shadow_replays
                WHERE task_id = ? ORDER BY created_at, replay_id
                """,
                (task_id,),
            ).fetchall()
        return tuple(json.loads(row["report_json"]) for row in rows)

    def record_comparison(
        self,
        *,
        comparison_id: str,
        task_id: str,
        report: dict[str, Any],
        report_sha256: str,
        created_at: datetime,
    ) -> bool:
        payload = canonical_json(report)
        if hashlib.sha256(payload).hexdigest() != report_sha256:
            raise ValueError("shadow comparison report digest mismatch")
        with self.connect() as connection:
            current = connection.execute(
                """
                SELECT report_sha256 FROM shadow_comparisons
                WHERE comparison_id = ?
                """,
                (comparison_id,),
            ).fetchone()
            if current is not None:
                if current["report_sha256"] != report_sha256:
                    raise ValueError(
                        "shadow comparison ID already has different content"
                    )
                return False
            connection.execute(
                """
                INSERT INTO shadow_comparisons (
                    comparison_id, task_id, report_json,
                    report_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    comparison_id,
                    task_id,
                    payload.decode(),
                    report_sha256,
                    _timestamp(created_at),
                ),
            )
        return True

    def comparison(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT report_json FROM shadow_comparisons WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return json.loads(row["report_json"]) if row is not None else None

    def processable_tasks(self, *, limit: int = 20) -> tuple[str, ...]:
        if limit < 1:
            raise ValueError("processable task limit must be positive")
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT task.task_id
                FROM shadow_tasks AS task
                JOIN shadow_attempts AS attempt
                  ON attempt.task_id = task.task_id
                JOIN shadow_hook_events AS event
                  ON event.task_id = task.task_id
                 AND event.event_type = 'stop'
                LEFT JOIN shadow_processing AS processing
                  ON processing.task_id = task.task_id
                WHERE task.status IN ('completed', 'quarantined')
                  AND processing.task_id IS NULL
                ORDER BY task.created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(row["task_id"] for row in rows)

    def record_processing(
        self,
        *,
        task_id: str,
        comparison_id: str,
        outcome: str,
        learning_event_id: str | None,
        created_at: datetime,
    ) -> bool:
        if outcome not in {"admitted", "rejected"}:
            raise ValueError("invalid shadow processing outcome")
        if (outcome == "admitted") != (learning_event_id is not None):
            raise ValueError("only admitted processing requires a learning event")
        with self.connect() as connection:
            current = connection.execute(
                "SELECT * FROM shadow_processing WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            expected = (comparison_id, outcome, learning_event_id)
            if current is not None:
                actual = (
                    current["comparison_id"],
                    current["outcome"],
                    current["learning_event_id"],
                )
                if actual != expected:
                    raise ValueError(
                        "shadow task already has a different processing result"
                    )
                return False
            connection.execute(
                """
                INSERT INTO shadow_processing (
                    task_id, comparison_id, outcome,
                    learning_event_id, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    comparison_id,
                    outcome,
                    learning_event_id,
                    _timestamp(created_at),
                ),
            )
        return True

    def status(self) -> dict[str, int]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM shadow_tasks GROUP BY status ORDER BY status
                """
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}
