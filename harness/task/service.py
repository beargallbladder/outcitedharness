from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from harness.storage.db import Store, utcnow
from harness.task.context import ContextManager
from harness.task.models import AttemptRecord, Decision, Evidence, Task, WorkPacket

CLINE_INTENT = "cline session"
SESSION_IDLE = timedelta(minutes=30)


def _new_id() -> str:
    stamp = utcnow().replace("-", "").replace(":", "")
    return f"t{stamp}_{secrets.token_hex(2)}"


def _parse_ts(value: str) -> datetime:
    raw = (value or "").replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class TaskService:
    store: Store

    def start(self, intent: str, plan: str = "", hypothesis: str = "") -> Task:
        task = Task(
            task_id=_new_id(),
            intent=intent.strip(),
            status="open",
            created_at=utcnow(),
            plan=plan,
            hypothesis=hypothesis,
        )
        if not task.intent:
            raise ValueError("intent required")
        with self.store.connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (task_id, intent, status, created_at, plan, hypothesis, intervened, frontier_required)
                VALUES (?, ?, ?, ?, ?, ?, 0, 0)
                """,
                (task.task_id, task.intent, task.status, task.created_at, task.plan, task.hypothesis),
            )
        return task

    def get(self, task_id: str) -> Task:
        with self.store.connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if not row:
            raise KeyError(task_id)
        return Task(
            task_id=row["task_id"],
            intent=row["intent"],
            status=row["status"],
            created_at=row["created_at"],
            plan=row["plan"] or "",
            hypothesis=row["hypothesis"] or "",
            intervened=bool(row["intervened"]),
            frontier_required=bool(row["frontier_required"]),
        )

    def list_tasks(self, limit: int = 20) -> list[Task]:
        with self.store.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self.get(r["task_id"]) for r in rows]

    def latest_session(self) -> Task | None:
        with self.store.connect() as conn:
            row = conn.execute(
                """
                SELECT task_id FROM tasks
                WHERE intent = ? AND status = 'open'
                ORDER BY created_at DESC, rowid DESC LIMIT 1
                """,
                (CLINE_INTENT,),
            ).fetchone()
        if not row:
            return None
        return self.get(row["task_id"])

    def session_task(self) -> Task:
        """Reuse the open Cline session only if it had a turn in the last 30 minutes."""
        current = self.latest_session()
        if current:
            turns = self.attempts(current.task_id)
            last = turns[-1].finished_at if turns else current.created_at
            if last and datetime.now(timezone.utc) - _parse_ts(last) <= SESSION_IDLE:
                return current
        return self.start(CLINE_INTENT)

    def record_turn(self, rec: AttemptRecord) -> AttemptRecord:
        """Append a Cline turn. Keep the session open; do not mark the job done."""
        saved = self.record(rec, close=False)
        with self.store.connect() as conn:
            conn.execute(
                "UPDATE tasks SET status = 'open' WHERE task_id = ?",
                (saved.task_id,),
            )
        return saved

    def context_from_task(self, task: Task) -> ContextManager:
        decisions = [d.text for d in self.decisions(task.task_id) if d.accepted]
        latest = self.attempts(task.task_id)
        files: list[str] = []
        failed = ""
        if latest:
            files = latest[-1].files_changed
            if latest[-1].tests_failed:
                failed = f"tests failed={latest[-1].tests_failed} passed={latest[-1].tests_passed}"
        return ContextManager(
            intent=task.intent,
            plan=task.plan,
            files=files,
            failed_tests=failed,
            hypothesis=task.hypothesis,
            decisions=decisions,
        )

    def packet(self, task_id: str, worker: str) -> WorkPacket:
        task = self.get(task_id)
        return self.context_from_task(task).packet(task_id, worker)

    def record(self, rec: AttemptRecord, close: bool = True) -> AttemptRecord:
        task = self.get(rec.task_id)
        with self.store.connect() as conn:
            next_n = conn.execute(
                "SELECT COALESCE(MAX(attempt), 0) + 1 FROM attempts WHERE task_id = ?",
                (rec.task_id,),
            ).fetchone()[0]
            rec.attempt = int(next_n)
            rec.started_at = rec.started_at or utcnow()
            rec.finished_at = rec.finished_at or utcnow()
            conn.execute(
                """
                INSERT INTO attempts (
                    task_id, attempt, worker, started_at, finished_at, result,
                    files_changed, commands, tests_passed, tests_failed,
                    ttft_ms, tokens_per_sec, tool_calls, input_tokens, output_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.task_id,
                    rec.attempt,
                    rec.worker,
                    rec.started_at,
                    rec.finished_at,
                    rec.result,
                    json.dumps(rec.files_changed),
                    json.dumps(rec.commands),
                    rec.tests_passed,
                    rec.tests_failed,
                    rec.ttft_ms,
                    rec.tokens_per_sec,
                    rec.tool_calls,
                    rec.input_tokens,
                    rec.output_tokens,
                ),
            )
            if close:
                status = "success" if rec.result == "success" else "failed"
                conn.execute("UPDATE tasks SET status = ? WHERE task_id = ?", (status, rec.task_id))
                task.status = status
        self.add_evidence(
            Evidence(task_id=rec.task_id, attempt=rec.attempt, kind="attempt", payload=rec.to_evidence_json())
        )
        return rec

    def add_evidence(self, item: Evidence) -> None:
        item.created_at = item.created_at or utcnow()
        with self.store.connect() as conn:
            conn.execute(
                "INSERT INTO evidence (task_id, attempt, kind, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                (item.task_id, item.attempt, item.kind, json.dumps(item.payload), item.created_at),
            )

    def add_decision(self, item: Decision) -> None:
        item.created_at = item.created_at or utcnow()
        with self.store.connect() as conn:
            conn.execute(
                "INSERT INTO decisions (task_id, actor, text, accepted, created_at) VALUES (?, ?, ?, ?, ?)",
                (item.task_id, item.actor, item.text, int(item.accepted), item.created_at),
            )

    def attempts(self, task_id: str) -> list[AttemptRecord]:
        with self.store.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM attempts WHERE task_id = ? ORDER BY attempt",
                (task_id,),
            ).fetchall()
        out: list[AttemptRecord] = []
        for row in rows:
            out.append(
                AttemptRecord(
                    task_id=row["task_id"],
                    attempt=row["attempt"],
                    worker=row["worker"],
                    result=row["result"] or "",
                    files_changed=json.loads(row["files_changed"] or "[]"),
                    commands=json.loads(row["commands"] or "[]"),
                    tests_passed=row["tests_passed"],
                    tests_failed=row["tests_failed"],
                    ttft_ms=row["ttft_ms"],
                    tokens_per_sec=row["tokens_per_sec"],
                    tool_calls=row["tool_calls"],
                    input_tokens=row["input_tokens"],
                    output_tokens=row["output_tokens"],
                    started_at=row["started_at"] or "",
                    finished_at=row["finished_at"] or "",
                )
            )
        return out

    def decisions(self, task_id: str) -> list[Decision]:
        with self.store.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE task_id = ? ORDER BY id",
                (task_id,),
            ).fetchall()
        return [
            Decision(
                task_id=r["task_id"],
                actor=r["actor"],
                text=r["text"],
                accepted=bool(r["accepted"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]
