from __future__ import annotations

import json
import os
import sqlite3
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.storage.schema import SCHEMA_SQL


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RunRecord:
    run_id: str
    mode: str
    started_at: str
    finished_at: str | None
    case_count: int
    notes: str | None


class Store:
    def __init__(self, db_path: Path):
        self.db_path = db_path.expanduser().resolve(strict=False)
        if db_path.is_symlink():
            raise ValueError("database path cannot be a symlink")
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.db_path.parent.stat().st_uid != os.geteuid():
            raise PermissionError("database directory must be owned by this process")
        os.chmod(self.db_path.parent, 0o700)
        if not self.db_path.exists():
            descriptor = os.open(
                self.db_path,
                os.O_RDWR | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            os.close(descriptor)
        database_stat = self.db_path.stat()
        if database_stat.st_uid != os.geteuid():
            raise PermissionError("database must be owned by this process")
        if not stat.S_ISREG(database_stat.st_mode):
            raise ValueError("database path must be a regular file")
        os.chmod(self.db_path, 0o600)
        self._init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA recursive_triggers = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA trusted_schema = OFF")
        return conn

    def _init(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            self._ensure_column(conn, "gateway_turns", "task_id", "TEXT")
            self._ensure_column(conn, "tasks", "stage", "TEXT NOT NULL DEFAULT 'new'")
            self._ensure_column(conn, "tasks", "frontier_calls", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "tasks", "updated_at", "TEXT")
            self._ensure_column(conn, "tasks", "final_outcome", "TEXT")
            self._ensure_column(conn, "attempts", "estimated_cost", "REAL")
            self._ensure_column(conn, "training_jobs", "lease_token", "TEXT")
            self._ensure_column(
                conn,
                "training_jobs",
                "experiment_sha256",
                "TEXT",
            )
            self._ensure_column(conn, "training_jobs", "handler_pid", "INTEGER")
            self._ensure_column(conn, "training_jobs", "handler_pgid", "INTEGER")
            self._ensure_column(conn, "training_jobs", "handler_started_at", "TEXT")
            self._ensure_column(conn, "dataset_members", "repository_id", "TEXT")
            self._ensure_column(
                conn,
                "dataset_members",
                "component_family",
                "TEXT",
            )
            self._ensure_column(conn, "dataset_members", "temporal_bucket", "TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_gateway_turns_task ON gateway_turns(task_id)"
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_training_jobs_experiment
                ON training_jobs (
                    dataset_version_id, job_kind, experiment_sha256
                )
                WHERE experiment_sha256 IS NOT NULL
                """
            )
            conn.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS training_jobs_birth_guard_v1
                BEFORE INSERT ON training_jobs
                WHEN NEW.state != 'eligible'
                  OR NEW.attempt != 0
                  OR NEW.assigned_node IS NOT NULL
                  OR NEW.lease_expires_at IS NOT NULL
                  OR NEW.lease_token IS NOT NULL
                  OR NEW.handler_pid IS NOT NULL
                  OR NEW.handler_pgid IS NOT NULL
                  OR NEW.handler_started_at IS NOT NULL
                  OR NEW.checkpoint_uri IS NOT NULL
                  OR NEW.checkpoint_sha256 IS NOT NULL
                  OR NEW.priority <= 0
                  OR NEW.max_attempts < 1
                  OR NEW.dataset_version_id IS NULL
                  OR NEW.experiment_sha256 IS NULL
                  OR length(NEW.experiment_sha256) != 64
                  OR NEW.experiment_sha256 GLOB '*[^0-9a-f]*'
                  OR NOT EXISTS (
                    SELECT 1 FROM dataset_versions
                    WHERE dataset_version_id = NEW.dataset_version_id
                      AND state = 'eligible'
                  )
                  OR NOT EXISTS (
                    SELECT 1 FROM dataset_members
                    WHERE dataset_version_id = NEW.dataset_version_id
                      AND split = 'train'
                  )
                  OR EXISTS (
                    SELECT 1
                    FROM dataset_members AS dm
                    LEFT JOIN learning_admissions AS ad
                      ON ad.event_id = dm.event_id
                    LEFT JOIN learning_verifications AS v
                      ON v.verification_id = ad.verification_id
                    WHERE dm.dataset_version_id = NEW.dataset_version_id
                      AND (
                        ad.decision IS NOT 'eligible'
                        OR v.status IS NOT 'pass'
                      )
                  )
                BEGIN
                    SELECT RAISE(ABORT, 'invalid training job birth');
                END;

                CREATE TRIGGER IF NOT EXISTS training_jobs_no_delete_v1
                BEFORE DELETE ON training_jobs
                BEGIN
                    SELECT RAISE(ABORT, 'training jobs are immutable history');
                END;

                CREATE TRIGGER IF NOT EXISTS training_jobs_experiment_immutable_v1
                BEFORE UPDATE OF experiment_sha256 ON training_jobs
                WHEN NEW.experiment_sha256 IS NOT OLD.experiment_sha256
                BEGIN
                    SELECT RAISE(ABORT, 'training experiment identity is immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS training_jobs_process_guard_v1
                BEFORE UPDATE ON training_jobs
                WHEN (
                    (NEW.handler_pid IS NULL) != (NEW.handler_pgid IS NULL)
                    OR (NEW.handler_pid IS NULL) != (NEW.handler_started_at IS NULL)
                    OR (
                        NEW.handler_pid IS NOT NULL
                        AND (NEW.handler_pid <= 1 OR NEW.handler_pgid <= 1)
                    )
                    OR (
                        NEW.state != 'assigned'
                        AND (
                            NEW.handler_pid IS NOT NULL
                            OR NEW.handler_pgid IS NOT NULL
                            OR NEW.handler_started_at IS NOT NULL
                        )
                    )
                )
                BEGIN
                    SELECT RAISE(ABORT, 'training handler fields do not match state');
                END;
                """
            )

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        names = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in names:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def create_run(self, run_id: str, mode: str, notes: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, mode, started_at, case_count, notes)
                VALUES (?, ?, ?, 0, ?)
                """,
                (run_id, mode, utcnow(), notes),
            )

    def finish_run(self, run_id: str, case_count: int) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET finished_at = ?, case_count = ?
                WHERE run_id = ?
                """,
                (utcnow(), case_count, run_id),
            )

    def insert_case_run(self, payload: dict[str, Any]) -> None:
        columns = [
            "run_id",
            "case_id",
            "mode",
            "minimum_model_that_solved",
            "successful_tier",
            "total_escalation_latency_ms",
            "total_escalation_cost",
            "wasted_latency_before_success_ms",
            "wasted_cost_before_success",
            "failed_tiers",
            "started_at",
            "finished_at",
        ]
        values = [payload.get(col) for col in columns]
        with self.connect() as conn:
            conn.execute(
                f"INSERT INTO case_runs ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                values,
            )

    def insert_gateway_turn(self, payload: dict[str, Any]) -> int:
        columns = [
            "task_id",
            "started_at",
            "alias",
            "model_key",
            "upstream_model",
            "stream",
            "status",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "estimated_cost",
            "error",
            "message_count",
            "has_tools",
            "prompt_chars",
        ]
        values = [payload.get(col) for col in columns]
        with self.connect() as conn:
            cursor = conn.execute(
                f"INSERT INTO gateway_turns ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                values,
            )
            return int(cursor.lastrowid)

    def insert_model_result(self, payload: dict[str, Any]) -> None:
        columns = [
            "run_id",
            "case_id",
            "model_key",
            "provider",
            "model",
            "tier",
            "started_at",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "estimated_cost",
            "answer_path",
            "raw_path",
            "error",
            "verdict",
            "evaluator",
            "evaluation_detail",
        ]
        values = [payload.get(col) for col in columns]
        if isinstance(values[-1], (dict, list)):
            values[-1] = json.dumps(values[-1])
        with self.connect() as conn:
            conn.execute(
                f"INSERT INTO model_results ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                values,
            )

    def update_verdict(
        self,
        run_id: str,
        case_id: str,
        model_key: str,
        verdict: str,
        reason: str,
    ) -> None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT evaluation_detail FROM model_results
                WHERE run_id = ? AND case_id = ? AND model_key = ?
                ORDER BY id DESC LIMIT 1
                """,
                (run_id, case_id, model_key),
            ).fetchone()
            if not row:
                raise KeyError(f"No result for {run_id}/{case_id}/{model_key}")
            detail = json.loads(row["evaluation_detail"] or "{}")
            detail["human_reason"] = reason
            conn.execute(
                """
                UPDATE model_results
                SET verdict = ?, evaluator = 'human', evaluation_detail = ?
                WHERE run_id = ? AND case_id = ? AND model_key = ?
                """,
                (verdict, json.dumps(detail), run_id, case_id, model_key),
            )

    def list_runs(self, limit: int = 20) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    def case_runs(self, run_id: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM case_runs WHERE run_id = ? ORDER BY case_id",
                (run_id,),
            ).fetchall()

    def model_results(self, run_id: str, case_id: str | None = None) -> list[sqlite3.Row]:
        with self.connect() as conn:
            if case_id:
                return conn.execute(
                    """
                    SELECT * FROM model_results
                    WHERE run_id = ? AND case_id = ?
                    ORDER BY tier, model_key
                    """,
                    (run_id, case_id),
                ).fetchall()
            return conn.execute(
                """
                SELECT * FROM model_results
                WHERE run_id = ?
                ORDER BY case_id, tier, model_key
                """,
                (run_id,),
            ).fetchall()

    def results_for_case(self, case_id: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT * FROM model_results
                WHERE case_id = ?
                ORDER BY started_at DESC, tier
                """,
                (case_id,),
            ).fetchall()

    def latest_run(self, mode: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT * FROM runs
                WHERE mode = ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (mode,),
            ).fetchone()

    def recompute_case_run(
        self,
        run_id: str,
        case_id: str,
        short_names: dict[str, str] | None = None,
    ) -> None:
        rows = self.model_results(run_id, case_id)
        if not rows:
            return
        solved = [r for r in rows if r["verdict"] == "PASS"]
        failed_before = 0
        waste_ms = 0.0
        waste_cost = 0.0
        total_ms = sum((r["latency_ms"] or 0) for r in rows)
        costs = [r["estimated_cost"] for r in rows if r["estimated_cost"] is not None]
        total_cost = sum(costs) if costs else None
        winner = None
        if solved:
            winner = min(solved, key=lambda r: (r["tier"], r["model_key"]))
            for row in rows:
                if row["tier"] < winner["tier"]:
                    failed_before += 1
                    waste_ms += row["latency_ms"] or 0
                    if row["estimated_cost"] is not None:
                        waste_cost += row["estimated_cost"]
        names = short_names or {}
        min_label = "NONE"
        if winner:
            min_label = names.get(winner["model_key"], winner["model_key"])
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE case_runs
                SET minimum_model_that_solved = ?,
                    successful_tier = ?,
                    total_escalation_latency_ms = ?,
                    total_escalation_cost = ?,
                    wasted_latency_before_success_ms = ?,
                    wasted_cost_before_success = ?,
                    failed_tiers = ?
                WHERE run_id = ? AND case_id = ?
                """,
                (
                    min_label,
                    winner["tier"] if winner else None,
                    total_ms,
                    total_cost,
                    waste_ms if winner else total_ms,
                    waste_cost if winner or waste_cost else None,
                    failed_before if winner else len(rows),
                    run_id,
                    case_id,
                ),
            )
