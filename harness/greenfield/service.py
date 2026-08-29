from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any

from harness.greenfield.manifest import (
    ManifestDriftError,
    assert_manifest_unchanged,
    build_manifest,
    canonical_hash,
    destination_fingerprint,
)
from harness.greenfield.models import (
    GreenfieldDiscovery,
    GreenfieldManifest,
    GreenfieldRun,
    MilestonePlan,
    MilestoneSpec,
    MilestoneState,
    ProductSpec,
)
from harness.storage.db import Store, utcnow


TERMINAL_STATES = {"complete", "blocked", "exhausted", "cancelled"}
RUN_STATES = {
    "planning",
    "awaiting_approval",
    "provisioning",
    "running",
    "final_verification",
    *TERMINAL_STATES,
}


def _new_run_id() -> str:
    stamp = "".join(char for char in utcnow() if char.isalnum()).lower()
    return f"gf{stamp}_{secrets.token_hex(3)}"


class GreenfieldService:
    def __init__(self, store: Store):
        self.store = store

    def create(
        self,
        *,
        intent: str,
        name: str,
        stack: str,
        destination: Path,
        discovery: GreenfieldDiscovery,
        spec: ProductSpec,
        plan: MilestonePlan,
    ) -> GreenfieldRun:
        destination = destination.expanduser().absolute()
        if destination.exists() or destination.is_symlink():
            raise ValueError("greenfield destination must not already exist")
        if not destination.parent.exists():
            raise ValueError("greenfield destination parent must exist")
        run_id = _new_run_id()
        created = utcnow()
        spec_hash = canonical_hash(spec)
        plan_hash = canonical_hash(plan)
        fingerprint = destination_fingerprint(destination)
        with self.store.connect() as conn:
            conn.execute(
                """
                INSERT INTO greenfield_runs(
                    run_id, intent, project_name, stack, destination,
                    destination_fingerprint, status, discovery_json, spec_json,
                    plan_json, spec_hash, plan_hash, current_milestone,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'awaiting_approval', ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    run_id,
                    intent,
                    name,
                    stack,
                    str(destination),
                    fingerprint,
                    json.dumps(discovery.to_dict(), sort_keys=True),
                    json.dumps(spec.to_dict(), sort_keys=True),
                    json.dumps(plan.to_dict(), sort_keys=True),
                    spec_hash,
                    plan_hash,
                    created,
                    created,
                ),
            )
            for ordinal, milestone in enumerate(plan.milestones):
                conn.execute(
                    """
                    INSERT INTO greenfield_milestones(
                        run_id, ordinal, milestone_id, title, objective,
                        acceptance_json, state, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        run_id,
                        ordinal,
                        milestone.milestone_id,
                        milestone.title,
                        milestone.objective,
                        json.dumps(milestone.to_dict(), sort_keys=True),
                        created,
                    ),
                )
            self._event_conn(
                conn,
                run_id,
                "planned",
                {
                    "spec_hash": spec_hash,
                    "plan_hash": plan_hash,
                    "discovery_hash": canonical_hash(discovery),
                },
            )
        return self.get(run_id)

    def get(self, run_id: str) -> GreenfieldRun:
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT * FROM greenfield_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            milestone_rows = conn.execute(
                """
                SELECT * FROM greenfield_milestones
                WHERE run_id=? ORDER BY ordinal
                """,
                (run_id,),
            ).fetchall()
        if row is None:
            raise KeyError(run_id)
        discovery = GreenfieldDiscovery.from_dict(json.loads(row["discovery_json"]))
        spec = ProductSpec.from_dict(json.loads(row["spec_json"]))
        plan = MilestonePlan.from_dict(json.loads(row["plan_json"]))
        manifest = (
            GreenfieldManifest.from_dict(json.loads(row["manifest_json"]))
            if row["manifest_json"]
            else None
        )
        milestones = [
            MilestoneState(
                ordinal=int(item["ordinal"]),
                milestone=MilestoneSpec.from_dict(json.loads(item["acceptance_json"])),
                state=item["state"],
                task_id=item["task_id"],
                starting_commit=item["starting_commit"],
                verified_state_hash=item["verified_state_hash"],
                commit_sha=item["commit_sha"],
                attempts=int(item["attempts"] or 0),
                error=item["error"],
            )
            for item in milestone_rows
        ]
        return GreenfieldRun(
            run_id=row["run_id"],
            intent=row["intent"],
            project_name=row["project_name"],
            stack=row["stack"],
            destination=row["destination"],
            destination_fingerprint=row["destination_fingerprint"],
            status=row["status"],
            discovery=discovery,
            spec=spec,
            plan=plan,
            spec_hash=row["spec_hash"],
            plan_hash=row["plan_hash"],
            workspace_root=row["workspace_root"],
            manifest=manifest,
            manifest_hash=row["manifest_hash"],
            approved_at=row["approved_at"],
            current_milestone=int(row["current_milestone"] or 0),
            final_state_hash=row["final_state_hash"],
            published_path=row["published_path"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            milestones=milestones,
        )

    def list(self, limit: int = 20) -> list[GreenfieldRun]:
        with self.store.connect() as conn:
            rows = conn.execute(
                "SELECT run_id FROM greenfield_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self.get(row["run_id"]) for row in rows]

    def approve(self, run_id: str) -> GreenfieldRun:
        run = self.get(run_id)
        if run.status != "awaiting_approval":
            raise ValueError(f"run is not awaiting approval: {run.status}")
        if canonical_hash(run.spec) != run.spec_hash or canonical_hash(run.plan) != run.plan_hash:
            self.block(run_id, "replan required: persisted plan hash drift")
            raise ManifestDriftError("persisted greenfield plan drifted before approval")
        manifest = build_manifest(
            run_id=run.run_id,
            spec=run.spec,
            plan=run.plan,
            discovery=run.discovery,
            destination=run.destination,
            destination_fingerprint_value=run.destination_fingerprint,
        )
        approved = utcnow()
        manifest_hash = canonical_hash(manifest)
        with self.store.connect() as conn:
            changed = conn.execute(
                """
                UPDATE greenfield_runs
                SET manifest_json=?, manifest_hash=?, approved_at=?,
                    status='provisioning', updated_at=?
                WHERE run_id=? AND status='awaiting_approval'
                """,
                (
                    json.dumps(manifest.to_dict(), sort_keys=True),
                    manifest_hash,
                    approved,
                    approved,
                    run_id,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError("approval race; run state changed")
            self._event_conn(
                conn,
                run_id,
                "approved",
                {"manifest_hash": manifest_hash},
            )
        return self.get(run_id)

    def assert_approved(self, run_id: str) -> GreenfieldRun:
        run = self.get(run_id)
        if run.manifest is None or not run.approved_at:
            raise ValueError("greenfield run has not been approved")
        try:
            assert_manifest_unchanged(
                run.manifest,
                run.spec,
                run.plan,
                run.discovery,
            )
        except ManifestDriftError as exc:
            self.block(run_id, f"replan required: {exc}")
            raise
        if canonical_hash(run.manifest) != run.manifest_hash:
            self.block(run_id, "replan required: approved manifest hash drift")
            raise ManifestDriftError("approved manifest hash drifted")
        return run

    def set_workspace(self, run_id: str, workspace_root: Path) -> None:
        with self.store.connect() as conn:
            conn.execute(
                """
                UPDATE greenfield_runs
                SET workspace_root=?, updated_at=?
                WHERE run_id=? AND status IN ('provisioning', 'running')
                """,
                (str(workspace_root.resolve()), utcnow(), run_id),
            )
            self._event_conn(
                conn,
                run_id,
                "workspace_ready",
                {"workspace_root": str(workspace_root.resolve())},
            )

    def update_milestone(
        self,
        run_id: str,
        ordinal: int,
        *,
        state: str,
        task_id: str | None = None,
        starting_commit: str | None = None,
        verified_state_hash: str | None = None,
        commit_sha: str | None = None,
        error: str | None = None,
        increment_attempts: bool = False,
    ) -> None:
        with self.store.connect() as conn:
            changed = conn.execute(
                """
                UPDATE greenfield_milestones
                SET state=?,
                    task_id=COALESCE(?, task_id),
                    starting_commit=COALESCE(?, starting_commit),
                    verified_state_hash=COALESCE(?, verified_state_hash),
                    commit_sha=COALESCE(?, commit_sha),
                    error=?,
                    attempts=attempts + ?,
                    updated_at=?
                WHERE run_id=? AND ordinal=?
                """,
                (
                    state,
                    task_id,
                    starting_commit,
                    verified_state_hash,
                    commit_sha,
                    error,
                    int(increment_attempts),
                    utcnow(),
                    run_id,
                    ordinal,
                ),
            ).rowcount
            if changed != 1:
                raise KeyError((run_id, ordinal))
            conn.execute(
                """
                UPDATE greenfield_runs
                SET current_milestone=?, updated_at=?
                WHERE run_id=?
                """,
                (ordinal, utcnow(), run_id),
            )
            self._event_conn(
                conn,
                run_id,
                "milestone_state",
                {
                    "ordinal": ordinal,
                    "state": state,
                    "task_id": task_id,
                    "commit_sha": commit_sha,
                    "error_present": bool(error),
                },
            )

    def claim_milestone_task(
        self,
        run_id: str,
        ordinal: int,
        *,
        task_id: str,
        starting_commit: str | None,
    ) -> bool:
        with self.store.connect() as conn:
            changed = conn.execute(
                """
                UPDATE greenfield_milestones
                SET state='active', task_id=?, starting_commit=?,
                    attempts=attempts + 1, updated_at=?
                WHERE run_id=? AND ordinal=? AND task_id IS NULL AND state='pending'
                """,
                (task_id, starting_commit, utcnow(), run_id, ordinal),
            ).rowcount
            if changed:
                conn.execute(
                    """
                    UPDATE greenfield_runs
                    SET current_milestone=?, updated_at=?
                    WHERE run_id=? AND status='running'
                    """,
                    (ordinal, utcnow(), run_id),
                )
                self._event_conn(
                    conn,
                    run_id,
                    "milestone_task",
                    {"ordinal": ordinal, "task_id": task_id},
                )
        return bool(changed)

    def set_status(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
        final_state_hash: str | None = None,
        published_path: str | None = None,
    ) -> None:
        if status not in RUN_STATES:
            raise ValueError(f"invalid greenfield status: {status}")
        with self.store.connect() as conn:
            conn.execute(
                """
                UPDATE greenfield_runs
                SET status=?, error=?, final_state_hash=COALESCE(?, final_state_hash),
                    published_path=COALESCE(?, published_path), updated_at=?
                WHERE run_id=?
                """,
                (
                    status,
                    error,
                    final_state_hash,
                    published_path,
                    utcnow(),
                    run_id,
                ),
            )
            self._event_conn(
                conn,
                run_id,
                "status",
                {"status": status, "error_present": bool(error)},
            )

    def block(self, run_id: str, reason: str) -> None:
        self.set_status(run_id, "blocked", error=reason)

    def cancel(self, run_id: str) -> GreenfieldRun:
        run = self.get(run_id)
        if run.status == "complete":
            raise ValueError("completed run cannot be cancelled")
        self.set_status(run_id, "cancelled", error="cancelled by operator")
        return self.get(run_id)

    def events(self, run_id: str) -> list[dict[str, Any]]:
        with self.store.connect() as conn:
            rows = conn.execute(
                "SELECT kind, payload_json, created_at FROM greenfield_events WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [
            {
                "kind": row["kind"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    @staticmethod
    def _event_conn(conn, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO greenfield_events(run_id, kind, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, kind, json.dumps(payload, sort_keys=True), utcnow()),
        )
