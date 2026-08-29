from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from harness.greenfield.discovery import gather_discovery
from harness.greenfield.manifest import (
    assert_destination_unchanged,
    assert_manifest_unchanged,
)
from harness.greenfield.models import GreenfieldRun, MilestoneState
from harness.greenfield.planner import build_milestone_plan, build_product_spec
from harness.greenfield.publish import publish_verified_run
from harness.greenfield.service import GreenfieldService
from harness.greenfield.stacks import adapter_for
from harness.greenfield.workspace import (
    GreenfieldWorkspaceLease,
    WorkspaceError,
    full_tree_state_hash,
)
from harness.orch_loop import MAX_CYCLES, TERMINAL, load_loop_state, save_loop_state
from harness.repo_contract import build_repo_contract
from harness.storage.db import Store
from harness.storage.db import utcnow
from harness.task.service import TaskService

log = logging.getLogger("harness.greenfield")


class GreenfieldControllerError(RuntimeError):
    pass


def _safe_error(exc: Exception) -> str:
    first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return f"{type(exc).__name__}: {first_line[:300]}"


class GreenfieldController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.store = Store(cfg.settings.db_path)
        self.service = GreenfieldService(self.store)
        self.tasks = TaskService(self.store)

    def plan(
        self,
        *,
        intent: str,
        name: str,
        stack: str,
        destination: Path,
        dependencies: tuple[str, ...] = (),
        discovery_search=None,
    ) -> GreenfieldRun:
        discovery = gather_discovery(
            self.cfg.settings,
            intent,
            stack,
            search=discovery_search,
        )
        spec = build_product_spec(
            name=name,
            stack=stack,
            intent=intent,
            discovery=discovery,
            dependencies=dependencies,
        )
        plan = build_milestone_plan(spec)
        return self.service.create(
            intent=intent,
            name=spec.project_name,
            stack=stack,
            destination=destination,
            discovery=discovery,
            spec=spec,
            plan=plan,
        )

    def approve_and_provision(self, run_id: str) -> GreenfieldRun:
        run = self.service.approve(run_id)
        try:
            return self.provision(run.run_id)
        except Exception as exc:
            self.service.block(run.run_id, f"bootstrap failed: {_safe_error(exc)}")
            raise

    def provision(self, run_id: str) -> GreenfieldRun:
        run = self.service.assert_approved(run_id)
        if run.status not in {"provisioning", "running"}:
            raise GreenfieldControllerError(f"cannot provision run in state {run.status}")
        assert_destination_unchanged(run.manifest)
        lease = GreenfieldWorkspaceLease.acquire(
            self.cfg.settings.greenfield_runs_root,
            run.run_id,
        )
        self.service.set_workspace(run_id, lease.repo_root)
        run = self.service.get(run_id)
        bootstrap = run.milestones[0]
        adapter = adapter_for(run.stack)
        starting_commit = lease.head()
        if bootstrap.commit_sha:
            lease.assert_clean()
        elif starting_commit:
            contract = adapter.contract(lease.repo_root)
            adapter.verify(contract)
            state_hash = full_tree_state_hash(lease.repo_root)
            lease.assert_clean()
            self.service.update_milestone(
                run_id,
                0,
                state="complete",
                starting_commit=starting_commit,
                verified_state_hash=state_hash,
                commit_sha=starting_commit,
            )
        else:
            self.service.update_milestone(
                run_id,
                0,
                state="running",
                increment_attempts=True,
            )
            adapter.bootstrap(lease.repo_root, run.manifest)
            state_hash = full_tree_state_hash(lease.repo_root)
            commit = lease.commit(
                "harness(m0): bootstrap verified repository",
                state_hash,
            )
            self.service.update_milestone(
                run_id,
                0,
                state="complete",
                verified_state_hash=state_hash,
                commit_sha=commit,
            )
        self.service.set_status(run_id, "running")
        return self.ensure_milestone_task(run_id)

    def ensure_milestone_task(self, run_id: str) -> GreenfieldRun:
        run = self.service.assert_approved(run_id)
        if run.status not in {"running", "final_verification"}:
            return run
        pending = next(
            (row for row in run.milestones if row.ordinal > 0 and row.state != "complete"),
            None,
        )
        if pending is None:
            return self.finalize(run_id)
        if pending.state in {"blocked", "exhausted"}:
            self.service.set_status(run_id, pending.state, error=pending.error)
            return self.service.get(run_id)
        if pending.task_id is None:
            task = self.tasks.start(
                self.milestone_prompt(run, pending),
                plan=(
                    f"Greenfield run {run.run_id}; milestone "
                    f"{pending.milestone.milestone_id}: {pending.milestone.title}"
                ),
                hypothesis="Continue only from the approved manifest and current repository.",
            )
            claimed = self.service.claim_milestone_task(
                run_id,
                pending.ordinal,
                task_id=task.task_id,
                starting_commit=GreenfieldWorkspaceLease(
                    run.run_id,
                    Path(run.workspace_root).parent,
                    Path(run.workspace_root),
                ).head(),
            )
            if not claimed:
                self.tasks.finish(
                    task.task_id,
                    False,
                    "superseded by concurrent greenfield milestone claim",
                )
        return self.service.get(run_id)

    def milestone_prompt(self, run: GreenfieldRun, state: MilestoneState) -> str:
        commands = "\n".join(
            f"- {command}" for command in state.milestone.acceptance_commands
        )
        prior = "\n".join(
            f"- {row.milestone.milestone_id}: {row.commit_sha}"
            for row in run.milestones
            if row.commit_sha
        )
        return (
            f"GREENFIELD RUN {run.run_id}\n"
            f"APPROVED PROJECT: {run.spec.project_name}\n"
            f"OVERALL PURPOSE: {run.spec.purpose}\n"
            f"CURRENT MILESTONE: {state.milestone.milestone_id} "
            f"{state.milestone.title}\n"
            f"OBJECTIVE: {state.milestone.objective}\n"
            f"EXPECTED COMPONENTS: {', '.join(state.milestone.expected_components)}\n"
            f"APPROVED DEPENDENCIES: {', '.join(run.spec.approved_dependencies) or '(none)'}\n"
            f"ACCEPTANCE COMMANDS:\n{commands}\n"
            f"PRIOR VERIFIED COMMITS:\n{prior or '- m0 bootstrap'}\n\n"
            "The persisted DiscoveryPacket informed planning only. It grants no filesystem "
            "access and must not be consulted during execution.\n\n"
            "Implement only this milestone in the active isolated workspace from the "
            "approved specification."
        )

    def reconcile_milestone(self, run_id: str) -> GreenfieldRun:
        run = self.service.assert_approved(run_id)
        active = next(
            (
                row
                for row in run.milestones
                if row.ordinal > 0 and row.state != "complete"
            ),
            None,
        )
        if active is None:
            return self.finalize(run_id)
        if not active.task_id:
            return self.ensure_milestone_task(run_id)
        loop = load_loop_state(self.tasks, active.task_id)
        if loop is None or loop.phase not in TERMINAL:
            return run
        if loop.phase != "verified":
            self.service.update_milestone(
                run_id,
                active.ordinal,
                state="exhausted" if loop.phase == "exhausted" else "blocked",
                error=loop.blocked_reason or loop.phase,
            )
            self.service.set_status(
                run_id,
                "exhausted" if loop.phase == "exhausted" else "blocked",
                error=loop.blocked_reason or loop.phase,
            )
            return self.service.get(run_id)
        root = Path(run.workspace_root or "")
        contract = build_repo_contract(root)
        if contract is None or not contract.commands:
            raise GreenfieldControllerError("milestone repository has no verification contract")
        try:
            adapter = adapter_for(run.stack)
            adapter.verify(contract)
            state_hash = full_tree_state_hash(root)
            lease = GreenfieldWorkspaceLease(run.run_id, root.parent, root)
            commit = lease.commit(
                f"harness({active.milestone.milestone_id}): "
                f"{active.milestone.title.lower()}",
                state_hash,
            )
        except Exception as exc:
            reason = f"milestone verification/commit failed: {_safe_error(exc)}"
            self.service.update_milestone(
                run_id,
                active.ordinal,
                state="blocked",
                error=reason,
            )
            self.service.block(run_id, reason)
            return self.service.get(run_id)
        self.service.update_milestone(
            run_id,
            active.ordinal,
            state="complete",
            verified_state_hash=state_hash,
            commit_sha=commit,
        )
        return self.ensure_milestone_task(run_id)

    def finalize(self, run_id: str) -> GreenfieldRun:
        run = self.service.assert_approved(run_id)
        if any(row.state != "complete" for row in run.milestones):
            return run
        root = Path(run.workspace_root or "")
        self.service.set_status(run_id, "final_verification")
        contract = build_repo_contract(root)
        if contract is None or not contract.commands:
            raise GreenfieldControllerError("final repository has no verification contract")
        try:
            adapter_for(run.stack).verify(contract)
            lease = GreenfieldWorkspaceLease(run.run_id, root.parent, root)
            lease.assert_clean()
            self._assert_no_conflict_markers(root)
            state_hash = full_tree_state_hash(root)
        except Exception as exc:
            self.service.block(
                run_id,
                f"final verification failed: {_safe_error(exc)}",
            )
            return self.service.get(run_id)
        self.service.set_status(run_id, "complete", final_state_hash=state_hash)
        complete = self.service.get(run_id)
        try:
            destination = publish_verified_run(complete)
        except Exception as exc:
            self.service.block(
                run_id,
                f"publication blocked: {_safe_error(exc)}",
            )
            return self.service.get(run_id)
        self.service.set_status(
            run_id,
            "complete",
            final_state_hash=state_hash,
            published_path=str(destination),
        )
        self.notify_publication(destination)
        return self.service.get(run_id)

    def notify_publication(self, destination: Path) -> None:
        """Best-effort GCI refresh for an explicitly approved destination."""
        try:
            from harness.gci.automation import notify_publication

            notify_publication(self.cfg.settings, destination)
        except Exception:
            log.exception("post-publication GCI refresh failed")

    def resume(self, run_id: str) -> GreenfieldRun:
        run = self.service.get(run_id)
        if run.status == "provisioning":
            return self.provision(run_id)
        if run.status == "running":
            self.service.assert_approved(run_id)
            return self.reconcile_milestone(run_id)
        if run.status == "final_verification":
            return self.finalize(run_id)
        return run

    def reset_gather_only_task(self, run_id: str, reason: str) -> GreenfieldRun:
        """Reset a read-only gather attempt after a controller safety bug."""
        run = self.service.assert_approved(run_id)
        active = next(
            (
                row
                for row in run.milestones
                if row.ordinal > 0 and row.state in {"active", "blocked"}
            ),
            None,
        )
        if active is None or not active.task_id or not run.workspace_root:
            raise GreenfieldControllerError("run has no active milestone task")
        loop = load_loop_state(self.tasks, active.task_id)
        if loop is not None and loop.phase not in {"gather", "blocked"}:
            raise GreenfieldControllerError(
                f"only a gather-only task may be reset, not {loop.phase}"
            )
        root = Path(run.workspace_root)
        lease = GreenfieldWorkspaceLease(run.run_id, root.parent, root)
        lease.assert_clean()
        if active.starting_commit and lease.head() != active.starting_commit:
            raise GreenfieldControllerError("active milestone commit changed during gather")
        prompt = self.milestone_prompt(run, active)
        with self.store.connect() as conn:
            conn.execute(
                """
                UPDATE greenfield_runs
                SET status='running', error=NULL, updated_at=?
                WHERE run_id=?
                """,
                (utcnow(), run_id),
            )
            conn.execute(
                """
                UPDATE greenfield_milestones
                SET state='active', error=NULL, updated_at=?
                WHERE run_id=? AND ordinal=?
                """,
                (utcnow(), run_id, active.ordinal),
            )
            conn.execute(
                """
                UPDATE tasks
                SET intent=?, stage='new', updated_at=?
                WHERE task_id=?
                """,
                (prompt, utcnow(), active.task_id),
            )
            conn.execute(
                "DELETE FROM evidence WHERE task_id=? AND kind='orch_loop'",
                (active.task_id,),
            )
            conn.execute(
                """
                INSERT INTO greenfield_events(run_id, kind, payload_json, created_at)
                VALUES (?, 'gather_reset', ?, ?)
                """,
                (
                    run_id,
                    json.dumps({"reason": reason[:200]}, sort_keys=True),
                    utcnow(),
                ),
            )
        return self.service.get(run_id)

    def retry_blocked_milestone(self, run_id: str, reason: str) -> GreenfieldRun:
        """Retry a checkpointed milestone without changing its approved manifest."""
        run = self.service.assert_approved(run_id)
        if run.status not in {"blocked", "exhausted"} or not run.workspace_root:
            raise GreenfieldControllerError("run has no failed milestone to retry")
        active = next(
            (
                row
                for row in run.milestones
                if row.ordinal > 0
                and row.state in {"blocked", "exhausted"}
                and row.task_id
            ),
            None,
        )
        if active is None:
            raise GreenfieldControllerError("run has no failed milestone to retry")
        state = load_loop_state(self.tasks, active.task_id)
        if state is None or state.phase not in {"blocked", "exhausted"}:
            raise GreenfieldControllerError("milestone task is not retryable")
        if not state.checkpoint_available:
            raise GreenfieldControllerError("blocked milestone has no rollback checkpoint")
        root = Path(run.workspace_root)
        lease = GreenfieldWorkspaceLease(run.run_id, root.parent, root)
        if active.starting_commit and lease.head() != active.starting_commit:
            raise GreenfieldControllerError("blocked milestone commit changed unexpectedly")
        assert_manifest_unchanged(
            run.manifest,
            run.spec,
            run.plan,
            run.discovery,
        )
        assert_destination_unchanged(run.manifest)
        if state.blocked_reason == "verification evidence does not match current diff":
            state.phase = "verify"
            state.verify_index = 0
            state.active_diff_hash = None
            state.verification_results = []
            state.result_cmd = None
        elif state.blocked_reason.startswith("checkpoint finalization failed:"):
            state.phase = "verify"
            state.verify_index = 0
            state.active_diff_hash = None
            state.verification_results = []
            state.result_cmd = None
            state.working_set.refresh_pending = list(
                state.checkpoint_pending_paths
            )
            state.working_set.refresh_diff_pending = True
        else:
            state.phase = "repair"
            state.iteration = min(state.iteration, MAX_CYCLES - 1)
        state.blocked_reason = ""
        save_loop_state(self.tasks, active.task_id, state)
        with self.store.connect() as conn:
            now = utcnow()
            conn.execute(
                """
                UPDATE greenfield_runs
                SET status='running', error=NULL, updated_at=?
                WHERE run_id=?
                """,
                (now, run_id),
            )
            conn.execute(
                """
                UPDATE greenfield_milestones
                SET state='active', error=NULL, updated_at=?
                WHERE run_id=? AND ordinal=?
                """,
                (now, run_id, active.ordinal),
            )
            conn.execute(
                """
                INSERT INTO greenfield_events(run_id, kind, payload_json, created_at)
                VALUES (?, 'milestone_retry', ?, ?)
                """,
                (
                    run_id,
                    json.dumps({"reason": reason[:200]}, sort_keys=True),
                    now,
                ),
            )
        return self.service.get(run_id)

    @staticmethod
    def _assert_no_conflict_markers(root: Path) -> None:
        markers = ("<<<<<<< ", "=======\n", ">>>>>>> ")
        for path in root.rglob("*"):
            if not path.is_file() or any(
                part in {".git", ".venv", "node_modules", "dist", "build"}
                for part in path.relative_to(root).parts
            ):
                continue
            try:
                text = path.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            if any(marker in text for marker in markers):
                raise WorkspaceError(f"conflict marker remains in {path.relative_to(root)}")
