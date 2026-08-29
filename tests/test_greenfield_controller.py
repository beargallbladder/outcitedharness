from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.greenfield.controller import GreenfieldController
from harness.orch_loop import LoopState, load_loop_state, save_loop_state
from harness.repo_contract import build_repo_contract


class FakeAdapter:
    def bootstrap(self, root: Path, manifest):
        (root / "src").mkdir()
        (root / "src" / "app.py").write_text("def health():\n    return 'ok'\n")
        (root / ".harness.toml").write_text(
            """[verification]
required = ["unit"]

[verification.commands.unit]
argv = ["python", "-m", "pytest", "-q"]
timeout = 60
"""
        )
        (root / "pyproject.toml").write_text(
            """[project]
name = "sample"
version = "0.1.0"
dependencies = []

[tool.pytest.ini_options]
testpaths = ["tests"]
"""
        )
        return self.contract(root)

    @staticmethod
    def contract(root: Path):
        return build_repo_contract(root)

    @staticmethod
    def verify(_contract):
        return []


def _cfg(tmp_path: Path):
    return SimpleNamespace(
        settings=SimpleNamespace(
            db_path=tmp_path / "harness.sqlite",
            greenfield_runs_root=tmp_path / "runs",
            gci_enabled=False,
            gci_token_env="TEST_GCI_TOKEN",
            gci_url="http://spark.test:8810",
            gci_timeout_s=1,
        )
    )


def _controller(tmp_path: Path, monkeypatch) -> GreenfieldController:
    monkeypatch.setattr("harness.greenfield.controller.adapter_for", lambda _stack: FakeAdapter())
    return GreenfieldController(_cfg(tmp_path))


def test_approval_provisions_m0_once_and_resume_keeps_active_task(tmp_path: Path, monkeypatch):
    controller = _controller(tmp_path, monkeypatch)
    run = controller.plan(
        intent="Build a Python service that returns a mechanically tested health response",
        name="sample",
        stack="python",
        destination=tmp_path / "published",
        discovery_search=lambda *_args, **_kwargs: [],
    )
    assert run.status == "awaiting_approval"
    assert not (tmp_path / "runs").exists()
    active = controller.approve_and_provision(run.run_id)
    assert active.status == "running"
    assert active.milestones[0].state == "complete"
    assert active.milestones[0].commit_sha
    assert active.milestones[1].state == "active"
    first_task = active.milestones[1].task_id
    execution_prompt = controller.tasks.get(first_task).intent
    assert "gci://" not in execution_prompt
    assert "GREENFIELD DISCOVERY" not in execution_prompt
    assert "informed planning only" in execution_prompt

    restarted = _controller(tmp_path, monkeypatch)
    resumed = restarted.resume(run.run_id)
    assert resumed.milestones[1].task_id == first_task
    assert len([row for row in resumed.milestones if row.state == "active"]) == 1


def test_verified_milestones_commit_then_final_gate_publishes(
    tmp_path: Path,
    monkeypatch,
):
    notified: list[Path] = []
    monkeypatch.setattr(
        "harness.gci.automation.notify_publication",
        lambda _settings, destination: notified.append(destination),
    )
    controller = _controller(tmp_path, monkeypatch)
    run = controller.plan(
        intent="Build a Python service that returns a mechanically tested health response",
        name="sample",
        stack="python",
        destination=tmp_path / "published",
        discovery_search=lambda *_args, **_kwargs: [],
    )
    run = controller.approve_and_provision(run.run_id)
    root = Path(run.workspace_root)

    (root / "src" / "feature.py").write_text("VALUE = 1\n")
    first = run.milestones[1]
    save_loop_state(
        controller.tasks,
        first.task_id,
        LoopState(phase="verified", intent=controller.tasks.get(first.task_id).intent),
    )
    run = controller.reconcile_milestone(run.run_id)
    assert run.milestones[1].state == "complete"
    assert run.milestones[2].state == "active"

    (root / "README.md").write_text("# Complete\n")
    second = run.milestones[2]
    save_loop_state(
        controller.tasks,
        second.task_id,
        LoopState(phase="verified", intent=controller.tasks.get(second.task_id).intent),
    )
    complete = controller.reconcile_milestone(run.run_id)
    assert complete.status == "complete"
    assert complete.published_path == str(tmp_path / "published")
    assert (tmp_path / "published" / "src" / "feature.py").is_file()
    assert complete.final_state_hash
    assert notified == [tmp_path / "published"]


def test_destination_drift_blocks_publication_and_preserves_workspace(
    tmp_path: Path,
    monkeypatch,
):
    controller = _controller(tmp_path, monkeypatch)
    planned = controller.plan(
        intent="Build a Python service that returns a mechanically tested health response",
        name="sample",
        stack="python",
        destination=tmp_path / "published",
        discovery_search=lambda *_args, **_kwargs: [],
    )
    run = controller.approve_and_provision(planned.run_id)
    root = Path(run.workspace_root)
    for name in ("feature.py", "integration.py"):
        (root / "src" / name).write_text(f"VALUE = {name!r}\n")
        active = next(
            row for row in run.milestones if row.ordinal > 0 and row.state != "complete"
        )
        save_loop_state(
            controller.tasks,
            active.task_id,
            LoopState(
                phase="verified",
                intent=controller.tasks.get(active.task_id).intent,
            ),
        )
        if name == "integration.py":
            (tmp_path / "published").mkdir()
            (tmp_path / "published" / "user.txt").write_text("do not overwrite\n")
        run = controller.reconcile_milestone(run.run_id)
    assert run.status == "blocked"
    assert "publication blocked" in run.error
    assert (tmp_path / "published" / "user.txt").read_text() == "do not overwrite\n"
    assert (root / "src" / "integration.py").is_file()


def test_process_restart_after_approval_provisions_without_reapproval(
    tmp_path: Path,
    monkeypatch,
):
    controller = _controller(tmp_path, monkeypatch)
    planned = controller.plan(
        intent="Build a Python service that returns a mechanically tested health response",
        name="sample",
        stack="python",
        destination=tmp_path / "published",
        discovery_search=lambda *_args, **_kwargs: [],
    )
    approved = controller.service.approve(planned.run_id)
    assert approved.status == "provisioning"
    restarted = _controller(tmp_path, monkeypatch)
    resumed = restarted.resume(planned.run_id)
    assert resumed.status == "running"
    assert resumed.milestones[0].state == "complete"
    assert resumed.milestones[0].attempts == 1


def test_read_only_gather_can_be_safely_reset_after_boundary_bug(
    tmp_path: Path,
    monkeypatch,
):
    controller = _controller(tmp_path, monkeypatch)
    planned = controller.plan(
        intent="Build a Python service that returns a mechanically tested health response",
        name="sample",
        stack="python",
        destination=tmp_path / "published",
        discovery_search=lambda *_args, **_kwargs: [],
    )
    run = controller.approve_and_provision(planned.run_id)
    active = run.milestones[1]
    save_loop_state(
        controller.tasks,
        active.task_id,
        LoopState(
            phase="gather",
            intent=controller.tasks.get(active.task_id).intent,
        ),
    )
    controller.service.block(run.run_id, "transient controller safety block")
    assert controller.service.get(run.run_id).status == "blocked"
    reset = controller.reset_gather_only_task(run.run_id, "advisory path leak")
    assert reset.status == "running"
    assert reset.milestones[1].state == "active"
    assert load_loop_state(controller.tasks, active.task_id) is None
    assert "gci://" not in controller.tasks.get(active.task_id).intent


@pytest.mark.parametrize("terminal", ["blocked", "exhausted"])
def test_checkpointed_failed_milestone_can_retry_without_plan_change(
    tmp_path: Path,
    monkeypatch,
    terminal: str,
):
    controller = _controller(tmp_path, monkeypatch)
    planned = controller.plan(
        intent="Build a Python service that returns a mechanically tested health response",
        name="sample",
        stack="python",
        destination=tmp_path / "published",
        discovery_search=lambda *_args, **_kwargs: [],
    )
    run = controller.approve_and_provision(planned.run_id)
    active = run.milestones[1]
    state = LoopState(
        phase=terminal,
        intent=controller.tasks.get(active.task_id).intent,
        iteration=5,
        checkpoint_available=True,
        blocked_reason="repair model failed",
    )
    save_loop_state(controller.tasks, active.task_id, state)
    controller.service.update_milestone(
        run.run_id,
        active.ordinal,
        state=terminal,
        error="repair model failed",
    )
    controller.service.set_status(run.run_id, terminal, error="repair model failed")

    retried = controller.retry_blocked_milestone(run.run_id, "retry same plan")
    assert retried.status == "running"
    assert retried.milestones[1].state == "active"
    loop = load_loop_state(controller.tasks, active.task_id)
    assert loop.phase == "repair"
    assert loop.iteration == 4
    assert loop.blocked_reason == ""


@pytest.mark.parametrize(
    ("blocked_reason", "pending"),
    [
        ("verification evidence does not match current diff", []),
        (
            "checkpoint finalization failed: snapshot mismatch",
            ["src/app.py"],
        ),
    ],
)
def test_verification_bookkeeping_block_retries_without_code_repair(
    tmp_path: Path,
    monkeypatch,
    blocked_reason: str,
    pending: list[str],
):
    controller = _controller(tmp_path, monkeypatch)
    planned = controller.plan(
        intent="Build a Python service with mechanical verification",
        name="sample",
        stack="python",
        destination=tmp_path / "published",
        discovery_search=lambda *_args, **_kwargs: [],
    )
    run = controller.approve_and_provision(planned.run_id)
    active = run.milestones[1]
    state = LoopState(
        phase="blocked",
        intent=controller.tasks.get(active.task_id).intent,
        iteration=5,
        checkpoint_available=True,
        blocked_reason=blocked_reason,
        checkpoint_pending_paths=pending,
    )
    save_loop_state(controller.tasks, active.task_id, state)
    controller.service.update_milestone(
        run.run_id,
        active.ordinal,
        state="blocked",
        error=state.blocked_reason,
    )
    controller.service.set_status(
        run.run_id,
        "blocked",
        error=state.blocked_reason,
    )

    controller.retry_blocked_milestone(run.run_id, "rerun verification")
    loop = load_loop_state(controller.tasks, active.task_id)
    assert loop.phase == "verify"
    assert loop.verify_index == 0
    assert loop.verification_results == []
    assert loop.iteration == 5
    assert loop.working_set.refresh_pending == pending
