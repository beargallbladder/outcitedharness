from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from harness.checkpoints import CheckpointStore
from harness.cli import app
from harness.orch_loop import LoopState, save_loop_state
from harness.storage.db import Store
from harness.task.service import TaskService


def _prepared_task(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.py"
    created = workspace / "created.py"
    target.write_text("VALUE = 'before'\n")
    db_path = tmp_path / "harness.db"
    results_dir = tmp_path / "results"
    svc = TaskService(Store(db_path))
    task = svc.start("rollback CLI test")
    run_id = "cli-run"
    checkpoints = CheckpointStore(results_dir / "checkpoints")
    checkpoints.capture_baseline(
        task_id=task.task_id,
        run_id=run_id,
        repo_root=workspace,
        intent=task.intent,
        paths=["target.py", "created.py"],
        before_iteration=1,
    )
    target.write_text("VALUE = 'agent'\n")
    created.write_text("CREATED = True\n")
    manifest = checkpoints.record_checkpoint(
        task_id=task.task_id,
        run_id=run_id,
        number=1,
    )
    state = LoopState(
        phase="exhausted",
        intent=task.intent,
        iteration=1,
        checkpoint_task_id=task.task_id,
        checkpoint_run_id=run_id,
        checkpoint_count=1,
        checkpoint_available=True,
        checkpoint_last_manifest=str(
            checkpoints.manifest_path(task.task_id, run_id, 1)
        ),
        active_diff_hash=manifest.active_diff_hash,
    )
    save_loop_state(svc, task.task_id, state)
    cfg = SimpleNamespace(
        settings=SimpleNamespace(
            db_path=db_path,
            results_dir=results_dir,
            checkpoint_max_file_bytes=1_000_000,
        )
    )
    return cfg, svc, task.task_id, target, created


def test_rollback_cli_requires_confirmation_and_reports_paths(tmp_path: Path, monkeypatch):
    cfg, svc, task_id, target, created = _prepared_task(tmp_path)
    monkeypatch.setattr("harness.cli._cfg", lambda: cfg)
    runner = CliRunner()

    cancelled = runner.invoke(app, ["rollback-task", task_id], input="n\n")
    assert cancelled.exit_code == 1
    assert "Restore: target.py" in cancelled.output
    assert "Remove: created.py" in cancelled.output
    assert "Rollback cancelled" in cancelled.output
    assert target.read_text() == "VALUE = 'agent'\n"
    assert created.exists()

    completed = runner.invoke(app, ["rollback-task", task_id], input="y\n")
    assert completed.exit_code == 0, completed.output
    assert "Rollback complete" in completed.output
    assert target.read_text() == "VALUE = 'before'\n"
    assert not created.exists()
    evidence = svc.evidence(task_id, kind="task_rollback")
    assert evidence[-1].payload["status"] == "success"
    assert evidence[-1].payload["restored"] == ["target.py"]
    assert evidence[-1].payload["removed"] == ["created.py"]


def test_rollback_cli_refuses_conflict_without_partial_restore(tmp_path: Path, monkeypatch):
    cfg, svc, task_id, target, created = _prepared_task(tmp_path)
    monkeypatch.setattr("harness.cli._cfg", lambda: cfg)
    target.write_text("VALUE = 'user after agent'\n")

    result = CliRunner().invoke(app, ["rollback-task", task_id, "--yes"])
    assert result.exit_code == 1
    assert "Rollback refused" in result.output
    assert "target.py" in result.output
    assert target.read_text() == "VALUE = 'user after agent'\n"
    assert created.read_text() == "CREATED = True\n"
    evidence = svc.evidence(task_id, kind="task_rollback")
    assert evidence[-1].payload["status"] == "refused"
    assert evidence[-1].payload["conflicts"] == ["target.py"]


def test_rollback_cli_does_not_fall_back_to_an_older_loop_run(
    tmp_path: Path,
    monkeypatch,
):
    cfg, svc, task_id, target, created = _prepared_task(tmp_path)
    monkeypatch.setattr("harness.cli._cfg", lambda: cfg)
    save_loop_state(
        svc,
        task_id,
        LoopState(phase="gather", intent="newer task without mutations"),
    )

    result = CliRunner().invoke(app, ["rollback-task", task_id, "--yes"])
    assert result.exit_code == 1
    assert "No rollback checkpoint is available" in result.output
    assert target.read_text() == "VALUE = 'agent'\n"
    assert created.exists()
