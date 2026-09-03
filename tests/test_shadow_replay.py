from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.shadow.hook import capture_hook_event
from harness.shadow.comparison import compare_task
from harness.shadow.models import ShadowAttempt, canonical_json
from harness.shadow.processor import process_task
from harness.shadow.replay import replay_task
from harness.shadow.spool import ShadowSpool
from harness.storage.db import Store
from harness.training.ledger import LearningLedger


PATCH = """diff --git a/app.py b/app.py
index 7c02ec3..0cfa1df 100644
--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-value = 1
+value = 2
"""


def _run(root: Path, *argv: str) -> None:
    result = subprocess.run(
        list(argv),
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _run(root, "git", "init", "-q")
    (root / "app.py").write_text("value = 1\n")
    (root / "harness").mkdir()
    (root / "harness/__init__.py").write_text("SNAPSHOT_MARKER = 'captured'\n")
    (root / "test_app.py").write_text(
        "import app\n"
        "from harness import SNAPSHOT_MARKER\n\n\n"
        "def test_value_is_two():\n"
        "    assert SNAPSHOT_MARKER == 'captured'\n"
        "    assert app.value == 2\n"
    )
    (root / ".venv/bin").mkdir(parents=True)
    pytest_entrypoint = root / ".venv/bin/pytest"
    pytest_entrypoint.write_text(
        f"#!{sys.executable}\n"
        "from pytest import console_main\n"
        "raise SystemExit(console_main())\n"
    )
    pytest_entrypoint.chmod(0o755)
    (root / ".harness.toml").write_text(
        """[verification]
required = ["unit"]

[verification.commands.unit]
argv = [".venv/bin/pytest", "-q"]
timeout = 30
"""
    )
    _run(
        root,
        "git",
        "add",
        "app.py",
        "harness/__init__.py",
        "test_app.py",
        ".harness.toml",
    )
    _run(
        root,
        "git",
        "-c",
        "user.name=Harness Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "base",
    )
    (root / ".harness-shadow.json").write_text(
        json.dumps(
            {
                "version": 1,
                "enabled": True,
                "repository_id": "owner/replay-example",
                "allowed_paths": ["."],
                "excluded_paths": [
                    ".git",
                    ".harness-shadow.json",
                    "**/.git/**",
                ],
            }
        )
    )
    return root


def _task(root: Path, spool: ShadowSpool, generation: str) -> str:
    task_id = capture_hook_event(
        "beforeSubmitPrompt",
        {
            "conversation_id": f"session-{generation}",
            "generation_id": generation,
            "model": "gpt-5.6-sol-max-fast",
            "prompt": "Fix the failing value test.",
        },
        repository_root=root,
        spool_root=spool.root,
    )
    assert task_id
    return task_id


def test_local_patch_requires_fail_before_and_pass_after(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    spool = ShadowSpool(tmp_path / "spool")
    task_id = _task(root, spool, "local")
    (root / ".harness.toml").write_text(
        """[verification]
required = ["changed-after-capture"]

[verification.commands.changed-after-capture]
argv = ["python", "-m", "pytest", "-q"]
"""
    )
    lease = spool.claim()
    assert lease
    spool.complete(
        lease,
        ShadowAttempt(
            attempt_id=f"attempt-{task_id}-1",
            task_id=task_id,
            status="completed",
            model="qwen-local",
            model_endpoint_sha256="a" * 64,
            answer="Changed the value to two.",
            patch=PATCH,
            created_at=datetime.now(timezone.utc),
        ),
    )

    report = replay_task(
        spool,
        task_id,
        candidate_kind="local",
        command_names=("unit",),
        work_root=tmp_path / "replays",
    )

    assert report.verdict == "verified_correction"
    assert report.baseline_failed
    assert report.candidate_passed
    assert [row.returncode for row in report.commands] == [1, 0]
    unsigned = report.model_dump(mode="json", exclude={"evidence_sha256"})
    assert report.evidence_sha256 == hashlib.sha256(
        canonical_json(unsigned)
    ).hexdigest()
    assert len(spool.replays(task_id)) == 1
    assert (root / "app.py").read_text() == "value = 1\n"
    with sqlite3.connect(spool.database) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE shadow_replays SET candidate_kind = 'frontier'"
            )


def test_named_preexisting_failure_may_remain_without_regression(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "app.py").write_text("value = 1\n")
    (root / "test_app.py").write_text(
        "from app import value\n\n"
        "def test_preexisting_failure():\n"
        "    assert False\n\n"
        "def test_repair_target():\n"
        "    assert value == 2\n"
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname='shadow-test'\nversion='0.0.0'\n"
        "dependencies=['pytest']\n"
    )
    (root / ".harness.toml").write_text(
        """[verification]
required = ["unit"]

[verification.commands.unit]
argv = ["python", "-m", "pytest", "-q"]
timeout = 30
"""
    )
    (root / ".harness-shadow.json").write_text(
        json.dumps(
            {
                "version": 1,
                "enabled": True,
                "repository_id": "owner/replay-example",
                "allowed_paths": ["."],
                "excluded_paths": [
                    ".git",
                    ".harness-shadow.json",
                    "**/.git/**",
                ],
            }
        )
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "base",
        ],
        cwd=root,
        check=True,
    )
    spool = ShadowSpool(tmp_path / "spool")
    task_id = _task(root, spool, "reduce named failures")
    lease = spool.claim()
    assert lease
    spool.complete(
        lease,
        ShadowAttempt(
            attempt_id=f"attempt-{task_id}-1",
            task_id=task_id,
            status="completed",
            model="qwen-local",
            model_endpoint_sha256="a" * 64,
            answer="Fixed only the requested failure.",
            patch=PATCH,
            created_at=datetime.now(timezone.utc),
        ),
    )

    report = replay_task(
        spool,
        task_id,
        candidate_kind="local",
        command_names=("unit",),
        work_root=tmp_path / "replays",
    )

    assert report.verdict == "verified_correction"
    assert report.baseline_failed
    assert not report.candidate_passed
    assert report.candidate_no_regression
    assert report.candidate_improved
    before, after = report.commands
    assert len(before.failure_fingerprints) == 2
    assert after.failure_fingerprints == (
        "test_app.py::test_preexisting_failure",
    )


def test_only_mechanically_proven_frontier_correction_is_admitted(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    spool = ShadowSpool(tmp_path / "spool")
    task_id = _task(root, spool, "comparison")
    lease = spool.claim()
    assert lease
    spool.complete(
        lease,
        ShadowAttempt(
            attempt_id=f"attempt-{task_id}-1",
            task_id=task_id,
            status="completed",
            model="qwen-local",
            model_endpoint_sha256="a" * 64,
            answer="No change needed.",
            created_at=datetime.now(timezone.utc),
        ),
    )
    assert (
        capture_hook_event(
            "afterAgentResponse",
            {
                "conversation_id": "session-comparison",
                "generation_id": "comparison",
                "model": "gpt-5.6-sol-max-fast",
                "text": "Changed the value and verified the test.",
            },
            repository_root=root,
            spool_root=spool.root,
        )
        == task_id
    )
    (root / "app.py").write_text("value = 2\n")
    assert (
        capture_hook_event(
            "stop",
            {
                "conversation_id": "session-comparison",
                "generation_id": "comparison",
                "model": "gpt-5.6-sol-max-fast",
                "status": "completed",
                "loop_count": 1,
            },
            repository_root=root,
            spool_root=spool.root,
        )
        == task_id
    )
    replay_task(
        spool,
        task_id,
        candidate_kind="local",
        command_names=("unit",),
        work_root=tmp_path / "local-replays",
    )
    replay_task(
        spool,
        task_id,
        candidate_kind="frontier",
        command_names=("unit",),
        work_root=tmp_path / "frontier-replays",
    )

    comparison = compare_task(spool, task_id)
    assert comparison.decision == "frontier_correction"
    assert comparison.chosen == "frontier"
    assert comparison.rejected == "local"
    assert comparison.eligible
    assert spool.processable_tasks() == (task_id,)

    store = Store(tmp_path / "learning.db")
    ledger = LearningLedger(store, tmp_path / "learning-artifacts")
    result = process_task(spool, ledger, task_id)
    assert result.admission.decision == "eligible"
    assert spool.processable_tasks() == ()
    with store.connect() as connection:
        event = connection.execute(
            "SELECT * FROM learning_events WHERE event_id = ?",
            (result.capture["event_id"],),
        ).fetchone()
        assert event["event_type"] == "coding_frontier_correction"
        kinds = {
            row["kind"]
            for row in connection.execute(
                "SELECT kind FROM learning_artifacts WHERE event_id = ?",
                (event["event_id"],),
            )
        }
    assert {
        "coding_prompt",
        "coding_chosen_patch",
        "coding_rejected_replay",
        "coding_comparison",
        "coding_chosen_replay",
        "coding_frontier_response",
    } <= kinds
    assert "coding_rejected_patch" not in kinds


def test_comparison_rejects_teacher_identity_drift(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    spool = ShadowSpool(tmp_path / "spool")
    task_id = _task(root, spool, "identity")
    lease = spool.claim()
    assert lease
    spool.complete(
        lease,
        ShadowAttempt(
            attempt_id=f"attempt-{task_id}-1",
            task_id=task_id,
            status="completed",
            model="qwen-local",
            model_endpoint_sha256="a" * 64,
            answer="No change needed.",
            created_at=datetime.now(timezone.utc),
        ),
    )
    task, _ = spool.get_task(task_id)
    for candidate in ("local", "frontier"):
        value = {
            "version": 1,
            "replay_id": f"replay-{candidate}",
            "task_id": task_id,
            "candidate_kind": candidate,
            "repository_id": task.policy.repository_id,
            "source_revision": task.snapshot.revision,
            "parent_state_sha256": task.snapshot.state_sha256,
            "candidate_patch_sha256": "0" * 64,
            "candidate_patch_object_path": "objects/sha256/00/" + ("0" * 64),
            "candidate_patch_bytes": 0,
            "contract_fingerprint": "0" * 64,
            "commands": [],
            "verdict": "inconclusive",
            "baseline_failed": False,
            "candidate_passed": False,
            "network_isolation": "sandbox-exec",
            "created_at": datetime.now(timezone.utc),
        }
        from harness.shadow.replay import ReplayReport

        provisional = ReplayReport(**value, evidence_sha256="0" * 64)
        report = provisional.model_copy(
            update={
                "evidence_sha256": hashlib.sha256(
                    canonical_json(
                        provisional.model_dump(
                            mode="json",
                            exclude={"evidence_sha256"},
                        )
                    )
                ).hexdigest()
            }
        )
        payload = report.model_dump(mode="json", exclude_none=True)
        spool.record_replay(
            replay_id=report.replay_id,
            task_id=task_id,
            candidate_kind=candidate,
            report=payload,
            report_sha256=hashlib.sha256(canonical_json(payload)).hexdigest(),
            created_at=datetime.now(timezone.utc),
        )
    capture_hook_event(
        "stop",
        {
            "conversation_id": "session-identity",
            "generation_id": "identity",
            "model": "different-frontier-model",
            "status": "completed",
            "loop_count": 1,
        },
        repository_root=root,
        spool_root=spool.root,
    )

    comparison = process_task(
        spool,
        LearningLedger(
            Store(tmp_path / "identity-learning.db"),
            tmp_path / "identity-artifacts",
        ),
        task_id,
    )
    assert comparison.decision == "rejected"
    assert not comparison.eligible
    assert not comparison.teacher_identity_verified
    assert comparison.observed_teacher_models == (
        "different-frontier-model",
        "gpt-5.6-sol-max-fast",
    )
    assert spool.processable_tasks() == ()


def test_frontier_final_state_replays_against_same_parent(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    spool = ShadowSpool(tmp_path / "spool")
    task_id = _task(root, spool, "frontier")
    (root / "app.py").write_text("value = 2\n")
    assert (
        capture_hook_event(
            "stop",
            {
                "conversation_id": "session-frontier",
                "generation_id": "frontier",
                "status": "completed",
                "loop_count": 1,
            },
            repository_root=root,
            spool_root=spool.root,
        )
        == task_id
    )

    report = replay_task(
        spool,
        task_id,
        candidate_kind="frontier",
        command_names=("unit",),
        work_root=tmp_path / "replays",
    )

    assert report.verdict == "verified_correction"
    assert report.baseline_failed
    assert report.candidate_passed
