from __future__ import annotations

import json
import hashlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.storage.db import Store
from harness.training.ledger import ArtifactPayload, LearningLedger, VerificationPayload
from harness.training.models import LearningEvent, SourceKind
from harness.training.queue import (
    DatasetMember,
    DatasetVersionRegistry,
    PrioritySignals,
    TrainingQueue,
)
from harness.training.split import Split
from harness.training.worker import HandlerSpec, TrainingWorker


def _signals() -> PrioritySignals:
    return PrioritySignals(
        observed_frequency=1,
        frontier_cost=1,
        local_failure_rate=1,
        verification_strength=1,
        diversity=1,
        expected_gpu_hours=1,
    )


def _script(path: Path, body: str) -> Path:
    path.write_text(f"#!/bin/sh\nset -eu\n{body}\n")
    path.chmod(0o700)
    return path


def _dataset(store: Store, tmp_path: Path) -> str:
    ledger = LearningLedger(store, tmp_path / "artifacts")
    event = LearningEvent(
        event_id="worker-event",
        event_type="verified_repair",
        source_kind=SourceKind.GIT,
        source_uri="git+file:///owned/repo?revision=" + "a" * 40,
        source_revision="a" * 40,
        lineage_id="owned-repo",
        authorization_scope="test",
        created_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        metadata={"data_use": "training", "disposition": "verified"},
    )
    capture = ledger.capture(
        event,
        [ArtifactPayload(kind="patch", content="diff --git a/a b/a")],
        [
            VerificationPayload(
                kind="pytest",
                status="pass",
                verifier="pytest",
                output_kind="patch",
            )
        ],
    )
    ledger.admit_verified_event(
        event.event_id,
        capture.verifications[0].verification_id,
        policy_version="test-v1",
        reason="test proof",
    )
    DatasetVersionRegistry(store).create(
        dataset_version_id="worker-dataset",
        name="worker",
        version="1",
        source_revision="b" * 40,
        split_policy={"kind": "test"},
        members=[
            DatasetMember(
                event_id=event.event_id,
                artifact_id=capture.artifacts[0].artifact_id,
                split=Split.TRAIN,
                lineage_id=event.lineage_id,
            )
        ],
    )
    return "worker-dataset"


def test_worker_executes_allowlisted_handler_and_marks_trained(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    dataset = _dataset(store, tmp_path)
    TrainingQueue(store).enqueue(
        job_id="job-1",
        job_kind="electronics_text",
        signals=_signals(),
        dataset_version_id=dataset,
    )
    marker = tmp_path / "marker"
    checkpoint = tmp_path / "checkpoint"
    checkpoint_sha256 = hashlib.sha256(b"candidate").hexdigest()
    handler_result = json.dumps(
        {
            "checkpoint_uri": checkpoint.as_uri(),
            "checkpoint_sha256": checkpoint_sha256,
        },
        separators=(",", ":"),
    )
    executable = _script(
        tmp_path / "handler",
        f'''test -s "$HARNESS_TRAINING_JOB_PAYLOAD"
printf %s "$HARNESS_TRAINING_JOB_ID" > "{marker}"
printf %s candidate > "{checkpoint}"
printf '%s' '{handler_result}' > "$HARNESS_TRAINING_RESULT"''',
    )
    handler = HandlerSpec(
        job_kind="electronics_text",
        argv=(str(executable),),
        working_directory=tmp_path,
        allowed_nodes=frozenset({"dgx2"}),
    )
    worker = TrainingWorker(
        store,
        node="dgx2",
        handlers={"electronics_text": handler},
        log_root=tmp_path / "logs",
        lease_seconds=30,
        heartbeat_seconds=1,
    )

    result = worker.run_once()

    assert result.status == "trained"
    assert marker.read_text() == "job-1"
    assert result.log_path is not None
    assert os.stat(result.log_path).st_mode & 0o077 == 0
    with store.connect() as conn:
        assert (
            conn.execute(
                "SELECT state FROM training_jobs WHERE job_id = 'job-1'"
            ).fetchone()[0]
            == "trained"
        )


def test_worker_failure_respects_retry_limit(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    dataset = _dataset(store, tmp_path)
    TrainingQueue(store).enqueue(
        job_id="job-1",
        job_kind="frozen_evaluation",
        signals=_signals(),
        dataset_version_id=dataset,
        max_attempts=1,
    )
    executable = _script(tmp_path / "failure", "exit 7")
    handler = HandlerSpec(
        job_kind="frozen_evaluation",
        argv=(str(executable),),
        working_directory=tmp_path,
        allowed_nodes=frozenset({"asus1"}),
    )
    worker = TrainingWorker(
        store,
        node="asus1",
        handlers={"frozen_evaluation": handler},
        log_root=tmp_path / "logs",
        lease_seconds=30,
        heartbeat_seconds=1,
    )

    result = worker.run_once()

    assert result.status == "rejected"
    assert result.returncode == 7


def test_worker_idles_when_node_has_no_matching_handler(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    dataset = _dataset(store, tmp_path)
    TrainingQueue(store).enqueue(
        job_id="job-1",
        job_kind="electronics_text",
        signals=_signals(),
        dataset_version_id=dataset,
    )
    handler = HandlerSpec(
        job_kind="electronics_text",
        argv=(str(_script(tmp_path / "handler", "exit 0")),),
        working_directory=tmp_path,
        allowed_nodes=frozenset({"dgx2"}),
    )
    worker = TrainingWorker(
        store,
        node="asus1",
        handlers={"electronics_text": handler},
        log_root=tmp_path / "logs",
        lease_seconds=30,
        heartbeat_seconds=1,
    )

    assert worker.run_once().status == "idle"


def test_restarted_worker_reaps_expired_owned_handler_before_reassignment(
    tmp_path: Path,
    monkeypatch,
):
    store = Store(tmp_path / "harness.db")
    dataset = _dataset(store, tmp_path)
    queue = TrainingQueue(store)
    queue.enqueue(
        job_id="orphaned",
        job_kind="electronics_text",
        signals=_signals(),
        dataset_version_id=dataset,
        max_attempts=2,
    )
    started = datetime.now(timezone.utc) - timedelta(minutes=2)
    claimed = queue.claim("dgx2", lease_seconds=30, now=started)
    assert claimed is not None
    queue.attach_handler(
        claimed.job_id,
        node=claimed.assigned_node,
        attempt=claimed.attempt,
        lease_token=claimed.lease_token,
        pid=23456,
        pgid=23456,
        now=started,
    )
    terminated: list[int] = []
    monkeypatch.setattr(
        "harness.training.worker._process_group_exists",
        lambda _pgid: True,
    )
    monkeypatch.setattr(
        "harness.training.worker._handler_identity_matches",
        lambda _pid, _job_id, _attempt: True,
    )
    monkeypatch.setattr(
        "harness.training.worker._terminate_process_group_id",
        terminated.append,
    )
    worker = TrainingWorker(
        store,
        node="dgx2",
        handlers={},
        log_root=tmp_path / "logs",
        lease_seconds=30,
        heartbeat_seconds=1,
    )

    worker._reap_expired_handlers()

    assert terminated == [23456]
    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT state, handler_pid, handler_pgid
            FROM training_jobs WHERE job_id = 'orphaned'
            """
        ).fetchone()
    assert row["state"] == "eligible"
    assert row["handler_pid"] is None
    assert row["handler_pgid"] is None


def test_handler_rejects_shell_entrypoint(tmp_path: Path):
    with pytest.raises(ValidationError, match="not a shell"):
        HandlerSpec(
            job_kind="bad",
            argv=("/bin/sh", "-c", "anything"),
            working_directory=tmp_path,
            allowed_nodes=frozenset({"dgx2"}),
        )
