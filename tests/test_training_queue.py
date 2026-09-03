from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from harness.storage.db import Store
from harness.training.ledger import (
    ArtifactPayload,
    LearningLedger,
    VerificationPayload,
)
from harness.training.models import LearningEvent, SourceKind
from harness.training.queue import (
    DatasetMember,
    DatasetVersionRegistry,
    InvalidTransitionError,
    JobState,
    PrioritySignals,
    QueueError,
    TrainingQueue,
)
from harness.training.split import Split


NOW = datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)


def _member(
    ledger: LearningLedger,
    index: int,
    split: Split,
    *,
    lineage: str | None = None,
    document: str | None = None,
    repository: str | None = None,
    family: str | None = None,
    temporal_bucket: str | None = None,
) -> DatasetMember:
    event = LearningEvent(
        event_id=f"event-{index}",
        event_type="verified_repair",
        source_kind=SourceKind.GIT,
        source_uri=f"git+file:///owned/repo#{index}",
        source_revision=f"{index:040x}",
        lineage_id=lineage or f"lineage-{index}",
        authorization_scope="configured-owned-repository",
        created_at=NOW,
        metadata={"data_use": "training", "disposition": "verified"},
    )
    result = ledger.capture(
        event,
        [
            ArtifactPayload(
                kind="verified_patch",
                content=f"diff --git a/a{index} b/a{index}",
            )
        ],
        [
            VerificationPayload(
                kind="pytest",
                status="pass",
                verifier="pytest",
                output_kind="verified_patch",
            )
        ],
    )
    ledger.admit_verified_event(
        event.event_id,
        result.verifications[0].verification_id,
        policy_version="test-v1",
        reason="test fixture proof",
    )
    return DatasetMember(
        event_id=event.event_id,
        artifact_id=result.artifacts[0].artifact_id,
        split=split,
        lineage_id=event.lineage_id,
        source_document_sha256=document,
        repository_id=repository,
        component_family=family,
        temporal_bucket=temporal_bucket,
    )


def _dataset(store: Store, ledger: LearningLedger) -> str:
    registry = DatasetVersionRegistry(store)
    registry.create(
        dataset_version_id="repairs-v1",
        name="repairs",
        version="1",
        source_revision="f" * 40,
        split_policy={"kind": "lineage_hash", "seed": "test"},
        members=[
            _member(ledger, 1, Split.TRAIN),
            _member(ledger, 2, Split.TEST),
        ],
    )
    return "repairs-v1"


def _signals(*, cost: float) -> PrioritySignals:
    return PrioritySignals(
        observed_frequency=2,
        frontier_cost=cost,
        local_failure_rate=0.5,
        verification_strength=1,
        diversity=1,
        expected_gpu_hours=2,
    )


def test_dataset_version_name_cannot_cross_suspended_data_boundary(
    tmp_path: Path,
):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    members = [
        _member(ledger, 1, Split.TRAIN),
        _member(ledger, 2, Split.TEST),
    ]

    with pytest.raises(ValueError, match="dataset versions are disabled"):
        DatasetVersionRegistry(store).create(
            dataset_version_id="excluded-v1",
            name="category rank repairs",
            version="1",
            source_revision="f" * 40,
            split_policy={"kind": "lineage_hash"},
            members=members,
        )


def test_dataset_member_is_rechecked_for_suspended_source_at_admission(
    tmp_path: Path,
):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    member = _member(ledger, 1, Split.TRAIN)
    holdout = _member(ledger, 2, Split.TEST)
    with store.connect() as conn:
        conn.execute("DROP TRIGGER learning_events_no_update")
        conn.execute(
            """
            UPDATE learning_events
            SET source_uri = 'git+file:///owned/tapes/export'
            WHERE event_id = ?
            """,
            (member.event_id,),
        )

    with pytest.raises(ValueError, match="dataset admission is disabled"):
        DatasetVersionRegistry(store).create(
            dataset_version_id="tampered-v1",
            name="repairs",
            version="1",
            source_revision="f" * 40,
            split_policy={"kind": "lineage_hash"},
            members=[member, holdout],
        )


def test_training_job_cannot_cross_suspended_data_boundary(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    dataset = _dataset(store, ledger)

    with pytest.raises(ValueError, match="training jobs are disabled"):
        TrainingQueue(store).enqueue(
            job_id="job-1",
            job_kind="tapes-reproduction",
            dataset_version_id=dataset,
            signals=_signals(cost=1),
        )


def test_dataset_version_is_immutable_and_lineage_safe(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    first = _member(ledger, 1, Split.TRAIN)
    second = _member(ledger, 2, Split.TEST)
    registry = DatasetVersionRegistry(store)

    digest = registry.create(
        dataset_version_id="dataset-v1",
        name="dataset",
        version="1",
        source_revision="f" * 40,
        split_policy={"kind": "lineage_hash"},
        members=[first, second],
    )
    assert len(digest) == 64
    assert (
        registry.create(
            dataset_version_id="dataset-v1",
            name="dataset",
            version="1",
            source_revision="f" * 40,
            split_policy={"kind": "lineage_hash"},
            members=[first, second],
        )
        == digest
    )

    third = _member(ledger, 3, Split.VALIDATION)
    with pytest.raises(QueueError, match="immutable"):
        registry.create(
            dataset_version_id="dataset-v1",
            name="dataset",
            version="1",
            source_revision="f" * 40,
            split_policy={"kind": "lineage_hash"},
            members=[first, second, third],
        )


def test_dataset_rejects_lineage_and_document_leakage(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    registry = DatasetVersionRegistry(store)
    shared = "a" * 64

    with pytest.raises(ValueError, match="leaks across"):
        registry.create(
            dataset_version_id="lineage-leak",
            name="bad",
            version="1",
            source_revision="f" * 40,
            split_policy={},
            members=[
                _member(ledger, 1, Split.TRAIN, lineage="same"),
                _member(ledger, 2, Split.TEST, lineage="same"),
            ],
        )

    with pytest.raises(ValueError, match="canonical"):
        DatasetMember(
            event_id="event",
            artifact_id="artifact",
            split=Split.TRAIN,
            lineage_id=" padded ",
        )
    with pytest.raises(ValueError, match="source document"):
        registry.create(
            dataset_version_id="document-leak",
            name="bad",
            version="2",
            source_revision="f" * 40,
            split_policy={},
            members=[
                _member(ledger, 3, Split.TRAIN, document=shared),
                _member(ledger, 4, Split.TEST, document=shared),
            ],
        )


def test_dataset_requires_and_isolates_declared_leakage_dimensions(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    registry = DatasetVersionRegistry(store)

    with pytest.raises(ValueError, match="missing declared leakage key"):
        registry.create(
            dataset_version_id="missing-family",
            name="bad",
            version="1",
            source_revision="f" * 40,
            split_policy={"leakage_keys": ["lineage_id", "component_family"]},
            members=[
                _member(ledger, 1, Split.TRAIN),
                _member(ledger, 2, Split.TEST, family="mcu"),
            ],
        )

    with pytest.raises(ValueError, match="component_family.*leaks across"):
        registry.create(
            dataset_version_id="family-leak",
            name="bad",
            version="2",
            source_revision="f" * 40,
            split_policy={"leakage_keys": ["lineage_id", "component_family"]},
            members=[
                _member(ledger, 3, Split.TRAIN, family="mcu"),
                _member(ledger, 4, Split.TEST, family="mcu"),
            ],
        )


def test_dataset_rejects_event_without_admission(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    event = LearningEvent(
        event_id="unadmitted",
        event_type="candidate",
        source_kind=SourceKind.GIT,
        source_uri="git+file:///owned/repo?revision=" + "a" * 40,
        source_revision="a" * 40,
        lineage_id="repo",
        authorization_scope="test",
        created_at=NOW,
        metadata={"data_use": "training", "disposition": "verified"},
    )
    capture = ledger.capture(
        event,
        [ArtifactPayload(kind="patch", content="diff --git a/a b/a")],
    )

    with pytest.raises(ValueError, match="lacks verified admission"):
        DatasetVersionRegistry(store).create(
            dataset_version_id="bad",
            name="bad",
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

def test_queue_is_priority_ordered_and_bounded_by_retries(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    _dataset(store, ledger)
    queue = TrainingQueue(store)
    queue.enqueue(
        job_id="low",
        job_kind="ablation",
        dataset_version_id="repairs-v1",
        signals=_signals(cost=1),
    )
    queue.enqueue(
        job_id="high",
        job_kind="production_failure_replay",
        dataset_version_id="repairs-v1",
        signals=_signals(cost=10),
        max_attempts=2,
    )

    first = queue.claim("dgx2", now=NOW)
    assert first is not None and first.job_id == "high"
    assert (
        queue.fail(
            "high",
            "transient",
            node="dgx2",
            attempt=first.attempt,
            lease_token=first.lease_token,
            now=NOW,
        )
        is JobState.ELIGIBLE
    )
    second = queue.claim("dgx2", now=NOW)
    assert second is not None and second.job_id == "high"
    assert (
        queue.fail(
            "high",
            "repeat failure",
            node="dgx2",
            attempt=second.attempt,
            lease_token=second.lease_token,
            now=NOW,
        )
        is JobState.REJECTED
    )
    third = queue.claim("dgx2", now=NOW)
    assert third is not None and third.job_id == "low"
    checkpoint = tmp_path / "low.checkpoint"
    checkpoint.write_bytes(b"verified low checkpoint")
    queue.complete(
        "low",
        node="dgx2",
        attempt=third.attempt,
        lease_token=third.lease_token,
        checkpoint_uri=checkpoint.as_uri(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        now=NOW,
    )
    assert queue.claim("asus1", now=NOW) is None


def test_queue_rejects_duplicate_dataset_experiment(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    dataset = _dataset(store, ledger)
    queue = TrainingQueue(store)
    queue.enqueue(
        job_id="first",
        job_kind="electronics_text",
        dataset_version_id=dataset,
        signals=_signals(cost=1),
        config={"epochs": 1},
    )

    with pytest.raises(QueueError, match="identical.*experiment"):
        queue.enqueue(
            job_id="duplicate",
            job_kind="electronics_text",
            dataset_version_id=dataset,
            signals=_signals(cost=10),
            config={"epochs": 1},
        )

    queue.enqueue(
        job_id="declared-ablation",
        job_kind="electronics_text",
        dataset_version_id=dataset,
        signals=_signals(cost=1),
        config={"epochs": 2},
    )


def test_zero_priority_signal_is_rejected() -> None:
    with pytest.raises(ValueError, match="priority score must be positive"):
        _signals(cost=0)


def test_expired_leases_recover_then_reject(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    dataset = _dataset(store, ledger)
    queue = TrainingQueue(store)
    queue.enqueue(
        job_id="lease",
        job_kind="evaluation",
        dataset_version_id=dataset,
        signals=_signals(cost=1),
        max_attempts=2,
    )

    assert queue.claim("dgx2", lease_seconds=30, now=NOW) is not None
    assert queue.recover_expired(now=NOW + timedelta(seconds=31)) == 1
    assert queue.claim("asus1", lease_seconds=30, now=NOW + timedelta(minutes=1))
    assert queue.recover_expired(now=NOW + timedelta(minutes=2)) == 1
    assert queue.claim("dgx2", now=NOW + timedelta(minutes=3)) is None


def test_expired_attached_handler_blocks_reassignment_until_reaped(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    dataset = _dataset(store, ledger)
    queue = TrainingQueue(store)
    queue.enqueue(
        job_id="orphan-safe",
        job_kind="main_model_lora",
        dataset_version_id=dataset,
        signals=_signals(cost=1),
        max_attempts=2,
    )
    claimed = queue.claim("dgx2", lease_seconds=30, now=NOW)
    assert claimed is not None
    queue.attach_handler(
        claimed.job_id,
        node=claimed.assigned_node,
        attempt=claimed.attempt,
        lease_token=claimed.lease_token,
        pid=12345,
        pgid=12345,
        now=NOW,
    )

    expired_at = NOW + timedelta(seconds=31)
    assert queue.recover_expired(now=expired_at) == 0
    assert queue.claim("asus1", now=expired_at) is None
    handlers = queue.expired_handlers("dgx2", now=expired_at)
    assert len(handlers) == 1
    assert queue.release_expired_handler(
        handlers[0],
        now=expired_at,
    ) is JobState.ELIGIBLE
    reassigned = queue.claim("asus1", now=expired_at)
    assert reassigned is not None
    assert reassigned.job_id == "orphan-safe"


def test_queue_class_order_precedes_numeric_priority(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    dataset = _dataset(store, ledger)
    queue = TrainingQueue(store)
    queue.enqueue(
        job_id="sweep",
        job_kind="hyperparameter_sweep",
        dataset_version_id=dataset,
        signals=_signals(cost=1000),
    )
    queue.enqueue(
        job_id="replay",
        job_kind="production_failure_replay",
        dataset_version_id=dataset,
        signals=_signals(cost=0.001),
    )

    claimed = queue.claim("dgx2", now=NOW)

    assert claimed is not None and claimed.job_id == "replay"
    renewed = queue.renew(
        "replay",
        "dgx2",
        claimed.attempt,
        claimed.lease_token,
        lease_seconds=60,
        now=NOW + timedelta(seconds=10),
    )
    assert renewed == (NOW + timedelta(seconds=70)).isoformat(timespec="seconds")


def test_expired_worker_cannot_complete_reassigned_attempt(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    dataset = _dataset(store, ledger)
    queue = TrainingQueue(store)
    queue.enqueue(
        job_id="lease-owner",
        job_kind="main_model_lora",
        dataset_version_id=dataset,
        signals=_signals(cost=1),
        max_attempts=2,
    )
    stale = queue.claim("dgx2", lease_seconds=30, now=NOW)
    assert stale is not None
    queue.recover_expired(now=NOW + timedelta(seconds=31))
    current = queue.claim(
        "asus1",
        lease_seconds=30,
        now=NOW + timedelta(seconds=32),
    )
    assert current is not None
    stale_checkpoint = tmp_path / "stale.checkpoint"
    stale_checkpoint.write_bytes(b"stale")
    current_checkpoint = tmp_path / "current.checkpoint"
    current_checkpoint.write_bytes(b"current")

    with pytest.raises(InvalidTransitionError, match="current unexpired lease"):
        queue.complete(
            "lease-owner",
            node=stale.assigned_node,
            attempt=stale.attempt,
            lease_token=stale.lease_token,
            checkpoint_uri=stale_checkpoint.as_uri(),
            checkpoint_sha256=hashlib.sha256(
                stale_checkpoint.read_bytes()
            ).hexdigest(),
            now=NOW + timedelta(seconds=33),
        )
    queue.complete(
        "lease-owner",
        node=current.assigned_node,
        attempt=current.attempt,
        lease_token=current.lease_token,
        checkpoint_uri=current_checkpoint.as_uri(),
        checkpoint_sha256=hashlib.sha256(
            current_checkpoint.read_bytes()
        ).hexdigest(),
        now=NOW + timedelta(seconds=33),
    )


def test_completion_rejects_unverified_checkpoint_claim(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    dataset = _dataset(store, ledger)
    queue = TrainingQueue(store)
    queue.enqueue(
        job_id="fake-checkpoint",
        job_kind="main_model_lora",
        dataset_version_id=dataset,
        signals=_signals(cost=1),
    )
    claimed = queue.claim("dgx2", now=NOW)
    assert claimed is not None
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"real bytes")

    with pytest.raises(ValueError, match="does not match"):
        queue.complete(
            claimed.job_id,
            node=claimed.assigned_node,
            attempt=claimed.attempt,
            lease_token=claimed.lease_token,
            checkpoint_uri=checkpoint.as_uri(),
            checkpoint_sha256="0" * 64,
            now=NOW,
        )


def test_database_trigger_rejects_direct_promotion_bypass(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    dataset = _dataset(store, ledger)
    queue = TrainingQueue(store)
    queue.enqueue(
        job_id="bypass",
        job_kind="main_model_lora",
        dataset_version_id=dataset,
        signals=_signals(cost=1),
    )

    with store.connect() as conn:
        with pytest.raises(
            sqlite3.IntegrityError,
            match="state transition|promotion requires",
        ):
            conn.execute(
                "UPDATE training_jobs SET state = 'promoted' WHERE job_id = ?",
                ("bypass",),
            )


def test_database_guards_training_job_birth_and_deletion(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    dataset = _dataset(store, ledger)
    queue = TrainingQueue(store)
    queue.enqueue(
        job_id="immutable-job",
        job_kind="main_model_lora",
        dataset_version_id=dataset,
        signals=_signals(cost=1),
    )

    with store.connect() as conn:
        with pytest.raises(sqlite3.IntegrityError, match="invalid training job birth"):
            conn.execute(
                """
                INSERT INTO training_jobs (
                    job_id, job_kind, dataset_version_id, state, priority,
                    expected_gpu_hours, max_attempts, config_json,
                    experiment_sha256, created_at, updated_at
                ) VALUES (
                    'forged', 'main_model_lora', ?, 'promoted', 1,
                    1, 1, '{}', ?, ?, ?
                )
                """,
                (dataset, "a" * 64, NOW.isoformat(), NOW.isoformat()),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable history"):
            conn.execute(
                "DELETE FROM training_jobs WHERE job_id = 'immutable-job'"
            )
