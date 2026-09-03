from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.storage.db import Store
from harness.training.ledger import ArtifactPayload, LearningLedger, VerificationPayload
from harness.training.models import LearningEvent, SourceKind
from harness.training.promotion import (
    CandidateEvaluation,
    PromotionError,
    decide_promotion,
    record_evaluation,
)
from harness.training.queue import (
    DatasetMember,
    DatasetVersionRegistry,
    InvalidTransitionError,
    JobState,
    PrioritySignals,
    TrainingQueue,
)
from harness.training.split import Split


NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)
CHECKPOINT_SHA256 = hashlib.sha256(b"candidate").hexdigest()


def _evaluation(**updates) -> CandidateEvaluation:
    values = {
        "task_class": "coding",
        "frozen_holdout_id": "dataset-v1",
        "holdout_disjoint": True,
        "sample_count": 100,
        "verified_success_rate": 0.80,
        "baseline_verified_success_rate": 0.75,
        "frontier_escalation_rate": 0.15,
        "baseline_frontier_escalation_rate": 0.20,
        "cost_per_verified_success": 0.10,
        "baseline_cost_per_verified_success": 0.20,
        "p95_latency_ms": 1050,
        "baseline_p95_latency_ms": 1000,
        "first_pass_success_rate": 0.70,
        "mean_repair_cycles": 1.2,
        "critical_regressions": 0,
        "checkpoint_reproducible": True,
        "resume_verified": True,
        "candidate_sha256": CHECKPOINT_SHA256,
        "gpu_hours": 4,
    }
    values.update(updates)
    return CandidateEvaluation(**values)


def _store_with_job(tmp_path: Path) -> Store:
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    event = LearningEvent(
        event_id="event-1",
        event_type="verified_repair",
        source_kind=SourceKind.HARNESS,
        source_uri="harness://events/1",
        source_revision="e" * 40,
        lineage_id="lineage-1",
        authorization_scope="test",
        created_at=NOW,
        metadata={"data_use": "training", "disposition": "verified"},
    )
    capture = ledger.capture(
        event,
        [ArtifactPayload(kind="pair", content="verified pair")],
        [
            VerificationPayload(
                kind="pytest",
                status="pass",
                verifier="pytest",
                output_kind="pair",
            )
        ],
    )
    ledger.admit_verified_event(
        event.event_id,
        capture.verifications[0].verification_id,
        policy_version="test-v1",
        reason="test proof",
    )
    holdout_event = event.model_copy(
        update={
            "event_id": "event-test",
            "source_uri": "harness://events/test",
            "source_revision": "d" * 40,
            "lineage_id": "lineage-test",
        }
    )
    holdout_capture = ledger.capture(
        holdout_event,
        [ArtifactPayload(kind="pair", content="verified holdout pair")],
        [
            VerificationPayload(
                kind="pytest",
                status="pass",
                verifier="pytest",
                output_kind="pair",
            )
        ],
    )
    ledger.admit_verified_event(
        holdout_event.event_id,
        holdout_capture.verifications[0].verification_id,
        policy_version="test-v1",
        reason="test proof",
    )
    DatasetVersionRegistry(store).create(
        dataset_version_id="dataset-v1",
        name="repairs",
        version="1",
        source_revision="f" * 40,
        split_policy={"kind": "frozen"},
        members=[
            DatasetMember(
                event_id=event.event_id,
                artifact_id=capture.artifacts[0].artifact_id,
                split=Split.TRAIN,
                lineage_id=event.lineage_id,
            ),
            DatasetMember(
                event_id=holdout_event.event_id,
                artifact_id=holdout_capture.artifacts[0].artifact_id,
                split=Split.TEST,
                lineage_id=holdout_event.lineage_id,
            ),
        ],
    )
    TrainingQueue(store).enqueue(
        job_id="job-1",
        job_kind="qlora",
        dataset_version_id="dataset-v1",
        signals=PrioritySignals(
            observed_frequency=1,
            frontier_cost=1,
            local_failure_rate=1,
            verification_strength=1,
            diversity=1,
            expected_gpu_hours=1,
        ),
    )
    queue = TrainingQueue(store)
    claimed = queue.claim("dgx2", now=NOW)
    assert claimed is not None
    checkpoint = tmp_path / "candidate.adapter"
    checkpoint.write_bytes(b"candidate")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_sha256 == CHECKPOINT_SHA256
    queue.complete(
        "job-1",
        node="dgx2",
        attempt=claimed.attempt,
        lease_token=claimed.lease_token,
        checkpoint_uri=checkpoint.as_uri(),
        checkpoint_sha256=checkpoint_sha256,
        now=NOW,
    )
    return store


def test_offline_shadow_and_canary_require_quantitative_gain():
    evaluation = _evaluation()

    offline = decide_promotion(evaluation, stage="offline")
    shadow = decide_promotion(evaluation, stage="shadow")
    canary = decide_promotion(evaluation, stage="canary")

    assert offline.passed is True
    assert offline.action == "shadow"
    assert shadow.passed is True
    assert shadow.action == "canary"
    assert canary.passed is True
    assert canary.action == "promote"


def test_promotion_rejects_cost_and_regression_failures():
    evaluation = _evaluation(
        cost_per_verified_success=0.30,
        critical_regressions=1,
    )

    decision = decide_promotion(evaluation, stage="offline")

    assert decision.passed is False
    assert decision.action == "reject"
    assert "no_critical_regressions" in decision.reasons
    assert "cost_per_success_improved" in decision.reasons


def test_evaluation_record_is_immutable(tmp_path: Path):
    store = _store_with_job(tmp_path)
    evaluation = _evaluation()

    digest = record_evaluation(
        store,
        evaluation_id="eval-1",
        job_id="job-1",
        dataset_version_id="dataset-v1",
        evaluation=evaluation,
        stage="offline",
        created_at=NOW,
    )
    assert (
        record_evaluation(
            store,
            evaluation_id="eval-1",
            job_id="job-1",
            dataset_version_id="dataset-v1",
            evaluation=evaluation,
            stage="offline",
            created_at=NOW,
        )
        == digest
    )

    changed = _evaluation(verified_success_rate=0.90)
    with pytest.raises(PromotionError, match="immutable"):
        record_evaluation(
            store,
            evaluation_id="eval-1",
            job_id="job-1",
            dataset_version_id="dataset-v1",
            evaluation=changed,
            stage="offline",
            created_at=NOW,
        )


def test_rejected_stage_is_sticky_and_cannot_be_superseded(tmp_path: Path):
    store = _store_with_job(tmp_path)
    rejected = _evaluation(
        verified_success_rate=0.70,
        critical_regressions=1,
    )
    record_evaluation(
        store,
        evaluation_id="offline-reject",
        job_id="job-1",
        dataset_version_id="dataset-v1",
        evaluation=rejected,
        stage="offline",
        created_at=NOW,
    )

    with pytest.raises(PromotionError, match="offline evaluation.*immutable"):
        record_evaluation(
            store,
            evaluation_id="offline-pass",
            job_id="job-1",
            dataset_version_id="dataset-v1",
            evaluation=_evaluation(),
            stage="offline",
            created_at=NOW,
        )
    with pytest.raises(sqlite3.IntegrityError, match="evaluation job stage"):
        with store.connect() as conn:
            conn.execute(
                """
                INSERT INTO evaluation_results
                SELECT 'raw-bypass', job_id, dataset_version_id,
                       candidate_sha256, metrics_json, decision,
                       ?, gpu_hours, created_at
                FROM evaluation_results WHERE evaluation_id = 'offline-reject'
                """,
                ("f" * 64,),
            )
    with pytest.raises(InvalidTransitionError, match="passing offline evidence"):
        TrainingQueue(store).transition("job-1", JobState.EVALUATED)


def test_full_promotion_path_requires_stage_specific_evidence(tmp_path: Path):
    store = _store_with_job(tmp_path)
    queue = TrainingQueue(store)
    for evaluation_id, stage, target in (
        ("offline-pass", "offline", JobState.EVALUATED),
        ("shadow-pass", "shadow", JobState.CANARY),
        ("canary-pass", "canary", JobState.PROMOTED),
    ):
        metadata = (
            {}
            if stage == "offline"
            else {
                "dual_run_derived": True,
                "dual_run_schema": "harness.dual-run.v1",
                "dual_run_log_sha256": "c" * 64,
            }
        )
        record_evaluation(
            store,
            evaluation_id=evaluation_id,
            job_id="job-1",
            dataset_version_id="dataset-v1",
            evaluation=_evaluation(metadata=metadata),
            stage=stage,
            created_at=NOW,
        )
        if stage == "offline":
            queue.transition("job-1", JobState.EVALUATED)
            queue.transition("job-1", JobState.SHADOW)
        else:
            queue.transition("job-1", target)

    with store.connect() as conn:
        state = conn.execute(
            "SELECT state FROM training_jobs WHERE job_id = 'job-1'"
        ).fetchone()["state"]
    assert state == "promoted"


def test_shadow_metrics_without_dual_run_evidence_are_rejected(tmp_path: Path):
    store = _store_with_job(tmp_path)
    queue = TrainingQueue(store)
    record_evaluation(
        store,
        evaluation_id="offline-pass",
        job_id="job-1",
        dataset_version_id="dataset-v1",
        evaluation=_evaluation(),
        stage="offline",
        created_at=NOW,
    )
    queue.transition("job-1", JobState.EVALUATED)
    queue.transition("job-1", JobState.SHADOW)

    with pytest.raises(PromotionError, match="hash-bound dual-run evidence"):
        record_evaluation(
            store,
            evaluation_id="hand-authored-shadow",
            job_id="job-1",
            dataset_version_id="dataset-v1",
            evaluation=_evaluation(),
            stage="shadow",
            created_at=NOW,
        )


def test_electronics_offline_pass_requires_sealed_qualification(
    tmp_path: Path,
):
    store = _store_with_job(tmp_path)
    evaluation = _evaluation(task_class="electronics_pinout_extraction")

    with pytest.raises(PromotionError, match="sealed qualification evidence"):
        record_evaluation(
            store,
            evaluation_id="electronics-pass",
            job_id="job-1",
            dataset_version_id="dataset-v1",
            evaluation=evaluation,
            stage="offline",
            created_at=NOW,
        )

    qualified = evaluation.model_copy(
        update={
            "metadata": {
                "qualification_sha256": "a" * 64,
                "qualification_core_sha256": "b" * 64,
                "strict_qualification_passed": True,
            }
        }
    )
    digest = record_evaluation(
        store,
        evaluation_id="electronics-pass",
        job_id="job-1",
        dataset_version_id="dataset-v1",
        evaluation=qualified,
        stage="offline",
        created_at=NOW,
    )
    assert len(digest) == 64
