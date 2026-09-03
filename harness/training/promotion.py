from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.storage.db import Store
from harness.training.registry import canonical_json
from harness.training.security import assert_no_secrets, assert_value_no_secrets


class PromotionError(RuntimeError):
    pass


class PromotionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_verified_success_gain: float = Field(default=0.02, ge=0, le=1)
    maximum_latency_regression: float = Field(default=0.10, ge=0)
    minimum_offline_samples: int = Field(default=1, ge=1)
    minimum_shadow_samples: int = Field(default=50, ge=1)
    minimum_canary_samples: int = Field(default=100, ge=1)
    require_lower_cost_per_success: bool = True


class CandidateEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_class: str = Field(min_length=1)
    frozen_holdout_id: str = Field(min_length=1)
    holdout_disjoint: bool
    sample_count: int = Field(ge=1)
    verified_success_rate: float = Field(ge=0, le=1)
    baseline_verified_success_rate: float = Field(ge=0, le=1)
    frontier_escalation_rate: float = Field(ge=0, le=1)
    baseline_frontier_escalation_rate: float = Field(ge=0, le=1)
    cost_per_verified_success: float = Field(ge=0)
    baseline_cost_per_verified_success: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    baseline_p95_latency_ms: float = Field(ge=0)
    first_pass_success_rate: float = Field(ge=0, le=1)
    mean_repair_cycles: float = Field(ge=0)
    critical_regressions: int = Field(ge=0)
    checkpoint_reproducible: bool
    resume_verified: bool
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gpu_hours: float = Field(ge=0)
    pinout_leaf_f1: float | None = Field(default=None, ge=0, le=1)
    pinout_exact_rate: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def baseline_latency_is_defined(self) -> CandidateEvaluation:
        if self.baseline_p95_latency_ms == 0 and self.p95_latency_ms > 0:
            raise ValueError("nonzero candidate latency requires a baseline latency")
        assert_no_secrets(self.task_class, field="evaluation task_class")
        assert_no_secrets(
            self.frozen_holdout_id,
            field="evaluation frozen_holdout_id",
        )
        assert_value_no_secrets(self.metadata, field="evaluation metadata")
        return self


class PromotionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    action: Literal["shadow", "canary", "promote", "reject"]
    checks: dict[str, bool]
    reasons: tuple[str, ...]
    metrics: dict[str, float | int | bool | str | None]


def _holdout_evidence(
    conn: sqlite3.Connection,
    dataset_version_id: str,
) -> dict[str, int | bool]:
    test_members = conn.execute(
        """
        SELECT COUNT(*) FROM dataset_members
        WHERE dataset_version_id = ? AND split = 'test'
        """,
        (dataset_version_id,),
    ).fetchone()[0]
    if test_members < 1:
        raise PromotionError("evaluation dataset has no frozen test members")
    leakage = conn.execute(
        """
        SELECT 1
        FROM dataset_members AS train
        JOIN dataset_members AS test
          ON test.dataset_version_id = train.dataset_version_id
         AND test.split = 'test'
         AND train.split = 'train'
         AND (
           test.lineage_id = train.lineage_id
           OR (
             test.source_document_sha256 IS NOT NULL
             AND train.source_document_sha256 IS NOT NULL
             AND test.source_document_sha256 = train.source_document_sha256
           )
         )
        WHERE train.dataset_version_id = ?
        LIMIT 1
        """,
        (dataset_version_id,),
    ).fetchone()
    if leakage is not None:
        raise PromotionError("frozen holdout leaks into the training split")
    return {
        "computed_disjoint": True,
        "test_members": int(test_members),
    }


def _require_specialized_electronics_evidence(
    evaluation: CandidateEvaluation,
    *,
    stage: str,
    decision: PromotionDecision,
) -> None:
    if (
        evaluation.task_class != "electronics_pinout_extraction"
        or stage != "offline"
    ):
        return
    metadata = evaluation.metadata
    digest = metadata.get("qualification_sha256") or metadata.get(
        "canary_qualification_sha256"
    )
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise PromotionError(
            "electronics evaluation requires sealed qualification evidence"
        )
    if decision.passed and metadata.get("strict_qualification_passed") is not True:
        raise PromotionError(
            "electronics promotion requires a passing strict qualification"
        )
    if decision.passed:
        core_digest = metadata.get("qualification_core_sha256")
        if (
            not isinstance(core_digest, str)
            or len(core_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in core_digest
            )
        ):
            raise PromotionError(
                "electronics promotion requires qualification core evidence"
            )


def _require_dual_run_evidence(
    evaluation: CandidateEvaluation,
    *,
    stage: str,
) -> None:
    if stage == "offline":
        return
    metadata = evaluation.metadata
    digest = metadata.get("dual_run_log_sha256")
    if (
        metadata.get("dual_run_derived") is not True
        or metadata.get("dual_run_schema") != "harness.dual-run.v1"
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise PromotionError(
            f"{stage} evaluation requires hash-bound dual-run evidence"
        )


def decide_promotion(
    evaluation: CandidateEvaluation,
    *,
    stage: Literal["offline", "shadow", "canary"],
    policy: PromotionPolicy = PromotionPolicy(),
) -> PromotionDecision:
    success_gain = (
        evaluation.verified_success_rate
        - evaluation.baseline_verified_success_rate
    )
    latency_limit = evaluation.baseline_p95_latency_ms * (
        1 + policy.maximum_latency_regression
    )
    checks = {
        "frozen_holdout_disjoint": evaluation.holdout_disjoint,
        "minimum_verified_success_gain": (
            success_gain >= policy.minimum_verified_success_gain
        ),
        "no_critical_regressions": evaluation.critical_regressions == 0,
        "checkpoint_reproducible": evaluation.checkpoint_reproducible,
        "resume_verified": evaluation.resume_verified,
        "latency_within_bound": evaluation.p95_latency_ms <= latency_limit,
        "cost_per_success_improved": (
            not policy.require_lower_cost_per_success
            or evaluation.cost_per_verified_success
            < evaluation.baseline_cost_per_verified_success
        ),
        "frontier_escalation_non_regression": (
            evaluation.frontier_escalation_rate
            <= evaluation.baseline_frontier_escalation_rate
        ),
        "minimum_stage_samples": evaluation.sample_count
        >= {
            "offline": policy.minimum_offline_samples,
            "shadow": policy.minimum_shadow_samples,
            "canary": policy.minimum_canary_samples,
        }[stage],
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    passed = not reasons
    return PromotionDecision(
        passed=passed,
        action={
            "offline": "shadow",
            "shadow": "canary",
            "canary": "promote",
        }[stage]
        if passed
        else "reject",
        checks=checks,
        reasons=reasons,
        metrics={
            "task_class": evaluation.task_class,
            "sample_count": evaluation.sample_count,
            "verified_success_gain": success_gain,
            "frontier_escalation_delta": (
                evaluation.frontier_escalation_rate
                - evaluation.baseline_frontier_escalation_rate
            ),
            "cost_per_success_delta": (
                evaluation.cost_per_verified_success
                - evaluation.baseline_cost_per_verified_success
            ),
            "p95_latency_delta_ms": (
                evaluation.p95_latency_ms
                - evaluation.baseline_p95_latency_ms
            ),
            "gpu_hours": evaluation.gpu_hours,
            "pinout_leaf_f1": evaluation.pinout_leaf_f1,
            "pinout_exact_rate": evaluation.pinout_exact_rate,
        },
    )


def record_evaluation(
    store: Store,
    *,
    evaluation_id: str,
    job_id: str,
    dataset_version_id: str,
    evaluation: CandidateEvaluation,
    stage: Literal["offline", "shadow", "canary"],
    policy: PromotionPolicy = PromotionPolicy(),
    created_at: datetime | None = None,
) -> str:
    decision = decide_promotion(evaluation, stage=stage, policy=policy)
    _require_specialized_electronics_evidence(
        evaluation,
        stage=stage,
        decision=decision,
    )
    _require_dual_run_evidence(evaluation, stage=stage)
    timestamp = (created_at or datetime.now(timezone.utc)).astimezone(
        timezone.utc
    )
    with store.connect() as conn:
        holdout_evidence = _holdout_evidence(conn, dataset_version_id)
    payload = {
        "schema": "harness.evaluation-result.v1",
        "evaluation_id": evaluation_id,
        "job_id": job_id,
        "dataset_version_id": dataset_version_id,
        "evaluation": evaluation.model_dump(mode="json"),
        "stage": stage,
        "policy": policy.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
        "holdout_evidence": holdout_evidence,
        "created_at": timestamp.isoformat(timespec="seconds"),
    }
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        job = conn.execute(
            """
            SELECT state, dataset_version_id, checkpoint_sha256
            FROM training_jobs WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if job is None:
            raise PromotionError(f"unknown training job {job_id!r}")
        expected_state = {
            "offline": "trained",
            "shadow": "shadow",
            "canary": "canary",
        }[stage]
        if job["state"] != expected_state:
            raise PromotionError(
                f"{stage} evaluation requires job state {expected_state}"
            )
        if job["dataset_version_id"] != dataset_version_id:
            raise PromotionError("evaluation dataset does not match training job")
        if evaluation.frozen_holdout_id != dataset_version_id:
            raise PromotionError("evaluation holdout is not the immutable job dataset")
        if job["checkpoint_sha256"] != evaluation.candidate_sha256:
            raise PromotionError("evaluation candidate does not match job checkpoint")
        current = conn.execute(
            """
            SELECT result_sha256 FROM evaluation_results
            WHERE evaluation_id = ?
            """,
            (evaluation_id,),
        ).fetchone()
        if current:
            if current["result_sha256"] != digest:
                raise PromotionError(
                    f"evaluation {evaluation_id!r} is immutable"
                )
            return digest
        stage_evidence = conn.execute(
            """
            SELECT evaluation_id FROM evaluation_results
            WHERE job_id = ?
              AND json_extract(metrics_json, '$.stage') = ?
            """,
            (job_id, stage),
        ).fetchone()
        if stage_evidence is not None:
            raise PromotionError(
                f"{stage} evaluation for job {job_id!r} is immutable"
            )
        try:
            conn.execute(
                """
                INSERT INTO evaluation_results (
                    evaluation_id, job_id, dataset_version_id,
                    candidate_sha256, metrics_json, decision,
                    result_sha256, gpu_hours, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    job_id,
                    dataset_version_id,
                    evaluation.candidate_sha256,
                    json.dumps(
                        {
                            "evaluation": evaluation.model_dump(mode="json"),
                            "stage": stage,
                            "policy": policy.model_dump(mode="json"),
                            "decision": decision.model_dump(mode="json"),
                            "holdout_evidence": holdout_evidence,
                        },
                        sort_keys=True,
                    ),
                    decision.action,
                    digest,
                    evaluation.gpu_hours,
                    timestamp.isoformat(timespec="seconds"),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise PromotionError(
                f"could not record evaluation {evaluation_id!r}"
            ) from exc
    return digest
