#!/usr/bin/env python3
"""Derive and record shadow/canary metrics from paired dual-run evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.storage.db import Store
from harness.training.promotion import (
    CandidateEvaluation,
    PromotionError,
    PromotionPolicy,
    decide_promotion,
    record_evaluation,
)


class RunOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verified_success: bool
    first_pass: bool
    frontier_escalated: bool
    latency_ms: float = Field(ge=0)
    cost: float = Field(ge=0)
    repair_cycles: int = Field(ge=0)
    critical_regression: bool
    gpu_seconds: float = Field(default=0, ge=0)
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def first_pass_requires_success(self) -> RunOutcome:
        if self.first_pass and not self.verified_success:
            raise ValueError("first-pass outcome must be a verified success")
        return self


class DualRun(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    schema_name: Literal["harness.dual-run.v1"] = Field(alias="schema")
    request_id: str = Field(pattern=r"^[A-Za-z0-9._:-]+$")
    task_class: str = Field(pattern=r"^[A-Za-z0-9._:-]+$")
    route: str = Field(pattern=r"^[A-Za-z0-9._:/-]+$")
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate: RunOutcome
    baseline: RunOutcome


RunOutcome.model_rebuild()
DualRun.model_rebuild(
    _types_namespace={"Literal": Literal, "RunOutcome": RunOutcome}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dual_runs(path: Path) -> tuple[list[DualRun], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("dual-run log must be a regular non-symlink file")
    rows: list[DualRun] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(f"dual-run log has an empty line at {line_number}")
        try:
            value = json.loads(line)
            rows.append(DualRun.model_validate(value))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"invalid dual-run evidence at line {line_number}"
            ) from exc
    if not rows:
        raise ValueError("dual-run log is empty")
    request_ids = [row.request_id for row in rows]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("dual-run request IDs must be unique")
    return rows, _sha256(path)


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _cost_per_success(outcomes: list[RunOutcome]) -> float:
    successes = sum(outcome.verified_success for outcome in outcomes)
    if successes == 0:
        return 1_000_000_000_000.0
    return sum(outcome.cost for outcome in outcomes) / successes


def derive_evaluation(
    rows: list[DualRun],
    *,
    previous: CandidateEvaluation,
    log_sha256: str,
) -> CandidateEvaluation:
    if any(row.task_class != previous.task_class for row in rows):
        raise ValueError("dual-run task class differs from offline evidence")
    if any(row.candidate_sha256 != previous.candidate_sha256 for row in rows):
        raise ValueError("dual-run candidate differs from the trained checkpoint")
    baseline_ids = {row.baseline_sha256 for row in rows}
    routes = {row.route for row in rows}
    if len(baseline_ids) != 1 or len(routes) != 1:
        raise ValueError("dual-run baseline and route must be constant")
    candidate = [row.candidate for row in rows]
    baseline = [row.baseline for row in rows]
    return CandidateEvaluation(
        task_class=previous.task_class,
        frozen_holdout_id=previous.frozen_holdout_id,
        holdout_disjoint=previous.holdout_disjoint,
        sample_count=len(rows),
        verified_success_rate=_rate(
            [outcome.verified_success for outcome in candidate]
        ),
        baseline_verified_success_rate=_rate(
            [outcome.verified_success for outcome in baseline]
        ),
        frontier_escalation_rate=_rate(
            [outcome.frontier_escalated for outcome in candidate]
        ),
        baseline_frontier_escalation_rate=_rate(
            [outcome.frontier_escalated for outcome in baseline]
        ),
        cost_per_verified_success=_cost_per_success(candidate),
        baseline_cost_per_verified_success=_cost_per_success(baseline),
        p95_latency_ms=_p95([outcome.latency_ms for outcome in candidate]),
        baseline_p95_latency_ms=_p95(
            [outcome.latency_ms for outcome in baseline]
        ),
        first_pass_success_rate=_rate(
            [outcome.first_pass for outcome in candidate]
        ),
        mean_repair_cycles=(
            sum(outcome.repair_cycles for outcome in candidate) / len(candidate)
        ),
        critical_regressions=sum(
            outcome.critical_regression for outcome in candidate
        ),
        checkpoint_reproducible=previous.checkpoint_reproducible,
        resume_verified=previous.resume_verified,
        candidate_sha256=previous.candidate_sha256,
        gpu_hours=sum(outcome.gpu_seconds for outcome in candidate) / 3600,
        metadata={
            "dual_run_derived": True,
            "dual_run_schema": "harness.dual-run.v1",
            "dual_run_log_sha256": log_sha256,
            "baseline_sha256": next(iter(baseline_ids)),
            "route": next(iter(routes)),
            "paired_requests": len(rows),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--dual-run-log", required=True, type=Path)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--dataset-version-id", required=True)
    parser.add_argument("--stage", required=True, choices=("shadow", "canary"))
    parser.add_argument("--policy", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = Store(args.database)
    previous_stage = "offline" if args.stage == "shadow" else "shadow"
    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT evaluation_id, metrics_json FROM evaluation_results
            WHERE job_id = ?
              AND json_extract(metrics_json, '$.stage') = ?
            """,
            (args.job_id, previous_stage),
        ).fetchone()
    if row is None:
        raise PromotionError(
            f"{args.stage} collection requires {previous_stage} evidence"
        )
    previous_payload = json.loads(row["metrics_json"])
    previous = CandidateEvaluation.model_validate(
        previous_payload["evaluation"]
    )
    rows, log_sha256 = load_dual_runs(args.dual_run_log)
    evaluation = derive_evaluation(
        rows,
        previous=previous,
        log_sha256=log_sha256,
    )
    policy = PromotionPolicy()
    if args.policy is not None:
        if args.policy.is_symlink() or not args.policy.is_file():
            raise ValueError("policy must be a regular non-symlink file")
        policy = PromotionPolicy.model_validate_json(
            args.policy.read_text(encoding="utf-8")
        )
    decision = decide_promotion(evaluation, stage=args.stage, policy=policy)
    result_sha256 = record_evaluation(
        store,
        evaluation_id=args.evaluation_id,
        job_id=args.job_id,
        dataset_version_id=args.dataset_version_id,
        evaluation=evaluation,
        stage=args.stage,
        policy=policy,
    )
    print(
        json.dumps(
            {
                "evaluation_id": args.evaluation_id,
                "previous_evaluation_id": row["evaluation_id"],
                "result_sha256": result_sha256,
                "stage": args.stage,
                "sample_count": evaluation.sample_count,
                "decision": decision.action,
            },
            sort_keys=True,
        )
    )
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
