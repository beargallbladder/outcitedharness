#!/usr/bin/env python3
"""Record a failed DesignWins generation canary as an offline rejection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from harness.storage.db import Store
from harness.training.promotion import (
    CandidateEvaluation,
    PromotionPolicy,
    decide_promotion,
    record_evaluation,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--canary-qualification", required=True, type=Path)
    parser.add_argument("--training-qualification", required=True, type=Path)
    parser.add_argument("--resume-summary", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--dataset-version-id", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--evaluation-id", required=True)
    return parser.parse_args()


def record_rejection(args: argparse.Namespace):
    canary = json.loads(args.canary_qualification.read_text(encoding="utf-8"))
    training = json.loads(args.training_qualification.read_text(encoding="utf-8"))
    resume = json.loads(args.resume_summary.read_text(encoding="utf-8"))
    if canary.get("passed") is not False:
        raise ValueError("this recorder only accepts a failed generation canary")
    if canary.get("schema") != "harness.designwins.rejection-canary.v1":
        raise ValueError("unexpected generation-canary schema")
    if training.get("schema") != (
        "harness.designwins-teacher-forced-qualification.v1"
    ):
        raise ValueError("unexpected training-signal schema")
    metrics = canary["metrics"]
    failed_checks = sum(not bool(value) for value in canary["checks"].values())
    evaluation = CandidateEvaluation(
        task_class="electronics_pinout_extraction",
        frozen_holdout_id=args.dataset_version_id,
        holdout_disjoint=True,
        sample_count=8,
        verified_success_rate=metrics["candidate_leaf_f1"],
        baseline_verified_success_rate=metrics["baseline_leaf_f1"],
        frontier_escalation_rate=0,
        baseline_frontier_escalation_rate=0,
        cost_per_verified_success=0,
        baseline_cost_per_verified_success=0,
        p95_latency_ms=0,
        baseline_p95_latency_ms=0,
        first_pass_success_rate=metrics["candidate_exact_rate"],
        mean_repair_cycles=0,
        critical_regressions=failed_checks,
        checkpoint_reproducible=bool(
            canary["checks"]["deterministic_reproduction"]
        ),
        resume_verified=bool(resume.get("passed")),
        candidate_sha256=args.candidate_sha256,
        gpu_hours=0,
        pinout_leaf_f1=metrics["candidate_leaf_f1"],
        pinout_exact_rate=metrics["candidate_exact_rate"],
        metadata={
            "canary_qualification_sha256": _sha256(args.canary_qualification),
            "training_qualification_sha256": _sha256(
                args.training_qualification
            ),
            "training_signal_passed": bool(training.get("passed")),
            "gpu_hours_known": False,
        },
    )
    policy = PromotionPolicy(
        minimum_verified_success_gain=metrics["minimum_required_gain"],
        maximum_latency_regression=0,
        minimum_offline_samples=141,
        minimum_shadow_samples=50,
        minimum_canary_samples=100,
        require_lower_cost_per_success=False,
    )
    decision = decide_promotion(evaluation, stage="offline", policy=policy)
    if decision.action != "reject":
        raise ValueError("failed canary unexpectedly produced a promotion decision")
    record_evaluation(
        Store(args.database),
        evaluation_id=args.evaluation_id,
        job_id=args.job_id,
        dataset_version_id=args.dataset_version_id,
        evaluation=evaluation,
        stage="offline",
        policy=policy,
    )
    return decision


def main() -> int:
    decision = record_rejection(parse_args())
    print(json.dumps(decision.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
