#!/usr/bin/env python3
"""Record the strict DesignWins qualification as an offline promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

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


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_regular(path: Path, *, kind: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{kind} must be a regular non-symlink file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{kind} must contain a JSON object")
    return value


def _validate_qualification(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != "harness.designwins.qualification.v1":
        raise ValueError("qualification schema is invalid")
    identity = value.get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("schema")
        != "harness.designwins.qualification-identity.v1"
    ):
        raise ValueError("qualification is not bound to sealed evaluations")
    core = dict(value)
    core.pop("identity")
    if identity.get("core_sha256") != hashlib.sha256(_canonical(core)).hexdigest():
        raise ValueError("qualification core hash does not match its identity")
    for key in (
        "baseline_evaluation_sha256",
        "candidate_evaluation_sha256",
        "candidate_repeat_evaluation_sha256",
        "comparator_sha256",
    ):
        digest = identity.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"qualification identity has invalid {key}")
    return identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--qualification", required=True, type=Path)
    parser.add_argument("--resume-summary", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--dataset-version-id", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--evaluation-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qualification = _load_regular(args.qualification, kind="qualification")
    qualification_identity = _validate_qualification(qualification)
    resume = _load_regular(args.resume_summary, kind="resume summary")
    metrics = qualification["metrics"]
    reproduction = qualification["reproduction"]
    evaluation = CandidateEvaluation(
        task_class="electronics_pinout_extraction",
        frozen_holdout_id=args.dataset_version_id,
        holdout_disjoint=True,
        sample_count=141,
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
        critical_regressions=(
            len(qualification["regressed_families"])
            + sum(
                1
                for passed in qualification["checks"].values()
                if not passed
            )
        ),
        checkpoint_reproducible=(
            reproduction["candidate_fingerprint"]
            == reproduction["repeat_fingerprint"]
        ),
        resume_verified=bool(resume.get("passed")),
        candidate_sha256=args.candidate_sha256,
        gpu_hours=0,
        pinout_leaf_f1=metrics["candidate_leaf_f1"],
        pinout_exact_rate=metrics["candidate_exact_rate"],
        metadata={
            "qualification_sha256": _sha256(args.qualification),
            "qualification_core_sha256": qualification_identity["core_sha256"],
            "gpu_hours_known": False,
            "strict_qualification_passed": bool(qualification["passed"]),
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
    if not qualification["passed"] and decision.passed:
        raise ValueError(
            "generic promotion gate cannot override strict DesignWins rejection"
        )
    record_evaluation(
        Store(args.database),
        evaluation_id=args.evaluation_id,
        job_id=args.job_id,
        dataset_version_id=args.dataset_version_id,
        evaluation=evaluation,
        stage="offline",
        policy=policy,
    )
    print(json.dumps(decision.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
