#!/usr/bin/env python3
"""Apply promotion policy and immutably record a candidate evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.storage.db import Store
from harness.training.promotion import (
    CandidateEvaluation,
    PromotionPolicy,
    decide_promotion,
    record_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    policy = PromotionPolicy.model_validate(payload["policy"])
    evaluation = CandidateEvaluation.model_validate(payload["evaluation"])
    if evaluation.task_class == "electronics_pinout_extraction":
        raise ValueError(
            "electronics evaluations require the sealed DesignWins recorder"
        )
    stage = payload["stage"]
    if stage != "offline":
        raise ValueError(
            "shadow and canary evaluations require the dual-run collector"
        )
    decision = decide_promotion(evaluation, stage=stage, policy=policy)
    record_evaluation(
        Store(args.database),
        evaluation_id=payload["evaluation_id"],
        job_id=payload["job_id"],
        dataset_version_id=payload["dataset_version_id"],
        evaluation=evaluation,
        stage=stage,
        policy=policy,
    )
    print(json.dumps(decision.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
