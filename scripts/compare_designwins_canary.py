#!/usr/bin/env python3
"""Apply the strict DesignWins gate to an eight-record rejection canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_designwins_qualification import _load, _write_json, compare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-repeat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-f1-gain", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare(
        _load(args.baseline),
        _load(args.candidate),
        _load(args.candidate_repeat),
        minimum_f1_gain=args.minimum_f1_gain,
        maximum_family_regression=0.02,
        minimum_family_samples=5,
        expected_samples=8,
    )
    result["schema"] = "harness.designwins.rejection-canary.v1"
    result["scope"] = "rejection-only"
    result["production_promotion_eligible"] = False
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
