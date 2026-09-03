#!/usr/bin/env python3
"""Evaluate a model-ladder target without weakening any required gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from harness.training.capability import CapabilityEvidence, qualify_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ladder = yaml.safe_load(args.ladder.read_text(encoding="utf-8")) or {}
    evidence = CapabilityEvidence.model_validate_json(
        args.evidence.read_text(encoding="utf-8")
    )
    decision = qualify_target(ladder, evidence)
    print(json.dumps(decision.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if decision.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
