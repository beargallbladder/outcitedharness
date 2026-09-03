#!/usr/bin/env python3
"""Capture one allowlisted read-only API response into the learning ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.storage.db import Store
from harness.training.connectors import ReadOnlyConnector, load_connector_specs
from harness.training.ledger import LearningLedger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--connectors", required=True, type=Path)
    parser.add_argument("--connector", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--params-json", default="{}")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = {spec.name: spec for spec in load_connector_specs(args.connectors)}
    if args.connector not in specs:
        raise KeyError(f"unknown connector {args.connector!r}")
    params = json.loads(args.params_json)
    if not isinstance(params, dict):
        raise ValueError("--params-json must be an object")
    ledger = LearningLedger(Store(args.database), args.artifact_root)
    with ReadOnlyConnector(specs[args.connector]) as connector:
        capture = connector.capture_json(ledger, args.path, params=params)
    print(
        json.dumps(
            {
                "event_id": capture.event_id,
                "event_sha256": capture.event_sha256,
                "duplicate": capture.duplicate,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
