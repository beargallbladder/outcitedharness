#!/usr/bin/env python3
"""Append one verified local-versus-paid replacement observation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from harness.economics import ReplacementObservation, record_replacement_observation
from harness.storage.db import Store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    return parser.parse_args()


def load_observation(path: Path) -> ReplacementObservation:
    if path.is_symlink() or not path.is_file():
        raise ValueError("observation input must be a regular non-symlink file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("observation input must be a JSON object")
    timestamp = value.get("created_at")
    if not isinstance(timestamp, str):
        raise ValueError("observation created_at must be an ISO-8601 string")
    try:
        created_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("observation created_at is not valid ISO-8601") from exc
    payload = {**value, "created_at": created_at}
    for name in (
        "verified_success",
        "first_pass",
        "critical_regression",
        "frontier_escalated",
    ):
        if not isinstance(payload.get(name), bool):
            raise ValueError(f"observation {name} must be boolean")
    if not isinstance(payload.get("repair_cycles"), int) or isinstance(
        payload["repair_cycles"], bool
    ):
        raise ValueError("observation repair_cycles must be an integer")
    try:
        return ReplacementObservation(**payload)
    except TypeError as exc:
        raise ValueError("observation fields do not match the contract") from exc


def main() -> int:
    args = parse_args()
    observation = load_observation(args.input)
    record_replacement_observation(Store(args.database), observation)
    print(observation.observation_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
