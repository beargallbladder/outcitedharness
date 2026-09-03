#!/usr/bin/env python3
"""Apply precommitted capability, cost, and CR-import retain gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from harness.electronics.admission import CR_IMPORT_SCHEMA
from harness.electronics.claims import canonical_json
from harness.electronics.qualification import FactoryThresholds, qualify


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validated_cr(admission_root: Path) -> dict[str, Any]:
    root = admission_root.expanduser().resolve(strict=True)
    manifest = _json(root / "manifest.json")
    receipt = manifest.get("artifacts", {}).get("cr-pin-packages.jsonl", {})
    path = root / "cr-pin-packages.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get("sha256"):
        raise ValueError("CR import artifact hash mismatch")
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if value.get("schema") != CR_IMPORT_SCHEMA:
                raise ValueError(
                    f"invalid CR record schema at line {line_number}"
                )
            pins = value.get("pins")
            expected = value.get("expected_package_pins")
            if (
                not isinstance(pins, list)
                or not isinstance(expected, int)
                or len(pins) != expected
            ):
                raise ValueError(
                    f"incomplete CR package at line {line_number}"
                )
            count += 1
    return {
        "cr_package_records": manifest["counts"]["cr_package_records"],
        "validated_cr_package_records": count,
        "direct_database_write": manifest["policy"]["direct_cr_write"],
    }


def _write_new(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"immutable output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reproducibility", type=Path, required=True)
    parser.add_argument("--frontier-reconciliation", type=Path, required=True)
    parser.add_argument("--frontier-finalization", type=Path, required=True)
    parser.add_argument("--admission-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = yaml.safe_load(args.policy.expanduser().resolve(strict=True).read_text())
    if policy.get("schema") != (
        "harness.electronics-factory-qualification-policy.v1"
    ):
        raise ValueError("qualification policy schema is not supported")
    reconciliation = _json(args.frontier_reconciliation)
    finalization = _json(args.frontier_finalization)
    frontier = {
        "actual_batch_cost_usd": reconciliation["usage"][
            "actual_batch_cost_usd"
        ],
        "admitted_training_pairs": finalization["counts"]["training_pairs"],
    }
    report = qualify(
        baseline=_json(args.baseline),
        candidate=_json(args.candidate),
        reproducibility=_json(args.reproducibility),
        frontier=frontier,
        admissions=_validated_cr(args.admission_bundle),
        thresholds=FactoryThresholds.model_validate(policy["thresholds"]),
    )
    report["sources"] = {
        "policy": str(args.policy.resolve()),
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "reproducibility": str(args.reproducibility.resolve()),
        "frontier_reconciliation": str(args.frontier_reconciliation.resolve()),
        "frontier_finalization": str(args.frontier_finalization.resolve()),
        "admission_bundle": str(args.admission_bundle.resolve()),
    }
    report.pop("evidence_sha256", None)
    report["evidence_sha256"] = hashlib.sha256(
        canonical_json(report)
    ).hexdigest()
    _write_new(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["promotion_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
