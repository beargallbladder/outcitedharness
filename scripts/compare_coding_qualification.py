#!/usr/bin/env python3
"""Compare base and adapter coding evaluations with a strict promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "harness.coding-repair-evaluation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("coding evaluation must be a regular file")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported coding evaluation")
    expected = value.get("evidence_sha256")
    core = {key: item for key, item in value.items() if key != "evidence_sha256"}
    if expected != _sha256(_canonical(core)):
        raise ValueError("coding evaluation evidence digest mismatch")
    return value


def compare(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    minimum_gain: float,
    maximum_latency_regression: float,
) -> dict[str, Any]:
    if baseline["fixture_sha256"] != candidate["fixture_sha256"]:
        raise ValueError("coding evaluations use different frozen fixtures")
    if baseline["model"] == candidate["model"]:
        raise ValueError("candidate model identity must differ from baseline")
    baseline_cases = {row["case_id"]: row for row in baseline["cases"]}
    candidate_cases = {row["case_id"]: row for row in candidate["cases"]}
    if (
        baseline_cases.keys() != candidate_cases.keys()
        or baseline["sample_count"] != len(baseline_cases)
        or candidate["sample_count"] != len(candidate_cases)
    ):
        raise ValueError("coding evaluation case sets differ")
    regressions = sorted(
        case_id
        for case_id in baseline_cases
        if baseline_cases[case_id]["passed"]
        and not candidate_cases[case_id]["passed"]
    )
    improvements = sorted(
        case_id
        for case_id in baseline_cases
        if not baseline_cases[case_id]["passed"]
        and candidate_cases[case_id]["passed"]
    )
    gain = (
        candidate["verified_success_rate"]
        - baseline["verified_success_rate"]
    )
    latency_limit = baseline["p95_latency_ms"] * (
        1 + maximum_latency_regression
    )
    checks = {
        "minimum_verified_success_gain": gain >= minimum_gain,
        "no_case_regressions": not regressions,
        "latency_within_bound": candidate["p95_latency_ms"] <= latency_limit,
        "complete_case_set": candidate["sample_count"] >= 5,
    }
    core: dict[str, Any] = {
        "schema": "harness.coding-qualification.v1",
        "fixture_sha256": baseline["fixture_sha256"],
        "baseline_model": baseline["model"],
        "candidate_model": candidate["model"],
        "baseline_evidence_sha256": baseline["evidence_sha256"],
        "candidate_evidence_sha256": candidate["evidence_sha256"],
        "minimum_gain": minimum_gain,
        "maximum_latency_regression": maximum_latency_regression,
        "verified_success_gain": gain,
        "regressions": regressions,
        "improvements": improvements,
        "checks": checks,
        "passed": all(checks.values()),
        "action": "shadow" if all(checks.values()) else "reject",
    }
    core["qualification_sha256"] = _sha256(_canonical(core))
    return core


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("coding qualification output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o444)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-gain", type=float, default=0.02)
    parser.add_argument("--maximum-latency-regression", type=float, default=0.10)
    arguments = parser.parse_args()
    if not 0 <= arguments.minimum_gain <= 1:
        raise ValueError("minimum gain must be between zero and one")
    if arguments.maximum_latency_regression < 0:
        raise ValueError("latency regression bound cannot be negative")
    result = compare(
        baseline=load(arguments.baseline),
        candidate=load(arguments.candidate),
        minimum_gain=arguments.minimum_gain,
        maximum_latency_regression=arguments.maximum_latency_regression,
    )
    _write_once(arguments.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
