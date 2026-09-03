#!/usr/bin/env python3
"""Compare frozen base and adapter evaluations with a precommitted retain gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "harness.pinout-vision-candidate-decision.v1"
EVALUATION_SCHEMA = "harness.pinout-vision-evaluation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, kind: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{kind} must be a regular file")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != EVALUATION_SCHEMA:
        raise ValueError(f"{kind} schema is not supported")
    expected = value.get("evidence_sha256")
    core = {
        key: item
        for key, item in value.items()
        if key not in ("created_at", "evidence_sha256")
    }
    if hashlib.sha256(_canonical(core)).hexdigest() != expected:
        raise ValueError(f"{kind} evidence digest is invalid")
    if value.get("cohort", {}).get("limited") is not False:
        raise ValueError(f"{kind} is not a complete frozen-cohort evaluation")
    return value


def _rates(value: dict[str, Any]) -> dict[str, float]:
    aggregate = value["aggregate"]
    examples = int(aggregate["examples"])
    if examples < 1:
        raise ValueError("evaluation contains no examples")
    rich = aggregate["rich"]
    return {
        "json_valid_rate": int(aggregate["json_valid"]) / examples,
        "identity_exact_rate": int(aggregate["identity_exact"]) / examples,
        "identity_f1": float(aggregate["identity"]["f1"]),
        "type_accuracy": float(rich["type_accuracy"]),
        "direction_accuracy": float(rich["direction_accuracy"]),
        "functions_exact_rate": float(rich["functions_exact_rate"]),
    }


def _utility(rates: dict[str, float]) -> float:
    return (
        0.10 * rates["json_valid_rate"]
        + 0.50 * rates["identity_f1"]
        + 0.15 * rates["type_accuracy"]
        + 0.10 * rates["direction_accuracy"]
        + 0.15 * rates["functions_exact_rate"]
    )


def compare(
    *,
    base_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    base = _load(base_path, "base evaluation")
    candidate = _load(candidate_path, "candidate evaluation")
    if (
        base["cohort"]["sha256"] != candidate["cohort"]["sha256"]
        or base["cohort"]["evidence_sha256"]
        != candidate["cohort"]["evidence_sha256"]
    ):
        raise ValueError("evaluations do not use the same frozen cohort")
    if base["model"]["config_sha256"] != candidate["model"]["config_sha256"]:
        raise ValueError("evaluations do not use the same base model")
    if base["model"].get("adapter") is not None:
        raise ValueError("base evaluation unexpectedly uses an adapter")
    if not isinstance(candidate["model"].get("adapter"), dict):
        raise ValueError("candidate evaluation has no adapter")
    if base["generation"] != candidate["generation"]:
        raise ValueError("evaluations use different generation settings")

    base_rates = _rates(base)
    candidate_rates = _rates(candidate)
    base_utility = _utility(base_rates)
    candidate_utility = _utility(candidate_rates)
    reasons: list[str] = []
    if candidate_rates["json_valid_rate"] + 0.02 < base_rates["json_valid_rate"]:
        reasons.append("JSON validity regressed by more than 0.02")
    if candidate_rates["identity_f1"] + 0.03 < base_rates["identity_f1"]:
        reasons.append("pin identity F1 regressed by more than 0.03")
    if candidate_rates["identity_f1"] < 0.80:
        reasons.append("candidate pin identity F1 is below 0.80")
    if candidate_utility - base_utility < 0.03:
        reasons.append("weighted utility improvement is below 0.03")
    if not (
        candidate_rates["type_accuracy"] - base_rates["type_accuracy"] >= 0.10
        or candidate_rates["direction_accuracy"]
        - base_rates["direction_accuracy"]
        >= 0.10
        or candidate_rates["functions_exact_rate"]
        - base_rates["functions_exact_rate"]
        >= 0.05
    ):
        reasons.append("no rich pin field improved by the required margin")

    vendor_regressions: dict[str, dict[str, float]] = {}
    shared_vendors = set(base["by_vendor"]) & set(candidate["by_vendor"])
    for vendor in sorted(shared_vendors):
        base_vendor = base["by_vendor"][vendor]
        candidate_vendor = candidate["by_vendor"][vendor]
        if min(
            int(base_vendor["examples"]),
            int(candidate_vendor["examples"]),
        ) < 3:
            continue
        base_f1 = float(base_vendor["identity"]["f1"])
        candidate_f1 = float(candidate_vendor["identity"]["f1"])
        if candidate_f1 + 0.05 < base_f1:
            vendor_regressions[vendor] = {
                "base_identity_f1": base_f1,
                "candidate_identity_f1": candidate_f1,
            }
    if vendor_regressions:
        reasons.append("one or more sufficiently represented vendors regressed")

    core: dict[str, Any] = {
        "schema": SCHEMA,
        "passed": not reasons,
        "decision": "retain_for_further_qualification" if not reasons else "reject",
        "promotion_authorized": False,
        "base": {
            "path": str(base_path.resolve(strict=True)),
            "sha256": _sha256(base_path),
            "rates": base_rates,
            "weighted_utility": base_utility,
        },
        "candidate": {
            "path": str(candidate_path.resolve(strict=True)),
            "sha256": _sha256(candidate_path),
            "adapter": candidate["model"]["adapter"],
            "rates": candidate_rates,
            "weighted_utility": candidate_utility,
        },
        "deltas": {
            key: candidate_rates[key] - base_rates[key]
            for key in base_rates
        }
        | {"weighted_utility": candidate_utility - base_utility},
        "vendor_regressions": vendor_regressions,
        "reasons": reasons,
    }
    core["evidence_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **core,
    }


def write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    decision = compare(
        base_path=arguments.base,
        candidate_path=arguments.candidate,
    )
    write_new(arguments.output, decision)
    print(json.dumps(decision, sort_keys=True))
    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
