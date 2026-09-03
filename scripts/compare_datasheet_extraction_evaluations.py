#!/usr/bin/env python3
"""Seal a production-safety decision for base-versus-candidate extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.electronics.claims import canonical_json


SCHEMA = "harness.electronics-extraction-candidate-decision.v1"
EVALUATION_SCHEMA = "harness.electronics-extraction-evaluation.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, kind: str) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{kind} must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != EVALUATION_SCHEMA:
        raise ValueError(f"{kind} has an unsupported schema")
    core = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "evidence_sha256"}
    }
    core["evidence_sha256"] = value.get("evidence_sha256")
    supplied = core.pop("evidence_sha256")
    if hashlib.sha256(canonical_json(core)).hexdigest() != supplied:
        raise ValueError(f"{kind} has an invalid evidence digest")
    return value


def _metric(value: dict[str, Any], name: str) -> float | None:
    raw = value["aggregate"].get(name)
    return None if raw is None else float(raw)


def _lane(
    *,
    base_path: Path,
    candidate_path: Path,
    metric_names: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base = _load(base_path, "base evaluation")
    candidate = _load(candidate_path, "candidate evaluation")
    for key in ("evidence_sha256", "work_queue_sha256", "labels_sha256"):
        if base["cohort"][key] != candidate["cohort"][key]:
            raise ValueError(f"base and candidate cohort {key} differ")
    base_metrics = {name: _metric(base, name) for name in metric_names}
    candidate_metrics = {
        name: _metric(candidate, name) for name in metric_names
    }
    deltas = {
        name: (
            None
            if base_metrics[name] is None or candidate_metrics[name] is None
            else candidate_metrics[name] - base_metrics[name]
        )
        for name in metric_names
    }
    receipt = {
        "cohort": base["cohort"],
        "base": {
            "path": str(base_path.expanduser().resolve(strict=True)),
            "sha256": _sha256(base_path),
            "model": base["model"],
            "metrics": base_metrics,
        },
        "candidate": {
            "path": str(candidate_path.expanduser().resolve(strict=True)),
            "sha256": _sha256(candidate_path),
            "model": candidate["model"],
            "metrics": candidate_metrics,
        },
        "deltas": deltas,
    }
    return receipt, base_metrics, candidate_metrics


def compare(
    *,
    pin_base_path: Path,
    pin_candidate_path: Path,
    parametric_base_path: Path,
    parametric_candidate_path: Path,
) -> dict[str, Any]:
    pin_names = (
        "json_valid_rate",
        "identity_exact_rate",
        "pin_identity_f1",
        "type_accuracy",
        "direction_accuracy",
            "supply_domain_accuracy",
        "function_f1",
        "evidence_grounding_rate",
        "hallucination_rate",
        "paid_call_replacement_rate",
    )
    parametric_names = (
        "json_valid_rate",
        "parametric_accuracy",
        "parametric_exact_rate",
        "evidence_grounding_rate",
        "hallucination_rate",
        "paid_call_replacement_rate",
    )
    pin, pin_base, pin_candidate = _lane(
        base_path=pin_base_path,
        candidate_path=pin_candidate_path,
        metric_names=pin_names,
    )
    parametric, parametric_base, parametric_candidate = _lane(
        base_path=parametric_base_path,
        candidate_path=parametric_candidate_path,
        metric_names=parametric_names,
    )

    reasons: list[str] = []
    pin_non_regression = (
        "json_valid_rate",
        "identity_exact_rate",
        "pin_identity_f1",
        "type_accuracy",
        "direction_accuracy",
            "supply_domain_accuracy",
        "function_f1",
        "evidence_grounding_rate",
        "paid_call_replacement_rate",
    )
    for name in pin_non_regression:
        before = pin_base[name]
        after = pin_candidate[name]
        if before is not None and after is not None and after + 1e-12 < before:
            reasons.append(f"pin {name} regressed")
    if (
        pin_candidate["hallucination_rate"] is not None
        and pin_base["hallucination_rate"] is not None
        and pin_candidate["hallucination_rate"]
        > pin_base["hallucination_rate"] + 1e-12
    ):
        reasons.append("pin hallucination_rate regressed")
    for name in (
        "json_valid_rate",
        "parametric_accuracy",
        "parametric_exact_rate",
        "evidence_grounding_rate",
        "paid_call_replacement_rate",
    ):
        before = parametric_base[name]
        after = parametric_candidate[name]
        if before is not None and after is not None and after + 1e-12 < before:
            reasons.append(f"parametric {name} regressed")
    if (
        parametric_candidate["hallucination_rate"] is not None
        and parametric_base["hallucination_rate"] is not None
        and parametric_candidate["hallucination_rate"]
        > parametric_base["hallucination_rate"] + 1e-12
    ):
        reasons.append("parametric hallucination_rate regressed")

    passed = not reasons
    core = {
        "schema": SCHEMA,
        "passed": passed,
        "decision": "retain" if passed else "reject",
        "promotion_authorized": False,
        "pin": pin,
        "parametric": parametric,
        "reasons": reasons,
        "policy": {
            "all_scored_production_metrics_must_be_non_regressive": True,
            "promotion_requires_separate_authorization": True,
        },
    }
    core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
    return {"created_at": datetime.now(timezone.utc).isoformat(), **core}


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
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ).encode()
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin-base", required=True, type=Path)
    parser.add_argument("--pin-candidate", required=True, type=Path)
    parser.add_argument("--parametric-base", required=True, type=Path)
    parser.add_argument("--parametric-candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    decision = compare(
        pin_base_path=args.pin_base,
        pin_candidate_path=args.pin_candidate,
        parametric_base_path=args.parametric_base,
        parametric_candidate_path=args.parametric_candidate,
    )
    _write_new(args.output, decision)
    print(json.dumps(decision, sort_keys=True))
    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
