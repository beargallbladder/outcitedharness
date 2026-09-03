#!/usr/bin/env python3
"""Compare base and adapted datasheet-vision results on a frozen fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "harness.datasheet-vision-qualification.v1"
RESULT_SCHEMA = "harness.datasheet-modality-evaluation.v4"


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_result(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("evaluation result must be a regular file")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != RESULT_SCHEMA:
        raise ValueError("unsupported datasheet evaluation result")
    identity = value.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("evaluation identity is missing")
    core = {key: item for key, item in value.items() if key != "identity"}
    if identity.get("core_sha256") != hashlib.sha256(_canonical(core)).hexdigest():
        raise ValueError("evaluation identity digest mismatch")
    if value.get("configuration", {}).get("modes") != ["image_rows"]:
        raise ValueError("qualification requires image_rows-only evaluations")
    return value


def _case_scores(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scores = {}
    for case in value.get("cases", []):
        case_id = case.get("id") if isinstance(case, dict) else None
        mode = case.get("modalities", {}).get("image_rows", {})
        score = mode.get("score") if isinstance(mode, dict) else None
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in scores
            or not isinstance(score, dict)
            or not isinstance(score.get("pair_f1"), (int, float))
        ):
            raise ValueError("evaluation contains an invalid image_rows case")
        scores[case_id] = {
            "pair_f1": float(score["pair_f1"]),
            "contract_valid": mode.get("contract_valid") is True,
            "physical_identity_error": mode.get("physical_identity_error"),
        }
    if not scores:
        raise ValueError("evaluation has no image_rows cases")
    return scores


def compare(
    *,
    baseline_path: Path,
    candidate_path: Path,
    minimum_mean_delta: float,
    maximum_case_regression: float,
) -> dict[str, Any]:
    baseline = _load_result(baseline_path)
    candidate = _load_result(candidate_path)
    if (
        baseline["fixture_sha256"] != candidate["fixture_sha256"]
        or baseline["identity"]["source_gold_set_sha256"]
        != candidate["identity"]["source_gold_set_sha256"]
    ):
        raise ValueError("baseline and candidate fixture identities differ")
    baseline_scores = _case_scores(baseline)
    candidate_scores = _case_scores(candidate)
    if baseline_scores.keys() != candidate_scores.keys():
        raise ValueError("baseline and candidate case sets differ")

    cases = []
    for case_id in sorted(baseline_scores):
        baseline_score = baseline_scores[case_id]
        candidate_score = candidate_scores[case_id]
        cases.append(
            {
                "id": case_id,
                "baseline_pair_f1": baseline_score["pair_f1"],
                "candidate_pair_f1": candidate_score["pair_f1"],
                "pair_f1_delta": (
                    candidate_score["pair_f1"] - baseline_score["pair_f1"]
                ),
                "baseline_contract_valid": baseline_score["contract_valid"],
                "candidate_contract_valid": candidate_score["contract_valid"],
                "candidate_physical_identity_error": candidate_score[
                    "physical_identity_error"
                ],
            }
        )
    mean_delta = sum(row["pair_f1_delta"] for row in cases) / len(cases)
    worst_delta = min(row["pair_f1_delta"] for row in cases)
    candidate_contract_valid = sum(
        row["candidate_contract_valid"] for row in cases
    )
    baseline_contract_valid = sum(
        row["baseline_contract_valid"] for row in cases
    )
    passed = (
        mean_delta >= minimum_mean_delta
        and worst_delta >= -maximum_case_regression
        and candidate_contract_valid == len(cases)
        and baseline_contract_valid == len(cases)
    )
    core: dict[str, Any] = {
        "schema": SCHEMA,
        "baseline_result_sha256": _sha256(baseline_path),
        "candidate_result_sha256": _sha256(candidate_path),
        "comparator_sha256": _sha256(Path(__file__).resolve()),
        "fixture_sha256": baseline["fixture_sha256"],
        "baseline_model": baseline["model"],
        "candidate_model": candidate["model"],
        "thresholds": {
            "minimum_mean_pair_f1_delta": minimum_mean_delta,
            "maximum_case_pair_f1_regression": maximum_case_regression,
            "all_contracts_valid": True,
        },
        "metrics": {
            "cases": len(cases),
            "baseline_mean_pair_f1": (
                sum(row["baseline_pair_f1"] for row in cases) / len(cases)
            ),
            "candidate_mean_pair_f1": (
                sum(row["candidate_pair_f1"] for row in cases) / len(cases)
            ),
            "mean_pair_f1_delta": mean_delta,
            "worst_case_pair_f1_delta": worst_delta,
            "baseline_contract_valid_cases": baseline_contract_valid,
            "candidate_contract_valid_cases": candidate_contract_valid,
        },
        "cases": cases,
        "passed": passed,
    }
    core["qualification_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    return core


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError(f"output already exists: {path}")
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
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-mean-delta", type=float, default=0.01)
    parser.add_argument("--maximum-case-regression", type=float, default=0.02)
    args = parser.parse_args()
    if args.maximum_case_regression < 0:
        parser.error("maximum case regression must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    result = compare(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        minimum_mean_delta=args.minimum_mean_delta,
        maximum_case_regression=args.maximum_case_regression,
    )
    _write_new(args.output, result)
    print(json.dumps(result["metrics"], sort_keys=True))
    print("PASS" if result["passed"] else "REJECT")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
