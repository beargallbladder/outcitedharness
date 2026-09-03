#!/usr/bin/env python3
"""Apply the immutable DesignWins base-versus-adapter promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("summary"), dict):
        raise ValueError(f"{path} is not a DesignWins evaluation")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_identity(
    result: dict[str, Any],
    *,
    baseline: Path,
    candidate: Path,
    candidate_repeat: Path,
) -> dict[str, Any]:
    if result.get("identity") is not None:
        raise ValueError("qualification is already identity-bound")
    bound = dict(result)
    bound["identity"] = {
        "schema": "harness.designwins.qualification-identity.v1",
        "core_sha256": hashlib.sha256(_canonical(result)).hexdigest(),
        "baseline_evaluation_sha256": _sha256(baseline),
        "candidate_evaluation_sha256": _sha256(candidate),
        "candidate_repeat_evaluation_sha256": _sha256(candidate_repeat),
        "comparator_sha256": _sha256(Path(__file__).resolve()),
    }
    return bound


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "valid_json_rate": sum(bool(row["score"]["valid_json"]) for row in rows)
        / len(rows),
        "generation_limit_hits": sum(
            bool(row["hit_generation_limit"]) for row in rows
        ),
        "exact_rate": sum(bool(row["score"]["exact"]) for row in rows)
        / len(rows),
        **{
            f"mean_{name}": sum(float(row["score"][name]) for row in rows)
            / len(rows)
            for name in ("leaf_precision", "leaf_recall", "leaf_f1")
        },
    }


def _summary(details: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = {}
    for row in details:
        families.setdefault(str(row["family"]), []).append(row)
    return {
        **_aggregate(details),
        "by_family": {
            family: _aggregate(rows) for family, rows in sorted(families.items())
        },
    }


def _validate(
    evaluation: dict[str, Any],
    *,
    expected_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    details = evaluation.get("details")
    identity = evaluation.get("identity")
    if not isinstance(details, list) or len(details) != expected_samples:
        raise ValueError(
            f"evaluation must contain exactly {expected_samples} detail rows"
        )
    expected_indices = list(range(expected_samples))
    indices = [row.get("index") for row in details]
    if indices != expected_indices:
        raise ValueError("evaluation indices must be unique, ordered, and complete")
    record_ids = [(row.get("part"), row.get("family")) for row in details]
    if any(not part or not family for part, family in record_ids):
        raise ValueError("every evaluation row requires part and family identity")
    if len(set(record_ids)) != expected_samples:
        raise ValueError("evaluation record identities must be unique")
    for row in details:
        score = row.get("score")
        if not isinstance(score, dict) or any(
            key not in score
            for key in (
                "valid_json",
                "exact",
                "leaf_precision",
                "leaf_recall",
                "leaf_f1",
            )
        ):
            raise ValueError("evaluation row has incomplete score")
        for key in ("leaf_precision", "leaf_recall", "leaf_f1"):
            value = float(score[key])
            if not 0 <= value <= 1:
                raise ValueError(f"score {key} is outside [0, 1]")
    recomputed = _summary(details)
    if evaluation.get("summary") != recomputed:
        raise ValueError("evaluation summary does not match detail rows")
    if (
        not isinstance(identity, dict)
        or identity.get("schema")
        != "harness.designwins-evaluation-identity.v1"
    ):
        raise ValueError("evaluation is not bound to immutable inputs")
    core = dict(evaluation)
    core.pop("identity", None)
    if identity.get("core_sha256") != hashlib.sha256(_canonical(core)).hexdigest():
        raise ValueError("evaluation core hash does not match sealed identity")
    for key in (
        "dataset_sha256",
        "model_manifest_sha256",
        "scorer_sha256",
    ):
        digest = identity.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"evaluation identity has invalid {key}")
    adapter_digest = identity.get("adapter_manifest_sha256")
    if adapter_digest is not None and (
        not isinstance(adapter_digest, str)
        or len(adapter_digest) != 64
        or any(char not in "0123456789abcdef" for char in adapter_digest)
    ):
        raise ValueError("evaluation identity has invalid adapter manifest")
    runtime_image_id = identity.get("runtime_image_id")
    if (
        not isinstance(runtime_image_id, str)
        or not runtime_image_id.startswith("sha256:")
        or len(runtime_image_id) != 71
    ):
        raise ValueError("evaluation identity has invalid runtime image ID")
    if identity.get("max_samples") != expected_samples:
        raise ValueError("sealed max_samples does not match evaluation")
    for key in (
        "cutoff_len",
        "max_new_tokens",
        "batch_size",
        "generation_slack_tokens",
    ):
        if not isinstance(identity.get(key), int) or identity[key] < 1:
            raise ValueError(f"evaluation identity has invalid {key}")
    if evaluation.get("schema") == "harness.designwins-chunk-aggregation.v1":
        inputs = evaluation.get("identity_inputs")
        required_inputs = (
            "raw_evaluation_sha256",
            "chunks_sha256",
            "parents_sha256",
            "aggregator_sha256",
            "generation_scorer_sha256",
            "chunk_artifact_manifest_sha256",
        )
        if not isinstance(inputs, dict):
            raise ValueError("chunk evaluation lacks immutable identity inputs")
        for key in required_inputs:
            digest = inputs.get(key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError(f"chunk evaluation has invalid {key}")
        coverage = evaluation.get("coverage")
        if (
            not isinstance(coverage, dict)
            or coverage.get("parent_cases") != expected_samples
            or int(coverage.get("excluded_physical_pins", -1)) < 0
        ):
            raise ValueError("chunk evaluation has invalid coverage evidence")
    return details, identity


def _metric_fingerprint(evaluation: dict[str, Any]) -> str:
    rows = [
        {
            "index": row["index"],
            "part": row.get("part"),
            "family": row.get("family"),
            "score": row["score"],
            "generated_tokens": row["generated_tokens"],
            "generation_budget": row.get("generation_budget"),
            "hit_generation_limit": row["hit_generation_limit"],
        }
        for row in evaluation.get("details", [])
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    candidate_repeat: dict[str, Any],
    *,
    minimum_f1_gain: float,
    maximum_family_regression: float,
    minimum_family_samples: int,
    expected_samples: int = 141,
) -> dict[str, Any]:
    baseline_details, baseline_identity = _validate(
        baseline,
        expected_samples=expected_samples,
    )
    candidate_details, candidate_identity = _validate(
        candidate,
        expected_samples=expected_samples,
    )
    repeat_details, repeat_identity = _validate(
        candidate_repeat,
        expected_samples=expected_samples,
    )
    baseline_records = [
        (row["index"], row["part"], row["family"]) for row in baseline_details
    ]
    candidate_records = [
        (row["index"], row["part"], row["family"]) for row in candidate_details
    ]
    repeat_records = [
        (row["index"], row["part"], row["family"]) for row in repeat_details
    ]
    if baseline_records != candidate_records or candidate_records != repeat_records:
        raise ValueError("evaluations do not cover identical held-out records")
    if not (
        baseline.get("coverage")
        == candidate.get("coverage")
        == candidate_repeat.get("coverage")
    ):
        raise ValueError("evaluations do not cover identical grounded labels")
    common_identity_keys = (
        "dataset_sha256",
        "model_manifest_sha256",
        "scorer_sha256",
        "runtime_image_id",
        "max_samples",
        "cutoff_len",
        "max_new_tokens",
        "batch_size",
        "generation_slack_tokens",
    )
    for key in common_identity_keys:
        if not (
            baseline_identity.get(key)
            == candidate_identity.get(key)
            == repeat_identity.get(key)
        ):
            raise ValueError(f"evaluation identity mismatch for {key}")
    adapter_digest = candidate_identity.get("adapter_manifest_sha256")
    if (
        baseline_identity.get("adapter_manifest_sha256") is not None
        or not adapter_digest
        or adapter_digest != repeat_identity.get("adapter_manifest_sha256")
    ):
        raise ValueError("candidate adapter identity is missing or inconsistent")

    base = baseline["summary"]
    cand = candidate["summary"]
    repeat = candidate_repeat["summary"]

    f1_gain = float(cand["mean_leaf_f1"]) - float(base["mean_leaf_f1"])
    valid_json_delta = float(cand["valid_json_rate"]) - float(
        base["valid_json_rate"]
    )
    family_checks: dict[str, dict[str, Any]] = {}
    regressions: list[str] = []
    base_families = base.get("by_family") or {}
    cand_families = cand.get("by_family") or {}
    if set(base_families) != set(cand_families):
        raise ValueError("baseline and candidate family sets differ")
    for family in sorted(base_families):
        base_row = base_families[family]
        cand_row = cand_families[family]
        samples = min(int(base_row["samples"]), int(cand_row["samples"]))
        delta = float(cand_row["mean_leaf_f1"]) - float(base_row["mean_leaf_f1"])
        eligible = samples >= minimum_family_samples
        passed = not eligible or delta >= -maximum_family_regression
        family_checks[family] = {
            "samples": samples,
            "baseline_leaf_f1": float(base_row["mean_leaf_f1"]),
            "candidate_leaf_f1": float(cand_row["mean_leaf_f1"]),
            "delta": delta,
            "gate_applies": eligible,
            "passed": passed,
        }
        if not passed:
            regressions.append(family)

    candidate_fingerprint = _metric_fingerprint(candidate)
    repeat_fingerprint = _metric_fingerprint(candidate_repeat)
    reproduction_passed = (
        candidate_fingerprint == repeat_fingerprint
        and cand["mean_leaf_f1"] == repeat["mean_leaf_f1"]
        and cand["valid_json_rate"] == repeat["valid_json_rate"]
    )
    checks = {
        "minimum_f1_gain": f1_gain >= minimum_f1_gain,
        "valid_json_non_regression": valid_json_delta >= 0,
        "family_non_regression": not regressions,
        "deterministic_reproduction": reproduction_passed,
        "candidate_generation_complete": int(cand["generation_limit_hits"]) == 0,
        "repeat_generation_complete": int(repeat["generation_limit_hits"]) == 0,
    }
    return {
        "schema": "harness.designwins.qualification.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "baseline_leaf_f1": float(base["mean_leaf_f1"]),
            "candidate_leaf_f1": float(cand["mean_leaf_f1"]),
            "leaf_f1_gain": f1_gain,
            "minimum_required_gain": minimum_f1_gain,
            "baseline_valid_json_rate": float(base["valid_json_rate"]),
            "candidate_valid_json_rate": float(cand["valid_json_rate"]),
            "valid_json_delta": valid_json_delta,
            "candidate_exact_rate": float(cand["exact_rate"]),
            "baseline_generation_limit_hits": int(
                base["generation_limit_hits"]
            ),
            "candidate_generation_limit_hits": int(
                cand["generation_limit_hits"]
            ),
        },
        "family_checks": family_checks,
        "regressed_families": regressions,
        "reproduction": {
            "candidate_fingerprint": candidate_fingerprint,
            "repeat_fingerprint": repeat_fingerprint,
        },
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != temporary.read_bytes():
                raise FileExistsError(
                    f"refusing to overwrite qualification evidence: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-repeat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-f1-gain", type=float, default=0.05)
    parser.add_argument("--maximum-family-regression", type=float, default=0.02)
    parser.add_argument("--minimum-family-samples", type=int, default=5)
    args = parser.parse_args()
    if not 0 <= args.minimum_f1_gain <= 1:
        parser.error("--minimum-f1-gain must be between 0 and 1")
    if not 0 <= args.maximum_family_regression <= 1:
        parser.error("--maximum-family-regression must be between 0 and 1")
    if args.minimum_family_samples < 1:
        parser.error("--minimum-family-samples must be positive")
    return args


def main() -> int:
    args = parse_args()
    result = bind_identity(
        compare(
        _load(args.baseline),
        _load(args.candidate),
        _load(args.candidate_repeat),
        minimum_f1_gain=args.minimum_f1_gain,
        maximum_family_regression=args.maximum_family_regression,
        minimum_family_samples=args.minimum_family_samples,
        expected_samples=141,
        ),
        baseline=args.baseline,
        candidate=args.candidate,
        candidate_repeat=args.candidate_repeat,
    )
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
