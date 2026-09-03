#!/usr/bin/env python3
"""Seal a full-corpus DesignWins teacher-forced training-signal decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _write_once(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == payload:
            return
        raise FileExistsError(f"refusing to overwrite qualification: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _recompute(evaluation: dict[str, Any]) -> tuple[list[dict[str, Any]], dict]:
    if evaluation.get("schema") != "harness.designwins-teacher-forced-evaluation.v1":
        raise ValueError("unexpected teacher-forced evaluation schema")
    if evaluation.get("max_samples") != 141:
        raise ValueError("evaluation must cover all 141 held-out records")
    passes = evaluation.get("passes")
    if not isinstance(passes, list) or len(passes) != 1:
        raise ValueError("evaluation must contain exactly one pass")
    details = passes[0].get("details")
    if not isinstance(details, list) or len(details) != 141:
        raise ValueError("evaluation pass is incomplete")
    if [row.get("index") for row in details] != list(range(141)):
        raise ValueError("evaluation indices are incomplete or out of order")
    for row in details:
        scored_tokens = row.get("scored_tokens")
        correct_tokens = row.get("correct_tokens")
        total_nll = row.get("total_nll")
        mean_nll = row.get("mean_token_nll")
        if (
            not isinstance(scored_tokens, int)
            or scored_tokens < 1
            or not isinstance(correct_tokens, int)
            or not 0 <= correct_tokens <= scored_tokens
            or not math.isfinite(float(total_nll))
            or float(total_nll) < 0
            or not math.isfinite(float(mean_nll))
            or float(mean_nll) < 0
        ):
            raise ValueError(f"invalid metrics for evaluation row {row.get('index')}")
    token_count = sum(int(row["scored_tokens"]) for row in details)
    total_nll = sum(float(row["total_nll"]) for row in details)
    correct = sum(int(row["correct_tokens"]) for row in details)
    mean_nll = total_nll / token_count
    fingerprint = hashlib.sha256(_canonical(details)).hexdigest()
    summary = passes[0].get("summary")
    expected = {
        "samples": 141,
        "scored_tokens": token_count,
        "total_nll": total_nll,
        "mean_token_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 50)),
        "token_accuracy": correct / token_count,
        "fingerprint": fingerprint,
    }
    if not isinstance(summary, dict):
        raise ValueError("evaluation summary is missing")
    for key, value in expected.items():
        actual = summary.get(key)
        if isinstance(value, float):
            if not math.isclose(float(actual), value, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"evaluation summary mismatch for {key}")
        elif actual != value:
            raise ValueError(f"evaluation summary mismatch for {key}")
    return details, summary


def compare(args: argparse.Namespace) -> dict[str, Any]:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    repeat = json.loads(args.repeat.read_text(encoding="utf-8"))
    baseline_details, baseline_summary = _recompute(baseline)
    candidate_details, candidate_summary = _recompute(candidate)
    repeat_details, repeat_summary = _recompute(repeat)
    common_keys = (
        "model",
        "dataset",
        "max_samples",
        "cutoff_len",
        "max_response_tokens",
    )
    for key in common_keys:
        if not baseline.get(key) == candidate.get(key) == repeat.get(key):
            raise ValueError(f"evaluation mismatch for {key}")
    identities = [
        [(row["index"], row["part"], row["family"]) for row in rows]
        for rows in (baseline_details, candidate_details, repeat_details)
    ]
    if not identities[0] == identities[1] == identities[2]:
        raise ValueError("evaluations do not cover identical records")
    if baseline.get("adapter") is not None:
        raise ValueError("baseline unexpectedly uses an adapter")
    if not candidate.get("adapter") or candidate.get("adapter") != repeat.get(
        "adapter"
    ):
        raise ValueError("candidate adapter is missing or inconsistent")

    baseline_nll = float(baseline_summary["mean_token_nll"])
    candidate_nll = float(candidate_summary["mean_token_nll"])
    repeat_nll = float(repeat_summary["mean_token_nll"])
    nll_reduction = (baseline_nll - candidate_nll) / baseline_nll
    accuracy_delta = float(candidate_summary["token_accuracy"]) - float(
        baseline_summary["token_accuracy"]
    )
    record_wins = sum(
        float(candidate_row["mean_token_nll"])
        < float(baseline_row["mean_token_nll"])
        for baseline_row, candidate_row in zip(
            baseline_details, candidate_details, strict=True
        )
    )
    record_win_rate = record_wins / len(baseline_details)
    repeat_nll_delta = abs(candidate_nll - repeat_nll)
    repeat_accuracy_delta = abs(
        float(candidate_summary["token_accuracy"])
        - float(repeat_summary["token_accuracy"])
    )
    checks = {
        "minimum_relative_nll_reduction": (
            nll_reduction >= args.minimum_relative_nll_reduction
        ),
        "maximum_token_accuracy_regression": (
            accuracy_delta >= -args.maximum_token_accuracy_regression
        ),
        "minimum_record_win_rate": record_win_rate >= args.minimum_record_win_rate,
        "repeat_mean_nll": repeat_nll_delta <= args.maximum_repeat_nll_delta,
        "repeat_token_accuracy": (
            repeat_accuracy_delta <= args.maximum_repeat_accuracy_delta
        ),
    }
    return {
        "schema": "harness.designwins-teacher-forced-qualification.v1",
        "passed": all(checks.values()),
        "scope": "training-signal-only",
        "production_promotion_eligible": False,
        "production_gate": "requires separate frozen autoregressive evaluation",
        "checks": checks,
        "thresholds": {
            "minimum_relative_nll_reduction": args.minimum_relative_nll_reduction,
            "maximum_token_accuracy_regression": (
                args.maximum_token_accuracy_regression
            ),
            "minimum_record_win_rate": args.minimum_record_win_rate,
            "maximum_repeat_nll_delta": args.maximum_repeat_nll_delta,
            "maximum_repeat_accuracy_delta": args.maximum_repeat_accuracy_delta,
        },
        "metrics": {
            "samples": 141,
            "scored_tokens": baseline_summary["scored_tokens"],
            "baseline_mean_token_nll": baseline_nll,
            "candidate_mean_token_nll": candidate_nll,
            "candidate_repeat_mean_token_nll": repeat_nll,
            "relative_nll_reduction": nll_reduction,
            "baseline_token_accuracy": baseline_summary["token_accuracy"],
            "candidate_token_accuracy": candidate_summary["token_accuracy"],
            "candidate_repeat_token_accuracy": repeat_summary["token_accuracy"],
            "token_accuracy_delta": accuracy_delta,
            "record_win_rate": record_win_rate,
            "candidate_repeat_nll_delta": repeat_nll_delta,
            "candidate_repeat_accuracy_delta": repeat_accuracy_delta,
        },
        "identity": {
            "baseline_sha256": _sha256(args.baseline),
            "candidate_sha256": _sha256(args.candidate),
            "candidate_repeat_sha256": _sha256(args.repeat),
            "dataset_sha256": _sha256(args.dataset),
            "model_manifest_sha256": _sha256(args.model_manifest),
            "adapter_manifest_sha256": _sha256(args.adapter_manifest),
            "evaluator_sha256": _sha256(args.evaluator),
            "evaluator_dependency_sha256": _sha256(args.evaluator_dependency),
            "comparator_sha256": _sha256(Path(__file__)),
            "runtime_image_id": args.runtime_image_id,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--repeat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--adapter-manifest", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--evaluator-dependency", required=True, type=Path)
    parser.add_argument("--runtime-image-id", required=True)
    parser.add_argument("--minimum-relative-nll-reduction", type=float, default=0.1)
    parser.add_argument(
        "--maximum-token-accuracy-regression", type=float, default=0.002
    )
    parser.add_argument("--minimum-record-win-rate", type=float, default=0.6)
    parser.add_argument("--maximum-repeat-nll-delta", type=float, default=1e-6)
    parser.add_argument("--maximum-repeat-accuracy-delta", type=float, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        not args.runtime_image_id.startswith("sha256:")
        or len(args.runtime_image_id) != 71
        or not 0 <= args.minimum_relative_nll_reduction <= 1
        or not 0 <= args.maximum_token_accuracy_regression <= 1
        or not 0 <= args.minimum_record_win_rate <= 1
        or args.maximum_repeat_nll_delta < 0
        or args.maximum_repeat_accuracy_delta < 0
    ):
        raise ValueError("invalid runtime identity or qualification threshold")
    result = compare(args)
    _write_once(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
