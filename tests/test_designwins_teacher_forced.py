from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_designwins_teacher_forced.py"
SPEC = importlib.util.spec_from_file_location(
    "compare_designwins_teacher_forced", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def _evaluation(*, adapter: str | None, mean_nll: float, correct: int) -> dict:
    details = [
        {
            "index": index,
            "part": f"part-{index}",
            "family": f"family-{index % 4}",
            "prompt_tokens": 20,
            "scored_tokens": 10,
            "total_nll": mean_nll * 10,
            "mean_token_nll": mean_nll,
            "correct_tokens": correct,
            "token_accuracy": correct / 10,
        }
        for index in range(141)
    ]
    payload = json.dumps(
        details, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    token_count = 1410
    total_nll = mean_nll * token_count
    return {
        "schema": "harness.designwins-teacher-forced-evaluation.v1",
        "model": "/training/model",
        "adapter": adapter,
        "dataset": "/training/test.json",
        "max_samples": 141,
        "cutoff_len": 4096,
        "max_response_tokens": 8192,
        "passes": [
            {
                "summary": {
                    "samples": 141,
                    "scored_tokens": token_count,
                    "total_nll": total_nll,
                    "mean_token_nll": mean_nll,
                    "perplexity": math.exp(mean_nll),
                    "token_accuracy": correct / 10,
                    "elapsed_seconds": 1,
                    "fingerprint": hashlib.sha256(payload).hexdigest(),
                },
                "details": details,
            }
        ],
    }


def _arguments(tmp_path: Path) -> argparse.Namespace:
    values = {
        "baseline": _evaluation(adapter=None, mean_nll=1.0, correct=8),
        "candidate": _evaluation(
            adapter="/training/adapter", mean_nll=0.5, correct=9
        ),
        "repeat": _evaluation(
            adapter="/training/adapter", mean_nll=0.5, correct=9
        ),
    }
    paths = {}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        paths[name] = path
    evidence = {}
    for name in (
        "dataset",
        "model_manifest",
        "adapter_manifest",
        "evaluator",
        "evaluator_dependency",
    ):
        path = tmp_path / name
        path.write_text(name)
        evidence[name] = path
    return argparse.Namespace(
        **paths,
        **evidence,
        output=tmp_path / "qualification.json",
        runtime_image_id="sha256:" + "a" * 64,
        minimum_relative_nll_reduction=0.1,
        maximum_token_accuracy_regression=0.002,
        minimum_record_win_rate=0.6,
        maximum_repeat_nll_delta=1e-6,
        maximum_repeat_accuracy_delta=0,
    )


def test_teacher_forced_comparison_passes_full_reproducible_gain(
    tmp_path: Path,
) -> None:
    result = comparison.compare(_arguments(tmp_path))

    assert result["passed"] is True
    assert result["production_promotion_eligible"] is False
    assert result["metrics"]["relative_nll_reduction"] == 0.5
    assert result["metrics"]["record_win_rate"] == 1


def test_teacher_forced_comparison_recomputes_summary(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    value = json.loads(args.candidate.read_text())
    value["passes"][0]["summary"]["mean_token_nll"] = 0
    args.candidate.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="summary mismatch"):
        comparison.compare(args)
