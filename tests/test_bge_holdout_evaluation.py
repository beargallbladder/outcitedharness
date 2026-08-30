from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_bge_holdout.py"
SPEC = importlib.util.spec_from_file_location("evaluate_bge_holdout", SCRIPT)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_load_sample_is_seeded_and_stratified(tmp_path: Path) -> None:
    holdout = tmp_path / "holdout.jsonl"
    rows = [
        {
            "text": f"query-{index}",
            "positive_label": f"positive-{index}",
            "hard_negative_label": f"negative-{index}",
            "label_type": "a" if index < 4 else "b",
        }
        for index in range(8)
    ]
    holdout.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    first = evaluation.load_sample(holdout, per_type=2, seed=42)
    second = evaluation.load_sample(holdout, per_type=2, seed=42)

    assert first == second
    assert len(first) == 4
    assert {row[0] for row in first} == {"a", "b"}


def test_comparison_requires_overall_gain_without_type_regression() -> None:
    baseline = {
        "overall": {"positive_at_1": 0.7},
        "by_type": {
            "a": {"positive_at_1": 0.6},
            "b": {"positive_at_1": 0.8},
        },
    }
    candidate = {
        "overall": {"positive_at_1": 0.75},
        "by_type": {
            "a": {"positive_at_1": 0.7},
            "b": {"positive_at_1": 0.8},
        },
    }
    improved = evaluation.comparison(baseline, candidate)
    assert improved["candidate_improves_without_type_regression"] is True

    candidate["by_type"]["b"]["positive_at_1"] = 0.79
    regressed = evaluation.comparison(baseline, candidate)
    assert regressed["candidate_improves_without_type_regression"] is False
