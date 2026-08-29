from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_categoryrank_persistence_pilot.py"
SPEC = importlib.util.spec_from_file_location("categoryrank_persistence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def _row(brand: str, week: str, mentions: int = 2) -> dict:
    return {
        "brand_domain": brand,
        "kim_slug": "microcontrollers",
        "model_id": "lane-a",
        "time_window": week,
        "n_mentions": mentions,
        "avg_strength": 80.0,
        "avg_rank": 2.0,
    }


def test_transition_examples_are_deidentified_and_time_bounded(tmp_path: Path) -> None:
    source = tmp_path / "categoryrank.jsonl"
    rows = [
        _row("a.example", "2026-W32"),
        _row("a.example", "2026-W33"),
        _row("a.example", "2026-W34"),
        _row("b.example", "2026-W32"),
        _row("b.example", "2026-W34"),
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    examples = list(pilot.transition_examples(source, through_week="2026-W34"))

    assert [(week, label) for week, _, label in examples] == [
        ("2026-W32", 1),
        ("2026-W33", 1),
        ("2026-W32", 0),
    ]
    assert all(len(features) == len(pilot.FEATURE_NAMES) for _, features, _ in examples)
    assert all("a.example" not in repr(example) for example in examples)


def test_transition_examples_reject_noncanonical_sorting(tmp_path: Path) -> None:
    source = tmp_path / "categoryrank.jsonl"
    rows = [
        _row("b.example", "2026-W32"),
        _row("a.example", "2026-W32"),
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="canonical-sorted"):
        list(pilot.transition_examples(source, through_week="2026-W34"))


def test_auc_handles_perfect_and_tied_rankings() -> None:
    assert pilot._auc([0, 1], [0.1, 0.9]) == 1.0
    assert pilot._auc([0, 1], [0.5, 0.5]) == 0.5
