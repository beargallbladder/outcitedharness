from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_designwins_chunk_evaluation.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("aggregate_designwins_chunks", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
aggregator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregator)


def _fixture():
    pins = [
        {"pin_no": 1, "name": "PA0"},
        {"pin_no": 2, "name": "PA1"},
        {"pin_no": 3, "name": "UNGROUNDED"},
    ]
    parent = {
        "pair_id": "part-1",
        "response": json.dumps({"pins": pins}),
    }
    chunks = [
        {
            "response": json.dumps({"pins": [pins[0]]}),
            "metadata": {
                "parent_pair_id": "part-1",
                "chunk_index": 0,
                "chunk_count": 2,
                "part": "FAMILY_part-1",
            },
        },
        {
            "response": json.dumps({"pins": [pins[1]]}),
            "metadata": {
                "parent_pair_id": "part-1",
                "chunk_index": 1,
                "chunk_count": 2,
                "part": "FAMILY_part-1",
            },
        },
    ]
    details = [
        {
            "index": index,
            "expected_tokens": 10,
            "generated_tokens": 8,
            "generation_budget": 20,
            "hit_generation_limit": False,
            "response": chunk["response"],
        }
        for index, chunk in enumerate(chunks)
    ]
    return {"details": details, "model": "base", "adapter": None}, chunks, [parent]


def test_chunk_scores_aggregate_to_parent_case_and_report_coverage():
    raw, chunks, parents = _fixture()

    result = aggregator.aggregate(raw, chunks, parents)

    assert result["summary"]["samples"] == 1
    assert result["summary"]["exact_rate"] == 1
    assert result["details"][0]["chunk_count"] == 2
    assert result["coverage"] == {
        "parent_cases": 1,
        "original_physical_pins": 3,
        "evaluated_grounded_physical_pins": 2,
        "excluded_physical_pins": 1,
        "evaluated_pin_rate": 2 / 3,
    }


def test_chunk_aggregation_fails_when_parent_is_incomplete():
    raw, chunks, parents = _fixture()
    raw["details"] = raw["details"][:1]
    chunks = chunks[:1]

    with pytest.raises(ValueError, match="incomplete chunks"):
        aggregator.aggregate(raw, chunks, parents)
