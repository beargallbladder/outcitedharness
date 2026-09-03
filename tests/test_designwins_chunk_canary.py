from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "select_designwins_chunk_canary.py"
SPEC = importlib.util.spec_from_file_location("select_designwins_canary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def _chunk(parent: str, index: int, count: int) -> dict:
    return {
        "pair_id": f"{parent}-{index}",
        "metadata": {
            "parent_pair_id": parent,
            "chunk_index": index,
            "chunk_count": count,
        },
    }


def test_canary_selects_complete_groups_for_parent_cases():
    parents = [{"pair_id": "a"}, {"pair_id": "b"}, {"pair_id": "c"}]
    chunks = [
        _chunk("a", 0, 2),
        _chunk("a", 1, 2),
        _chunk("b", 0, 1),
        _chunk("c", 0, 1),
    ]
    llama = [{"row": index} for index in range(4)]

    selected, selected_llama, parent_ids = selector.select(
        parents,
        chunks,
        llama,
        parent_cases=2,
    )

    assert parent_ids == ["a", "b"]
    assert [row["pair_id"] for row in selected] == ["a-0", "a-1", "b-0"]
    assert selected_llama == llama[:3]


def test_canary_rejects_incomplete_parent_group():
    with pytest.raises(ValueError, match="incomplete chunks"):
        selector.select(
            [{"pair_id": "a"}],
            [_chunk("a", 0, 2)],
            [{"row": 0}],
            parent_cases=1,
        )
