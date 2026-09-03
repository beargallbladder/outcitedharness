from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_designwins_text.py"
SPEC = importlib.util.spec_from_file_location("evaluate_designwins_text", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_extract_json_handles_thinking_and_fenced_output() -> None:
    output = '<think>reasoning</think>\n```json\n{"pins": [1, 2]}\n```'
    assert evaluation.extract_json(output) == {"pins": [1, 2]}


def test_extract_json_finds_first_embedded_document() -> None:
    output = 'Here is the result: {"pins": [{"name": "VCC"}]} trailing text'
    assert evaluation.extract_json(output) == {"pins": [{"name": "VCC"}]}


def test_extract_json_rejects_nested_fragment_from_incomplete_document() -> None:
    output = '{"pins": [{"name": "VCC"}'
    assert evaluation.extract_json(output) is None


def test_score_json_reports_leaf_precision_recall_and_exactness() -> None:
    expected = {"pins": [{"name": "VCC"}, {"name": "GND"}], "count": 2}
    predicted = {"pins": [{"name": "VCC"}, {"name": "IO"}], "count": 2}

    score = evaluation.score_json(expected, predicted)

    assert score["valid_json"] is True
    assert score["exact"] is False
    assert score["leaf_precision"] == 2 / 3
    assert score["leaf_recall"] == 2 / 3
    assert score["leaf_f1"] == 2 / 3


def test_score_json_rejects_unparseable_output() -> None:
    assert evaluation.score_json({"pins": []}, None) == {
        "valid_json": False,
        "exact": False,
        "leaf_precision": 0.0,
        "leaf_recall": 0.0,
        "leaf_f1": 0.0,
    }


def test_target_token_lengths_measure_each_expected_response() -> None:
    class Tokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool):
            assert add_special_tokens is False
            return {"input_ids": text.split()}

    assert evaluation.target_token_lengths(
        Tokenizer(),
        [{"output": "one two"}, {"output": "one two three"}],
    ) == [2, 3]


def test_load_records_preserves_canonical_part_metadata(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": "Extract pins",
                "response": '{"pins":[]}',
                "metadata": {"part": "microchip_pic18"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = evaluation.load_records(path)

    assert records == [
        {
            "instruction": "Extract pins",
            "output": '{"pins":[]}',
            "metadata": {"part": "microchip_pic18"},
        }
    ]
    assert evaluation._record_identity(records[0], 0) == (
        "microchip_pic18",
        "microchip",
    )


def test_load_records_rejects_missing_output(tmp_path: Path) -> None:
    path = tmp_path / "test.json"
    path.write_text('[{"instruction":"Extract pins"}]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="lacks an output"):
        evaluation.load_records(path)


def test_evaluation_output_is_write_once(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.json"
    evaluation._write_json(path, {"first": True})

    with pytest.raises(FileExistsError):
        evaluation._write_json(path, {"first": False})

    assert json.loads(path.read_text()) == {"first": True}


def test_summarize_details_reports_family_slices() -> None:
    rows = [
        {
            "family": "microchip",
            "hit_generation_limit": False,
            "score": {
                "valid_json": True,
                "exact": True,
                "leaf_precision": 1.0,
                "leaf_recall": 1.0,
                "leaf_f1": 1.0,
            },
        },
        {
            "family": "st",
            "hit_generation_limit": True,
            "score": {
                "valid_json": False,
                "exact": False,
                "leaf_precision": 0.0,
                "leaf_recall": 0.0,
                "leaf_f1": 0.0,
            },
        },
    ]

    summary = evaluation.summarize_details(rows)

    assert summary["samples"] == 2
    assert summary["mean_leaf_f1"] == 0.5
    assert summary["generation_limit_hits"] == 1
    assert summary["by_family"]["microchip"]["mean_leaf_f1"] == 1.0
    assert summary["by_family"]["st"]["valid_json_rate"] == 0.0
