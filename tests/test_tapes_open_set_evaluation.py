from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_tapes_open_set.py"
SPEC = importlib.util.spec_from_file_location("evaluate_tapes_open_set", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_require_pin_accepts_exact_sha_prefix_and_rejects_drift(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"frozen": true}\n', encoding="utf-8")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()[:16]

    assert evaluation.require_pin(artifact, expected) == expected
    with pytest.raises(ValueError, match="pin mismatch"):
        evaluation.require_pin(artifact, "0" * 16)


def test_normalize_matches_tapes_retrieval_protocol() -> None:
    assert evaluation.normalize("  DC-DC  Converters! ") == "dcdc converters"
    assert evaluation.normalize("Voltage_Regulators") == "voltage_regulators"


def test_load_jsonl_requires_one_metadata_row(tmp_path: Path) -> None:
    artifact = tmp_path / "split.jsonl"
    artifact.write_text(
        "\n".join(
            [
                json.dumps({"_meta": {"version": "v1.1"}}),
                json.dumps({"query": "MCU", "positive_doc_texts": ["Microcontroller"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    meta, rows = evaluation._load_jsonl(artifact)

    assert meta["version"] == "v1.1"
    assert rows[0]["query"] == "MCU"


def test_comparison_requires_every_pinned_metric_to_match() -> None:
    expected = {
        "kim_tag": {"top1_accuracy": 0.765, "top3_accuracy": 0.9726},
        "retrieval": {
            "recall_at_1": 0.3103,
            "recall_at_3": 0.399,
            "recall_at_5": 0.5123,
            "recall_at_10": 0.6453,
            "category_alignment_recall_at_1": 0.7931,
        },
    }
    observed = {
        "kim_tag": dict(expected["kim_tag"]),
        "retrieval": {
            "recall_at_1": 0.3103,
            "recall_at_3": 0.399,
            "recall_at_5": 0.5123,
            "recall_at_10": 0.6453,
            "by_domain": {
                "category_alignment": {"recall_at_1": 0.7931},
            },
        },
    }

    assert evaluation._comparison(observed, expected)["exact_reproduction"] is True
    observed["retrieval"]["recall_at_1"] = 0.3
    comparison = evaluation._comparison(observed, expected)
    assert comparison["exact_reproduction"] is False
    assert comparison["metrics"]["retrieval_r1"]["delta"] == -0.0103
