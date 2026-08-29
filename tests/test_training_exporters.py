from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from harness.training.exporters import (
    ExportValidationError,
    export_harness_pass_pairs,
    extract_git_candidates,
    load_designwins_text_pairs,
    load_designwins_vision_pairs,
    load_native_designwins_text_pairs,
    load_native_designwins_vision_pairs,
)
from harness.training.models import DataUse
from harness.training.security import SecretDetectedError


def _provenance(
    kind: str,
    *,
    record_id: str = "record-1",
    data_use: str = "training",
) -> dict:
    return {
        "source_kind": kind,
        "source_uri": f"{kind}://dataset/{record_id}",
        "source_record_id": record_id,
        "collected_at": "2026-08-29T12:00:00Z",
        "content_sha256": hashlib.sha256(record_id.encode()).hexdigest(),
        "lineage_id": "lineage-1",
        "license": "internal-approved",
        "data_use": data_use,
    }


def test_harness_export_includes_only_verified_pass_and_redacts_pii(tmp_path: Path):
    cases = [
        {
            "id": "case-1",
            "prompt": "Reply to person@example.com",
            "provenance": _provenance("harness"),
        }
    ]
    results = [
        {
            "case_id": "case-1",
            "run_id": "run-1",
            "verdict": "FAIL",
            "evaluator": "exact",
            "answer": "bad",
        },
        {
            "case_id": "case-1",
            "run_id": "run-1",
            "verdict": "PASS",
            "evaluator": "exact",
            "answer": "Done for person@example.com",
        },
    ]
    destination = tmp_path / "pairs.jsonl"
    pairs = export_harness_pass_pairs(cases, results, destination=destination)
    assert len(pairs) == 1
    assert "[REDACTED_EMAIL]" in pairs[0].prompt
    assert "[REDACTED_EMAIL]" in pairs[0].response
    written = json.loads(destination.read_text())
    assert written["metadata"]["verdict"] == "PASS"


def test_harness_pass_requires_evaluator_and_complete_provenance():
    cases = [{"id": "case-1", "prompt": "prompt"}]
    with pytest.raises(ExportValidationError, match="evaluator"):
        export_harness_pass_pairs(
            cases,
            [{"case_id": "case-1", "verdict": "PASS", "answer": "answer"}],
        )
    with pytest.raises(ExportValidationError, match="provenance"):
        export_harness_pass_pairs(
            cases,
            [
                {
                    "case_id": "case-1",
                    "verdict": "PASS",
                    "evaluator": "exact",
                    "answer": "answer",
                }
            ],
        )


def test_designwins_text_and_vision_require_provenance():
    base = {
        "id": "pair-1",
        "instruction": "Describe the board",
        "output": "A two-layer board",
        "provenance": _provenance("designwins"),
    }
    text = load_designwins_text_pairs([base])
    assert text[0].prompt == "Describe the board"

    vision = load_designwins_vision_pairs(
        [
            {
                **base,
                "images": [{"uri": "file:///board.png", "sha256": "a" * 64}],
            }
        ]
    )
    assert vision[0].image_sha256 == ("a" * 64,)

    with pytest.raises(ExportValidationError, match="provenance"):
        load_designwins_text_pairs(
            [{"prompt": "p", "response": "r", "provenance": {"source_kind": "designwins"}}]
        )


def test_native_designwins_pairs_gain_provenance_and_image_hashes(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images" / "part-a"
    images.mkdir(parents=True)
    page = images / "page_1.png"
    page.write_bytes(b"png")
    target = '{"pins":[{"pin_no":1,"name":"VCC"}]}'
    text_source = tmp_path / "text_pairs.jsonl"
    vision_source = tmp_path / "vision_pairs.jsonl"
    text_source.write_text(
        json.dumps(
            {
                "part": "part-a",
                "prompt": "Extract pins",
                "target": target,
            }
        )
        + "\n"
    )
    vision_source.write_text(
        json.dumps(
            {
                "part": "part-a",
                "images": [str(page)],
                "prompt": "Extract pins",
                "target": target,
            }
        )
        + "\n"
    )

    text = load_native_designwins_text_pairs(text_source)
    vision = load_native_designwins_vision_pairs(vision_source)

    assert text[0].provenance.lineage_id == "designwins:part-a"
    assert text[0].response == target
    assert vision[0].provenance.lineage_id == text[0].provenance.lineage_id
    assert vision[0].image_sha256 == (hashlib.sha256(b"png").hexdigest(),)
    assert vision[0].image_uris == (
        "dataset://designwins/images/part-a/page_1.png",
    )


def test_native_designwins_projects_full_truth_onto_prompt_pin_schema(
    tmp_path: Path,
) -> None:
    source = tmp_path / "text_pairs.jsonl"
    source.write_text(
        json.dumps(
            {
                "part": "part-a",
                "prompt": 'Return ONLY valid JSON: {"pins": [...]}',
                "target": json.dumps(
                    {
                        "overview": {"manufacturer": "Example"},
                        "pinout": {
                            "pin_functions_summary": [
                                {"pin_no": 1, "name": "VCC"},
                                {"pin_no": 2, "name": "GND"},
                            ],
                            "total_pins_extracted": 2,
                        },
                    }
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    pairs = load_native_designwins_text_pairs(source)

    assert json.loads(pairs[0].response) == {
        "pins": [
            {"pin_no": 1, "name": "VCC"},
            {"pin_no": 2, "name": "GND"},
        ]
    }


def test_secret_bearing_training_data_is_rejected():
    with pytest.raises(SecretDetectedError):
        load_designwins_text_pairs(
            [
                {
                    "prompt": "Use api_key=supersecretvalue",
                    "response": "ok",
                    "provenance": _provenance("designwins"),
                }
            ]
        )


def test_git_candidates_are_always_quarantined():
    provenance = _provenance("git", data_use="quarantine")
    provenance["revision"] = "b" * 40
    candidates = extract_git_candidates(
        [
            {
                "problem": "Fix the parser",
                "patch": "diff --git a/parser.py b/parser.py\n--- a/parser.py\n+++ b/parser.py",
                "tests": [
                    {
                        "command": "pytest -q",
                        "status": "pass",
                        "output_sha256": "c" * 64,
                    }
                ],
                "provenance": provenance,
            }
        ]
    )
    assert candidates[0].data_use is DataUse.QUARANTINE
    assert candidates[0].approved_for_training is False

    bad = dict(provenance)
    bad["data_use"] = "training"
    with pytest.raises(ExportValidationError, match="quarantine"):
        extract_git_candidates(
            [
                {
                    "problem": "fix",
                    "patch": "diff --git a/a b/a",
                    "tests": [
                        {
                            "command": "pytest",
                            "status": "unknown",
                        }
                    ],
                    "provenance": bad,
                }
            ]
        )
