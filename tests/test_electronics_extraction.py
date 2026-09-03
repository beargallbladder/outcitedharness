from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from harness.electronics.corpus import CorpusInputs, build_corpus_registry
from harness.electronics.extraction import (
    build_extraction_work_queue,
    iter_ground_truth_claims,
    iter_pinout_row_claims,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_owned_extractors_preserve_page_package_and_split(tmp_path: Path):
    pdfs = tmp_path / "pdfs"
    gt = tmp_path / "gt"
    validated = tmp_path / "validated"
    rows = tmp_path / "rows"
    for path in (pdfs, gt, validated, rows / "canonical", rows / "images" / "train"):
        path.mkdir(parents=True, exist_ok=True)
    pdf = pdfs / "acme_atom1.pdf"
    pdf.write_bytes(b"%PDF-atom-one")
    gt_record = {
        "overview": {
            "manufacturer": "acme",
            "part_number": "ATOM1",
            "description": "An MCU for deterministic tests.",
        },
        "clock_specs": {"max_freq_mhz": 100},
        "_meta": {"batch_id": "msgbatch_gt"},
    }
    (gt / "acme_atom1.json").write_text(json.dumps(gt_record), encoding="utf-8")
    registry = build_corpus_registry(
        CorpusInputs(
            pdf_root=pdfs,
            ground_truth_root=gt,
            validated_root=validated,
        ),
        hash_workers=1,
    )
    document_sha = _sha(pdf)
    header = rows / "images" / "train" / "header.png"
    body = rows / "images" / "train" / "body.png"
    header.write_bytes(b"header")
    body.write_bytes(b"body")
    row = {
        "alignment": {
            "body_bbox": [10.0, 20.0, 30.0, 40.0],
            "header_bbox": [1.0, 2.0, 3.0, 4.0],
            "package": "LQFP2",
            "page_1based": 5,
        },
        "example_id": "example-1",
        "image_sha256": [_sha(header), _sha(body)],
        "images": [
            "images/train/header.png",
            "images/train/body.png",
        ],
        "prompt": "Extract one row.",
        "provenance": {
            "frontier_batch_id": "msgbatch_rows",
            "pdf_sha256": document_sha,
        },
        "record_id": "acme_atom1",
        "response": json.dumps(
            {
                "pins": [
                    {
                        "pin_no": 1,
                        "name": "PA0",
                        "type": "gpio",
                        "functions": ["ADC0"],
                        "dir": "I/O",
                    }
                ]
            },
            separators=(",", ":"),
        ),
        "split": "train",
    }
    canonical = rows / "canonical" / "train.jsonl"
    canonical.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = {
        "schema": "harness.pinout-vision-row-dataset.v1",
        "artifacts": {
            "canonical/train.jsonl": {
                "bytes": canonical.stat().st_size,
                "sha256": _sha(canonical),
            }
        },
    }
    (rows / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    captured_at = datetime(2026, 9, 1, tzinfo=timezone.utc)

    gt_claims = list(iter_ground_truth_claims(registry, created_at=captured_at))
    pin_claims = list(
        iter_pinout_row_claims(
            registry,
            rows,
            created_at=captured_at,
            splits=("train",),
        )
    )

    assert {claim.field for claim in gt_claims} == {
        "clock_specs.max_freq_mhz",
        "overview.description",
        "overview.manufacturer",
        "overview.part_number",
    }
    assert len(pin_claims) == 5
    assert {claim.field for claim in pin_claims} == {
        "pin.number",
        "pin.name",
        "pin.type",
        "pin.functions",
        "pin.direction",
    }
    assert all(claim.entity.package == "LQFP2" for claim in pin_claims)
    assert all(claim.conditions["source_split"] == "train" for claim in pin_claims)
    assert all(claim.evidence[0].page_1based == 5 for claim in pin_claims)

    queue = build_extraction_work_queue(
        registry,
        aligned_document_sha256={document_sha},
    )
    assert queue["work"][0]["routes"]["pin_or_ball"] == "owned_exact_vision_pair"
    assert queue["work"][0]["frontier_batch_eligibility"]["eligible"] is False
