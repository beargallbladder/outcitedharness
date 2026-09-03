from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from harness.electronics.claims import canonical_json
from harness.electronics.corpus import verify_corpus_registry
from harness.electronics.incremental_corpus import (
    build_incremental_corpus_registry,
)


def _pdf(path: Path, value: bytes = b"part") -> str:
    path.write_bytes(b"%PDF-1.7\n" + value + b"\n" + b"x" * 128 + b"\n%%EOF\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(path: Path, pdf: Path, digest: str) -> None:
    core = {
        "schema": "harness.electronics-incremental-source-snapshot.v1",
        "purpose": "stable_deduplicated_datasheet_intake",
        "cohort_id": "wave-1",
        "counts": {"documents": 1},
        "documents": [
            {
                "observation_id": "source-a",
                "source_path": str(pdf),
                "byte_size": pdf.stat().st_size,
                "mtime_ns": pdf.stat().st_mtime_ns,
                "sha256": digest,
            }
        ],
    }
    value = {
        "created_at": "2026-09-02T00:00:00+00:00",
        **core,
        "evidence_sha256": hashlib.sha256(canonical_json(core)).hexdigest(),
    }
    path.write_text(json.dumps(value))


def test_incremental_registry_verifies_sources_and_exact_stem_ground_truth(
    tmp_path: Path,
) -> None:
    pdf_root = tmp_path / "pdfs"
    gt_root = tmp_path / "gt"
    validated_root = tmp_path / "validated"
    for root in (pdf_root, gt_root, validated_root):
        root.mkdir()
    pdf = pdf_root / "ti_tms320.pdf"
    digest = _pdf(pdf)
    (gt_root / "ti_tms320.json").write_text(
        json.dumps(
            {
                "overview": {
                    "part_number": "TMS320",
                    "manufacturer": "Texas Instruments",
                },
                "pinout": {"pin_functions_summary": []},
            }
        )
    )
    snapshot = tmp_path / "snapshot.json"
    _snapshot(snapshot, pdf, digest)

    registry = build_incremental_corpus_registry(
        snapshot,
        pdf_root=pdf_root,
        ground_truth_root=gt_root,
        validated_root=validated_root,
    )
    verify_corpus_registry(registry)

    assert registry["counts"]["unique_pdf_sha256"] == 1
    assert registry["counts"]["documents_with_ground_truth"] == 1
    document = registry["documents"][0]
    assert document["paths"] == ["ti_tms320.pdf"]
    assert document["vendors"] == ["ti"]
    assert document["ground_truth"][0]["part_numbers"] == ["TMS320"]


def test_incremental_registry_rejects_source_changed_after_snapshot(
    tmp_path: Path,
) -> None:
    pdf_root = tmp_path / "pdfs"
    gt_root = tmp_path / "gt"
    validated_root = tmp_path / "validated"
    for root in (pdf_root, gt_root, validated_root):
        root.mkdir()
    pdf = pdf_root / "part.pdf"
    digest = _pdf(pdf)
    snapshot = tmp_path / "snapshot.json"
    _snapshot(snapshot, pdf, digest)
    _pdf(pdf, b"changed")

    with pytest.raises(ValueError, match="changed after intake"):
        build_incremental_corpus_registry(
            snapshot,
            pdf_root=pdf_root,
            ground_truth_root=gt_root,
            validated_root=validated_root,
        )
