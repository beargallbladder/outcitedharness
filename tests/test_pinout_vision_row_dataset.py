from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_pinout_vision_row_dataset.py"
SPEC = importlib.util.spec_from_file_location("pinout_row_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _write_evidence(path: Path, value: dict) -> dict:
    value["evidence_sha256"] = hashlib.sha256(
        builder._canonical(value)
    ).hexdigest()
    path.write_text(
        json.dumps({"created_at": "2026-09-01T00:00:00Z", **value})
    )
    return json.loads(path.read_text())


def test_split_lineages_is_deterministic_and_exclusive() -> None:
    weights = {f"{index:064x}": index + 1 for index in range(12)}

    first = builder._split_lineages(weights)
    second = builder._split_lineages(weights)

    assert first == second
    assert set(first) == set(weights)
    assert set(first.values()) == {"train", "validation", "test"}


def test_authorization_must_be_scope_and_evidence_bound(tmp_path: Path) -> None:
    evidence = tmp_path / "answer.md"
    evidence.write_text("ground truth is yours\n")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": builder.AUTHORIZATION_SCHEMA,
                "training_authorized": True,
                "scope": {
                    "dataset_kind": "frontier-validated-pinout-row-crops",
                    "model": "Qwen3-VL-8B-Instruct",
                    "method": "offline-lora-sft",
                },
                "constraints": {
                    "required_validation": "VALIDATED_GROUND_TRUTH",
                    "require_exact_rendered-page_alignment": True,
                    "network_during_training": "none",
                },
                "basis": [
                    {
                        "kind": "corpus-owner-response",
                        "path": str(evidence),
                        "sha256": builder._sha256(evidence),
                    }
                ],
            }
        )
    )

    result = builder._authorization(receipt)

    assert result["training_authorized"] is True
    evidence.write_text("changed\n")
    try:
        builder._authorization(receipt)
    except ValueError as error:
        assert "changed or is missing" in str(error)
    else:
        raise AssertionError("changed authorization evidence was accepted")


def test_build_dataset_renders_hash_bound_row_crops(tmp_path: Path) -> None:
    import pymupdf as fitz

    pdf_root = tmp_path / "pdf"
    truth_root = tmp_path / "truth"
    validated_root = tmp_path / "validated"
    for directory in (pdf_root, truth_root, validated_root):
        directory.mkdir()
    pdf_path = pdf_root / "part-1.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    page.insert_text((20, 30), "LQFP8 | Pin Name")
    for index in range(1, 9):
        page.insert_text((20, 50 + index * 12), f"{index} P{index} GPIO")
    document.save(pdf_path)
    document.close()

    pins = [
        {
            "pin_no": index,
            "name": f"P{index}",
            "type": "gpio",
            "functions": [f"P{index} GPIO"],
            "dir": "I/O",
        }
        for index in range(1, 9)
    ]
    truth_path = truth_root / "part-1.json"
    truth_path.write_text(
        json.dumps({"pinout": {"pin_functions_summary": pins}})
    )
    pdf_sha256 = builder._sha256(pdf_path)
    truth_sha256 = builder._sha256(truth_path)

    source_path = tmp_path / "source.json"
    source_core = {
        "schema": builder.SOURCE_SCHEMA,
        "policy": {"training_authorized": False},
        "sources": {
            "validated_root": str(validated_root),
            "ground_truth_root": str(truth_root),
            "pdf_root": str(pdf_root),
        },
        "counts": {},
        "candidates": [
            {
                "record_id": "part-1",
                "published_path": "part-1.json",
                "published_sha256": "a" * 64,
                "ground_truth_path": truth_path.name,
                "ground_truth_sha256": truth_sha256,
                "pdf_path": pdf_path.name,
                "pdf_sha256": pdf_sha256,
                "frontier_batch_id": "batch-1",
            }
        ],
        "rejections": [],
    }
    source = _write_evidence(source_path, source_core)

    alignment_path = tmp_path / "alignment.json"
    alignment_core = {
        "schema": builder.ALIGNMENT_SCHEMA,
        "source_audit": {
            "path": str(source_path),
            "sha256": builder._sha256(source_path),
            "evidence_sha256": source["evidence_sha256"],
        },
        "policy": {
            "limited_probe": False,
            "training_authorized": True,
        },
        "counts": {"row_crop_examples": 1},
        "elapsed_seconds": 0.1,
        "records": [
            {
                "record_id": "part-1",
                "pdf_sha256": pdf_sha256,
                "row_crop_status": "eligible",
                "row_crop_chunks": [
                    {
                        "page_1based": 1,
                        "table_index": 0,
                        "package_candidate": "LQFP8",
                        "package_header": "LQFP8",
                        "header_bbox": [15, 15, 285, 38],
                        "body_bbox": [15, 45, 285, 155],
                        "source_rows": list(range(2, 10)),
                        "target_indices": list(range(8)),
                    }
                ],
            }
        ],
    }
    _write_evidence(alignment_path, alignment_core)
    destination = tmp_path / "dataset"

    manifest = builder.build_dataset(
        alignment_audit_path=alignment_path,
        destination=destination,
        dpi=144,
    )

    assert manifest["authorization"]["training_authorized"] is False
    assert manifest["counts"]["examples"] == {
        "train": 1,
        "validation": 0,
        "test": 0,
    }
    assert manifest["counts"]["unique_images"] == 2
    canonical = json.loads(
        (destination / "canonical" / "train.jsonl").read_text()
    )
    assert json.loads(canonical["response"]) == {"pins": pins}
    for image in canonical["images"]:
        assert (destination / image).is_file()

    for path in sorted(destination.rglob("*")):
        if path.is_dir():
            os.chmod(path, 0o755)
    os.chmod(destination, 0o755)
