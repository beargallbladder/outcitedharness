from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from harness.electronics.corpus import (
    AssetSource,
    CorpusInputs,
    build_corpus_registry,
    canonical_json,
    verify_corpus_registry,
    write_new_registry,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_registry_joins_without_collapsing_duplicate_documents(tmp_path: Path):
    pdfs = tmp_path / "pdfs"
    gt = tmp_path / "gt"
    validated = tmp_path / "validated"
    pdfs.mkdir()
    gt.mkdir()
    validated.mkdir()
    (pdfs / "acme_atom1.pdf").write_bytes(b"%PDF-atom-one")
    (pdfs / "copy_atom1.pdf").write_bytes(b"%PDF-atom-one")
    (pdfs / "acme_atom2.pdf").write_bytes(b"%PDF-atom-two")
    _write_json(
        gt / "acme_atom1.json",
        {
            "overview": {
                "manufacturer": "acme",
                "part_number": "ATOM1",
            },
            "pinout": [{"pin_no": 1, "name": "PA0"}],
            "clock_specs": [{"max_mhz": 100}],
        },
    )
    _write_json(
        validated / "acme" / "acme_atom1.json",
        {
            "overview": {
                "manufacturer": "acme",
                "part_number": "ATOM1",
            },
            "pinout": [{"pin_no": 1, "name": "PA0"}],
            "_meta": {"provenance": {"validation": "VALIDATED_GROUND_TRUTH"}},
        },
    )
    atom2_sha = hashlib.sha256(b"%PDF-atom-two").hexdigest()
    asset = tmp_path / "asset.jsonl"
    asset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_id": "direct-1",
                        "pdf_sha256": atom2_sha,
                    }
                ),
                json.dumps(
                    {
                        "record_id": "part-1",
                        "vendor": "acme.com",
                        "part_number": "ATOM1",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    registry = build_corpus_registry(
        CorpusInputs(
            pdf_root=pdfs,
            ground_truth_root=gt,
            validated_root=validated,
            assets=(AssetSource("facts", asset),),
            expected_pdf_files=3,
        ),
        hash_workers=2,
    )

    verify_corpus_registry(registry)
    assert registry["counts"]["pdf_files"] == 3
    assert registry["counts"]["unique_pdf_sha256"] == 2
    assert registry["counts"]["duplicate_pdf_files"] == 1
    assert registry["counts"]["documents_with_ground_truth"] == 1
    assert registry["counts"]["ground_truth_sections"] == {
        "clock_specs": 1,
        "overview": 1,
        "pinout": 1,
    }
    assert registry["assets"][0]["joins"] == {
        "document_sha256": 1,
        "unambiguous_part": 1,
        "ambiguous": 0,
        "unresolved": 0,
    }
    atom1 = next(
        row for row in registry["documents"] if "acme_atom1" in row["stems"]
    )
    assert atom1["paths"] == ["acme_atom1.pdf", "copy_atom1.pdf"]
    assert atom1["asset_memberships"] == {
        "facts": ["part_number:ATOM1", "record_id:part-1"]
    }


def test_registry_digest_detects_tampering(tmp_path: Path):
    value = {
        "created_at": "2026-09-01T00:00:00+00:00",
        "schema": "harness.electronics-corpus-registry.v1",
        "policy": {},
        "sources": {},
        "counts": {"unique_pdf_sha256": 0},
        "assets": [],
        "documents": [],
        "orphans": {"ground_truth": [], "published_pinouts": []},
    }
    core = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "evidence_sha256"}
    }
    value["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
    verify_corpus_registry(value)
    value["counts"]["unique_pdf_sha256"] = 1
    with pytest.raises(ValueError, match="evidence digest"):
        verify_corpus_registry(value)


def test_registry_output_is_exclusive_and_read_only(tmp_path: Path):
    value = {
        "created_at": "2026-09-01T00:00:00+00:00",
        "schema": "harness.electronics-corpus-registry.v1",
        "policy": {},
        "sources": {},
        "counts": {"unique_pdf_sha256": 0},
        "assets": [],
        "documents": [],
        "orphans": {"ground_truth": [], "published_pinouts": []},
    }
    core = {key: item for key, item in value.items() if key != "created_at"}
    value["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
    output = tmp_path / "registry.json"

    write_new_registry(output, value)

    assert output.stat().st_mode & 0o777 == 0o444
    with pytest.raises(ValueError, match="already exists"):
        write_new_registry(output, value)
