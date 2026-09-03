from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from harness.electronics.corpus import CorpusInputs, build_corpus_registry
from harness.electronics.holdout import freeze_factory_holdout


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_holdout_excludes_existing_row_training_lineages_and_families(
    tmp_path: Path,
):
    pdfs = tmp_path / "pdfs"
    gt = tmp_path / "gt"
    validated = tmp_path / "validated"
    rows = tmp_path / "rows"
    for path in (pdfs, gt, validated, rows / "canonical"):
        path.mkdir(parents=True, exist_ok=True)
    for index in range(1, 4):
        pdf = pdfs / f"acme_atom{index}.pdf"
        pdf.write_bytes(f"%PDF-{index}".encode())
        (gt / f"acme_atom{index}.json").write_text(
            json.dumps(
                {
                    "overview": {
                        "manufacturer": "acme",
                        "part_number": f"ATOM{index}",
                    },
                    "pinout": {"packages": [f"LQFP{index * 16}"]},
                }
            ),
            encoding="utf-8",
        )
    registry = build_corpus_registry(
        CorpusInputs(
            pdf_root=pdfs,
            ground_truth_root=gt,
            validated_root=validated,
        ),
        hash_workers=1,
    )
    train_row = {
        "record_id": "acme_atom1",
        "provenance": {"pdf_sha256": _sha(pdfs / "acme_atom1.pdf")},
    }
    train_path = rows / "canonical" / "train.jsonl"
    train_path.write_text(json.dumps(train_row) + "\n", encoding="utf-8")
    (rows / "manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "canonical/train.jsonl": {
                        "sha256": _sha(train_path),
                        "bytes": train_path.stat().st_size,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    holdout = freeze_factory_holdout(
        registry,
        row_dataset_root=rows,
        page_index_root=None,
        fraction=0.2,
        minimum_documents=1,
        maximum_documents=2,
        temporal_cutoff=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )

    reserved = {
        row["document_sha256"] for row in holdout["reserved_documents"]
    }
    assert _sha(pdfs / "acme_atom1.pdf") not in reserved
    assert holdout["counts"]["excluded"]["row_train_lineage"] == 1
    assert holdout["policy"]["future_training_use"] == "prohibited"
