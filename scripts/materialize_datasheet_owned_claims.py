#!/usr/bin/env python3
"""Materialize existing owned datasheet evidence into immutable claims."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from harness.electronics.claims import (
    canonical_json,
    seal_claim_bundle,
    verify_claim_bundle,
)
from harness.electronics.corpus import sha256_file, verify_corpus_registry
from harness.electronics.extraction import (
    build_extraction_work_queue,
    iter_ground_truth_claims,
    iter_pinout_row_claims,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-registry", type=Path, required=True)
    parser.add_argument("--row-dataset", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--output-work-queue", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "validation", "test"),
        dest="splits",
        help="Row-dataset split to import; defaults to all splits.",
    )
    return parser


def _aligned_lineages(root: Path, splits: tuple[str, ...]) -> set[str]:
    output: set[str] = set()
    for split in splits:
        path = root / "canonical" / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                value = json.loads(line)
                try:
                    lineage = str(value["provenance"]["pdf_sha256"])
                except (KeyError, TypeError) as exc:
                    raise ValueError(
                        f"{path}: missing lineage at line {line_number}"
                    ) from exc
                output.add(lineage)
    return output


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"immutable output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = _parser().parse_args()
    registry_path = args.corpus_registry.expanduser().resolve(strict=True)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    verify_corpus_registry(registry)
    row_root = args.row_dataset.expanduser().resolve(strict=True)
    splits = tuple(args.splits or ("train", "validation", "test"))
    if args.output_bundle.exists() or args.output_work_queue.exists():
        raise ValueError("immutable output path already exists")
    created_at = datetime.now(timezone.utc)
    lineages = _aligned_lineages(row_root, splits)
    work_queue = build_extraction_work_queue(
        registry,
        aligned_document_sha256=lineages,
    )
    work_queue["created_at"] = created_at.isoformat()
    # Bind creation time after the deterministic evidence digest: timestamps do
    # not alter the queue's content identity.
    claims = itertools.chain(
        iter_ground_truth_claims(registry, created_at=created_at),
        iter_pinout_row_claims(
            registry,
            row_root,
            created_at=created_at,
            splits=splits,
        ),
    )
    manifest = seal_claim_bundle(
        args.output_bundle,
        claims,
        source_receipts={
            "corpus_registry": {
                "path": str(registry_path),
                "sha256": sha256_file(registry_path),
                "evidence_sha256": registry["evidence_sha256"],
            },
            "row_dataset_manifest": {
                "path": str((row_root / "manifest.json").resolve()),
                "sha256": sha256_file(row_root / "manifest.json"),
            },
            "row_splits": list(splits),
        },
        created_at=created_at,
    )
    verify_claim_bundle(args.output_bundle)
    _write_new_json(args.output_work_queue, work_queue)
    print(
        json.dumps(
            {
                "status": "sealed",
                "claim_bundle": str(args.output_bundle.resolve()),
                "claim_bundle_evidence_sha256": manifest["evidence_sha256"],
                "claims": manifest["counts"]["claims"],
                "work_queue": str(args.output_work_queue.resolve()),
                "work_queue_evidence_sha256": work_queue["evidence_sha256"],
                "aligned_pdf_lineages": len(lineages),
                "corpus_evidence_sha256": registry["evidence_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
