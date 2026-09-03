#!/usr/bin/env python3
"""Verify exact PDF-page and package-column alignment for pinout vision data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.electronics.vision_alignment import (
    align_record,
    definition_pages,
    extract_page_tables,
)


SCHEMA = "harness.pinout-vision-alignment-audit.v1"
SOURCE_SCHEMA = "harness.pinout-vision-source-audit.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input must be a regular file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"input must contain a JSON object: {path}")
    return value


def _pin_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    pinout = value.get("pinout")
    rows = pinout.get("pin_functions_summary") if isinstance(pinout, dict) else None
    if not isinstance(rows, list):
        raise ValueError("frontier target has no pin_functions_summary")
    return rows


def _legacy_hints(path: Path | None) -> dict[str, set[int]]:
    output: dict[str, set[int]] = defaultdict(set)
    if path is None:
        return output
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"legacy vision pairs must be a regular file: {path}")
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("part"), str):
            raise ValueError(f"legacy vision row {line_number} is malformed")
        for image in row.get("images") or []:
            match = re.search(r"/page_(\d+)\.[A-Za-z0-9]+$", str(image))
            if match:
                output[row["part"]].add(int(match.group(1)))
    return output


def _metadata_hints(value: dict[str, Any]) -> set[int]:
    output: set[int] = set()
    sources = (value, value.get("_meta"), value.get("pinout"))
    for source in sources:
        if not isinstance(source, dict):
            continue
        raw = source.get("pin_table_pages")
        if isinstance(raw, int):
            raw = [raw]
        if isinstance(raw, list):
            for page in raw:
                if isinstance(page, int) and page > 0:
                    output.add(page)
    return output


def audit_alignment(
    *,
    source_audit_path: Path,
    legacy_vision_pairs: Path | None = None,
    minimum_examples: int = 1101,
    minimum_coverage: float = 0.9,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    source = _json_object(source_audit_path)
    if source.get("schema") != SOURCE_SCHEMA:
        raise ValueError("source audit schema is not supported")
    expected_source_evidence = source.get("evidence_sha256")
    source_core = {
        key: value
        for key, value in source.items()
        if key not in ("created_at", "evidence_sha256")
    }
    if hashlib.sha256(_canonical(source_core)).hexdigest() != expected_source_evidence:
        raise ValueError("source audit evidence digest is invalid")
    source_candidates = source.get("candidates")
    if not isinstance(source_candidates, list):
        raise ValueError("source audit candidates are missing")
    if offset < 0:
        raise ValueError("offset cannot be negative")
    source_candidates = source_candidates[offset:]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        source_candidates = source_candidates[:limit]

    source_paths = source["sources"]
    validated_root = Path(source_paths["validated_root"]).resolve(strict=True)
    ground_truth_root = Path(source_paths["ground_truth_root"]).resolve(strict=True)
    pdf_root = Path(source_paths["pdf_root"]).resolve(strict=True)
    legacy = _legacy_hints(legacy_vision_pairs)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in source_candidates:
        grouped[str(candidate["pdf_sha256"])].append(candidate)

    import pymupdf as fitz

    records: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    documents_processed = 0
    started = time.perf_counter()
    for pdf_sha256, candidates in sorted(grouped.items()):
        representative = candidates[0]
        pdf_path = pdf_root / representative["pdf_path"]
        if _sha256(pdf_path) != pdf_sha256:
            raise ValueError(f"PDF changed after source audit: {pdf_path}")
        document = fitz.open(pdf_path)
        try:
            hints: set[int] = set()
            targets: dict[str, list[dict[str, Any]]] = {}
            for candidate in candidates:
                record_id = str(candidate["record_id"])
                ground_truth_path = ground_truth_root / candidate["ground_truth_path"]
                if _sha256(ground_truth_path) != candidate["ground_truth_sha256"]:
                    raise ValueError(
                        f"ground truth changed after source audit: {ground_truth_path}"
                    )
                published_path = validated_root / candidate["published_path"]
                published = _json_object(published_path)
                targets[record_id] = _pin_rows(
                    _json_object(ground_truth_path)
                )
                hints.update(_metadata_hints(published))
                hints.update(legacy.get(record_id, set()))

            pages = definition_pages(document, hinted_pages_1based=hints)
            tables = extract_page_tables(document, pages)
            for candidate in candidates:
                record_id = str(candidate["record_id"])
                try:
                    result = align_record(
                        tables_by_page=tables,
                        target_rows=targets[record_id],
                        package_candidates=candidate["package_candidates"],
                        minimum_coverage=minimum_coverage,
                    )
                except ValueError as error:
                    result = {
                        "status": "withhold",
                        "reason": f"target_invalid:{str(error)}",
                        "target_rows": len(targets[record_id]),
                        "matched_rows": 0,
                        "coverage": 0.0,
                        "package_candidates_scored": {},
                        "row_crop_status": "withhold",
                        "row_crop_examples": 0,
                        "row_crop_target_rows": 0,
                        "row_crop_chunks": [],
                        "tables": [],
                    }
                reasons[result["reason"]] += 1
                records.append(
                    {
                        "record_id": record_id,
                        "pdf_sha256": pdf_sha256,
                        "candidate_pages_1based": list(pages),
                        "pages_with_extracted_tables": sorted(tables),
                        **result,
                    }
                )
        finally:
            document.close()
        documents_processed += 1
        if documents_processed % 25 == 0:
            print(
                f"aligned {documents_processed}/{len(grouped)} unique PDFs "
                f"({len(records)} records)",
                flush=True,
            )

    aligned = [row for row in records if row["status"] == "aligned"]
    aligned_lineages = {row["pdf_sha256"] for row in aligned}
    row_crop_records = [
        row for row in records if row["row_crop_status"] == "eligible"
    ]
    row_crop_examples = sum(row["row_crop_examples"] for row in row_crop_records)
    row_crop_lineages = {row["pdf_sha256"] for row in row_crop_records}
    core: dict[str, Any] = {
        "schema": SCHEMA,
        "source_audit": {
            "path": str(source_audit_path.resolve(strict=True)),
            "sha256": _sha256(source_audit_path),
            "evidence_sha256": expected_source_evidence,
        },
        "policy": {
            "minimum_exact_row_coverage": minimum_coverage,
            "minimum_row_crop_examples": minimum_examples,
            "training_authorized": row_crop_examples >= minimum_examples,
            "limited_probe": limit is not None or offset != 0,
            "source_record_offset": offset,
        },
        "counts": {
            "source_records_examined": len(source_candidates),
            "unique_pdfs_examined": len(grouped),
            "aligned_records": len(aligned),
            "aligned_unique_pdf_sha256": len(aligned_lineages),
            "row_crop_eligible_records": len(row_crop_records),
            "row_crop_examples": row_crop_examples,
            "row_crop_target_rows": sum(
                row["row_crop_target_rows"] for row in row_crop_records
            ),
            "row_crop_unique_pdf_sha256": len(row_crop_lineages),
            "withheld_records": len(records) - len(aligned),
            "reasons": dict(sorted(reasons.items())),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "records": sorted(records, key=lambda row: row["record_id"]),
    }
    core["evidence_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **core,
    }


def write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", required=True, type=Path)
    parser.add_argument("--legacy-vision-pairs", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-examples", type=int, default=1101)
    parser.add_argument("--minimum-coverage", type=float, default=0.9)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.minimum_examples < 1:
        raise ValueError("minimum examples must be positive")
    if not 0.5 <= arguments.minimum_coverage <= 1:
        raise ValueError("minimum coverage must be between 0.5 and 1")
    result = audit_alignment(
        source_audit_path=arguments.source_audit,
        legacy_vision_pairs=arguments.legacy_vision_pairs,
        minimum_examples=arguments.minimum_examples,
        minimum_coverage=arguments.minimum_coverage,
        offset=arguments.offset,
        limit=arguments.limit,
    )
    write_new(arguments.output, result)
    print(json.dumps(result["counts"], sort_keys=True))
    if result["policy"]["training_authorized"]:
        print("ALIGNMENT_GATE_PASSED")
        return 0
    print("ALIGNMENT_GATE_BLOCKED")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
