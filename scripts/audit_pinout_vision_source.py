#!/usr/bin/env python3
"""Seal a pre-training inventory of frontier-validated MCU pinout sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "harness.pinout-vision-source-audit.v1"
VALIDATION = "VALIDATED_GROUND_TRUTH"
GROUND_TRUTH_SOURCE = "claude-sonnet-5-batch"


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


def _regular_file(path: Path, kind: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{kind} must be a regular file: {path}")
    return path


def _json_object(path: Path, kind: str) -> dict[str, Any]:
    value = json.loads(_regular_file(path, kind).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{kind} must contain a JSON object: {path}")
    return value


def _pin_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    pinout = value.get("pinout")
    rows = pinout.get("pin_functions_summary") if isinstance(pinout, dict) else None
    if not isinstance(rows, list):
        raise ValueError("pinout.pin_functions_summary is missing")
    output: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("pin row is not an object")
        raw_number = row.get("pin_no")
        raw_name = row.get("name")
        if (
            isinstance(raw_number, bool)
            or not isinstance(raw_number, (str, int))
            or not isinstance(raw_name, str)
            or not str(raw_number).strip()
            or not raw_name.strip()
        ):
            raise ValueError("pin row has an invalid physical identity")
        number = re.sub(r"\s+", "", str(raw_number).upper())
        number = re.sub(r"\(\d+\)$", "", number)
        name = re.sub(r"[^A-Z0-9]+", "", raw_name.upper())
        if not number or not name:
            raise ValueError("pin row normalizes to an empty physical identity")
        identity = (number, name)
        if identity in identities:
            raise ValueError("pin rows contain a duplicate physical identity")
        identities.add(identity)
        output.append(row)
    if len(output) < 8:
        raise ValueError("pinout contains fewer than eight physical rows")
    return output


def _packages(value: dict[str, Any]) -> list[str]:
    sources = (
        value,
        value.get("pinout"),
        value.get("_meta"),
        (value.get("_meta") or {}).get("provenance")
        if isinstance(value.get("_meta"), dict)
        else None,
    )
    packages: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("package", "packages", "requested_package", "selected_package"):
            raw = source.get(key)
            values = raw if isinstance(raw, list) else [raw]
            for item in values:
                package = re.sub(r"\s+", " ", str(item or "")).strip()
                if package and package not in packages:
                    packages.append(package)
    return packages


def _audit_maps(
    quality_path: Path,
    provenance_path: Path,
) -> tuple[dict[str, Any], set[str]]:
    quality = _json_object(quality_path, "quality audit")
    provenance = _json_object(provenance_path, "provenance audit")
    raw_suspect = provenance.get("suspect")
    if not isinstance(raw_suspect, list):
        raise ValueError("provenance audit has no suspect list")
    suspect: set[str] = set()
    for row in raw_suspect:
        if not isinstance(row, dict) or not isinstance(row.get("part"), str):
            raise ValueError("provenance audit has an invalid suspect record")
        suspect.add(row["part"])
    return quality, suspect


def audit_source(
    *,
    validated_root: Path,
    ground_truth_root: Path,
    pdf_root: Path,
    quality_audit: Path,
    provenance_audit: Path,
    minimum_records: int,
) -> dict[str, Any]:
    roots = {
        "validated": validated_root.resolve(strict=True),
        "ground_truth": ground_truth_root.resolve(strict=True),
        "pdf": pdf_root.resolve(strict=True),
    }
    if any(not path.is_dir() for path in roots.values()):
        raise ValueError("all corpus roots must be directories")
    quality, suspect = _audit_maps(quality_audit, provenance_audit)
    candidates: list[dict[str, Any]] = []
    rejections: list[dict[str, str]] = []
    reasons: Counter[str] = Counter()
    validation_counts: Counter[str] = Counter()

    def reject(path: Path, reason: str) -> None:
        reasons[reason] += 1
        rejections.append(
            {
                "record": path.relative_to(roots["validated"]).as_posix(),
                "reason": reason,
            }
        )

    for published_path in sorted(roots["validated"].rglob("*.json")):
        relative = published_path.relative_to(roots["validated"])
        try:
            published = _json_object(published_path, "published pinout")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            reject(published_path, f"published_json_invalid:{type(error).__name__}")
            continue
        metadata = published.get("_meta")
        provenance = metadata.get("provenance") if isinstance(metadata, dict) else None
        validation = (
            str(provenance.get("validation") or "")
            if isinstance(provenance, dict)
            else ""
        )
        validation_counts[validation or "missing"] += 1
        if validation != VALIDATION:
            reject(published_path, f"validation_not_eligible:{validation or 'missing'}")
            continue
        if provenance.get("ground_truth_source") != GROUND_TRUTH_SOURCE:
            reject(published_path, "frontier_source_not_eligible")
            continue
        if not isinstance(provenance.get("published_at"), str):
            reject(published_path, "published_at_missing")
            continue

        part = published_path.stem
        ground_truth_path = roots["ground_truth"] / f"{part}.json"
        pdf_path = roots["pdf"] / f"{part}.pdf"
        try:
            ground_truth = _json_object(ground_truth_path, "frontier ground truth")
            _regular_file(pdf_path, "source PDF")
            with pdf_path.open("rb") as handle:
                if handle.read(5) != b"%PDF-":
                    raise ValueError("source PDF signature is invalid")
            published_pins = _pin_rows(published)
            ground_truth_pins = _pin_rows(ground_truth)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            reject(published_path, f"source_invalid:{str(error)}")
            continue
        if published_pins != ground_truth_pins:
            reject(published_path, "published_and_frontier_pin_rows_differ")
            continue
        if part in suspect:
            reject(published_path, "provenance_audit_suspect")
            continue
        quality_row = quality.get(part)
        if isinstance(quality_row, dict) and quality_row.get("ok") is not True:
            reject(published_path, "quality_audit_failed")
            continue
        published_batch = metadata.get("batch_id") if isinstance(metadata, dict) else None
        frontier_meta = ground_truth.get("_meta")
        frontier_batch = (
            frontier_meta.get("batch_id") if isinstance(frontier_meta, dict) else None
        )
        if (
            not isinstance(published_batch, str)
            or not published_batch
            or published_batch != frontier_batch
        ):
            reject(published_path, "frontier_batch_identity_mismatch")
            continue
        custom_id = metadata.get("custom_id") if isinstance(metadata, dict) else None
        if custom_id != part:
            reject(published_path, "frontier_custom_id_mismatch")
            continue
        packages = _packages(published)
        if not packages:
            reject(published_path, "package_candidates_missing")
            continue

        candidates.append(
            {
                "record_id": part,
                "vendor": relative.parts[0] if len(relative.parts) > 1 else "unknown",
                "published_path": relative.as_posix(),
                "published_sha256": _sha256(published_path),
                "ground_truth_path": ground_truth_path.name,
                "ground_truth_sha256": _sha256(ground_truth_path),
                "pdf_path": pdf_path.name,
                "pdf_sha256": _sha256(pdf_path),
                "pin_rows": len(published_pins),
                "package_candidates": packages,
                "frontier_batch_id": published_batch,
                "published_at": provenance["published_at"],
                "quality_audit": (
                    "pass"
                    if isinstance(quality_row, dict) and quality_row.get("ok") is True
                    else "not_present"
                ),
                "alignment_status": "pending_exact_page_and_package_column",
            }
        )

    pdf_lineages = {row["pdf_sha256"] for row in candidates}
    core: dict[str, Any] = {
        "schema": SCHEMA,
        "policy": {
            "required_validation": VALIDATION,
            "required_ground_truth_source": GROUND_TRUTH_SOURCE,
            "minimum_training_records_after_alignment": minimum_records,
            "training_authorized": False,
            "next_gate": "exact_page_and_package_column_alignment",
        },
        "sources": {
            "validated_root": str(roots["validated"]),
            "ground_truth_root": str(roots["ground_truth"]),
            "pdf_root": str(roots["pdf"]),
            "quality_audit": {
                "path": str(quality_audit.resolve(strict=True)),
                "sha256": _sha256(quality_audit),
            },
            "provenance_audit": {
                "path": str(provenance_audit.resolve(strict=True)),
                "sha256": _sha256(provenance_audit),
            },
        },
        "counts": {
            "published_json": sum(validation_counts.values()),
            "validation_status": dict(sorted(validation_counts.items())),
            "prealignment_candidates": len(candidates),
            "unique_pdf_sha256": len(pdf_lineages),
            "rejected": len(rejections),
            "rejection_reasons": dict(sorted(reasons.items())),
        },
        "candidates": candidates,
        "rejections": rejections,
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
    parser.add_argument("--validated-root", required=True, type=Path)
    parser.add_argument("--ground-truth-root", required=True, type=Path)
    parser.add_argument("--pdf-root", required=True, type=Path)
    parser.add_argument("--quality-audit", required=True, type=Path)
    parser.add_argument("--provenance-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-records", type=int, default=1101)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.minimum_records < 1:
        raise ValueError("minimum records must be positive")
    result = audit_source(
        validated_root=arguments.validated_root,
        ground_truth_root=arguments.ground_truth_root,
        pdf_root=arguments.pdf_root,
        quality_audit=arguments.quality_audit,
        provenance_audit=arguments.provenance_audit,
        minimum_records=arguments.minimum_records,
    )
    write_new(arguments.output, result)
    print(json.dumps(result["counts"], sort_keys=True))
    if result["counts"]["prealignment_candidates"] < arguments.minimum_records:
        print("SOURCE_GATE_BLOCKED")
        return 2
    print("SOURCE_GATE_READY_FOR_ALIGNMENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
