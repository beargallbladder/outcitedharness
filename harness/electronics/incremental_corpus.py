"""Build a sealed corpus view from newly stabilized datasheet downloads."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from harness.electronics.claims import canonical_json
from harness.electronics.corpus import (
    ACTIVE_GROUND_TRUTH_SECTIONS,
    canonical_vendor,
    sha256_file,
)


def _verify_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve(strict=True)
    if path.expanduser().is_symlink() or not resolved.is_file():
        raise ValueError("source snapshot must be a regular non-symlink file")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        value.get("schema")
        != "harness.electronics-incremental-source-snapshot.v1"
    ):
        raise ValueError("unsupported incremental source snapshot")
    core = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "evidence_sha256"}
    }
    if hashlib.sha256(canonical_json(core)).hexdigest() != value.get(
        "evidence_sha256"
    ):
        raise ValueError("incremental source snapshot evidence is invalid")
    documents = value.get("documents")
    if (
        not isinstance(documents, list)
        or value.get("counts", {}).get("documents") != len(documents)
    ):
        raise ValueError("incremental source snapshot count is invalid")
    return value, sha256_file(resolved)


def _safe_files(root: Path, pattern: str, *, recursive: bool) -> list[Path]:
    resolved = root.expanduser().resolve(strict=True)
    if root.expanduser().is_symlink() or not resolved.is_dir():
        raise ValueError(f"corpus sidecar root must be a real directory: {root}")
    iterator = resolved.rglob(pattern) if recursive else resolved.glob(pattern)
    return sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and not path.is_symlink()
    )


def _part_key(value: Any) -> str | None:
    text = str("" if value is None else value).strip().upper()
    normalized = "".join(character for character in text if character.isalnum())
    return normalized or None


def _summary(path: Path, root: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"corpus sidecar JSON is not an object: {path}")
    metadata = value.get("_meta")
    provenance = (
        metadata.get("provenance") if isinstance(metadata, Mapping) else None
    )
    overview = value.get("overview")
    part_numbers = {
        part
        for candidate in (
            value.get("part_number"),
            value.get("base_mpn"),
            (
                overview.get("part_number")
                if isinstance(overview, Mapping)
                else None
            ),
        )
        if (part := _part_key(candidate)) is not None
    }
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "record_id": path.stem,
        "part_numbers": sorted(part_numbers),
        "vendor": (
            canonical_vendor(overview.get("manufacturer"))
            if isinstance(overview, Mapping)
            else None
        ),
        "sections": [
            section
            for section in ACTIVE_GROUND_TRUTH_SECTIONS
            if section in value
        ],
        "batch_id": (
            metadata.get("batch_id")
            if isinstance(metadata, Mapping)
            else None
        ),
        "validation": (
            provenance.get("validation")
            if isinstance(provenance, Mapping)
            else (
                metadata.get("validation")
                if isinstance(metadata, Mapping)
                else None
            )
        ),
    }


def _vendor_from_stem(stem: str) -> str | None:
    return canonical_vendor(stem.split("_", 1)[0])


def build_incremental_corpus_registry(
    source_snapshot: Path,
    *,
    pdf_root: Path,
    ground_truth_root: Path,
    validated_root: Path,
) -> dict[str, Any]:
    snapshot, snapshot_sha = _verify_snapshot(source_snapshot)
    pdf = pdf_root.expanduser().resolve(strict=True)
    ground_truth = ground_truth_root.expanduser().resolve(strict=True)
    validated = validated_root.expanduser().resolve(strict=True)
    if pdf_root.expanduser().is_symlink() or not pdf.is_dir():
        raise ValueError("PDF root must be a real directory")

    by_digest: dict[str, dict[str, Any]] = {}
    by_stem: dict[str, set[str]] = defaultdict(set)
    for row in snapshot["documents"]:
        source = Path(str(row["source_path"])).resolve(strict=True)
        try:
            relative = source.relative_to(pdf)
        except ValueError as exc:
            raise ValueError(f"snapshot source escapes PDF root: {source}") from exc
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"snapshot source is unsafe: {source}")
        digest = sha256_file(source)
        if (
            digest != row.get("sha256")
            or source.stat().st_size != row.get("byte_size")
        ):
            raise ValueError(f"snapshot source changed after intake: {source}")
        document = by_digest.setdefault(
            digest,
            {
                "document_sha256": digest,
                "byte_size": source.stat().st_size,
                "paths": set(),
                "stems": set(),
                "vendors": set(),
                "record_ids": set(),
                "ground_truth": [],
                "published_pinouts": [],
                "asset_memberships": defaultdict(set),
            },
        )
        document["paths"].add(relative.as_posix())
        document["stems"].add(source.stem)
        vendor = _vendor_from_stem(source.stem)
        if vendor:
            document["vendors"].add(vendor)
        directory_vendor = (
            canonical_vendor(relative.parts[0].split(".", 1)[0])
            if len(relative.parts) > 1
            else None
        )
        if directory_vendor:
            document["vendors"].add(directory_vendor)
            alias = f"{directory_vendor}_{source.stem}".casefold()
            document["stems"].add(alias)
            by_stem[alias].add(digest)
        for binding in row.get("bindings") or []:
            if not isinstance(binding, Mapping):
                continue
            record_id = binding.get("part_id") or binding.get("opn")
            opn = binding.get("opn")
            if isinstance(record_id, str) and record_id:
                document["record_ids"].add(record_id)
            if isinstance(opn, str) and opn:
                document["asset_memberships"]["manifested_opn"].add(opn)
            bound_vendor = canonical_vendor(binding.get("vendor"))
            if bound_vendor:
                document["vendors"].add(bound_vendor)
        by_stem[source.stem].add(digest)

    gt_by_stem = {
        path.stem: path
        for path in _safe_files(ground_truth, "*.json", recursive=False)
        if path.stem in by_stem
    }
    validated_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in _safe_files(validated, "*.json", recursive=True):
        if path.stem in by_stem and not any(
            part.startswith("_")
            for part in path.relative_to(validated).parts
        ):
            validated_by_stem[path.stem].append(path)

    for stem, digests in by_stem.items():
        if len(digests) != 1:
            continue
        document = by_digest[next(iter(digests))]
        gt_path = gt_by_stem.get(stem)
        if gt_path is not None:
            summary = _summary(gt_path, ground_truth)
            document["ground_truth"].append(summary)
            document["record_ids"].add(summary["record_id"])
            if summary["vendor"]:
                document["vendors"].add(summary["vendor"])
        for sidecar in validated_by_stem.get(stem, []):
            summary = _summary(sidecar, validated)
            document["published_pinouts"].append(summary)
            document["record_ids"].add(summary["record_id"])
            if summary["vendor"]:
                document["vendors"].add(summary["vendor"])

    document_rows = []
    for digest, document in sorted(by_digest.items()):
        document_rows.append(
            {
                "document_sha256": digest,
                "byte_size": document["byte_size"],
                "paths": sorted(document["paths"]),
                "stems": sorted(document["stems"]),
                "vendors": sorted(document["vendors"]),
                "record_ids": sorted(document["record_ids"]),
                "ground_truth": sorted(
                    document["ground_truth"],
                    key=lambda row: (row["path"], row["sha256"]),
                ),
                "published_pinouts": sorted(
                    document["published_pinouts"],
                    key=lambda row: (row["path"], row["sha256"]),
                ),
                "asset_memberships": {
                    name: sorted(values)
                    for name, values in sorted(
                        document["asset_memberships"].items()
                    )
                },
            }
        )
    section_counts = Counter(
        section
        for document in document_rows
        for record in document["ground_truth"]
        for section in record["sections"]
    )
    core = {
        "schema": "harness.electronics-corpus-registry.v1",
        "policy": {
            "join_keys": ["document_sha256", "exact_source_stem"],
            "active_ground_truth_only": True,
            "active_published_pinouts_only": True,
            "symlinks_followed": False,
            "purpose": "incremental_local_training_pair_generation",
        },
        "sources": {
            "pdf_root": str(pdf),
            "ground_truth_root": str(ground_truth),
            "validated_root": str(validated),
            "incremental_source_snapshot": {
                "path": str(source_snapshot.expanduser().resolve(strict=True)),
                "sha256": snapshot_sha,
                "evidence_sha256": snapshot["evidence_sha256"],
            },
        },
        "counts": {
            "pdf_files": len(snapshot["documents"]),
            "unique_pdf_sha256": len(document_rows),
            "duplicate_pdf_files": (
                len(snapshot["documents"]) - len(document_rows)
            ),
            "active_ground_truth": sum(
                len(document["ground_truth"]) for document in document_rows
            ),
            "active_published_pinouts": sum(
                len(document["published_pinouts"])
                for document in document_rows
            ),
            "documents_with_ground_truth": sum(
                bool(document["ground_truth"]) for document in document_rows
            ),
            "documents_with_published_pinouts": sum(
                bool(document["published_pinouts"])
                for document in document_rows
            ),
            "orphan_ground_truth": 0,
            "orphan_published_pinouts": 0,
            "ground_truth_sections": dict(sorted(section_counts.items())),
        },
        "assets": [],
        "documents": document_rows,
        "orphans": {"ground_truth": [], "published_pinouts": []},
    }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **core,
        "evidence_sha256": hashlib.sha256(canonical_json(core)).hexdigest(),
    }


__all__ = ["build_incremental_corpus_registry"]
