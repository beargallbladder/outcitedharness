"""Build a hash-bound join over existing datasheet and CR assets."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "harness.electronics-corpus-registry.v1"
ACTIVE_GROUND_TRUTH_SECTIONS = (
    "pinout",
    "interface_atoms",
    "peripheral_depth",
    "clock_specs",
    "abs_max_ratings",
    "power_modes",
    "overview",
)
DOCUMENT_SHA_KEYS = (
    "pdf_sha256",
    "source_document_sha256",
    "source_content_sha256",
)
RECORD_ID_KEYS = (
    "record_id",
    "queue_id",
    "part_number",
    "base_mpn",
    "series_key",
    "custom_id",
)


@dataclass(frozen=True)
class AssetSource:
    name: str
    path: Path
    kind: str = "auto"

    def __post_init__(self) -> None:
        if not self.name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in self.name):
            raise ValueError(f"invalid asset name: {self.name!r}")
        if self.kind not in {"auto", "json", "jsonl", "directory"}:
            raise ValueError(f"invalid asset kind: {self.kind!r}")


@dataclass(frozen=True)
class CorpusInputs:
    pdf_root: Path
    ground_truth_root: Path
    validated_root: Path
    source_audit: Path | None = None
    row_dataset_manifest: Path | None = None
    assets: tuple[AssetSource, ...] = ()
    expected_pdf_files: int | None = None


@dataclass
class _Document:
    digest: str
    byte_size: int
    paths: set[str] = field(default_factory=set)
    stems: set[str] = field(default_factory=set)
    vendors: set[str] = field(default_factory=set)
    ground_truth: list[dict[str, Any]] = field(default_factory=list)
    published_pinouts: list[dict[str, Any]] = field(default_factory=list)
    asset_memberships: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_files(root: Path, pattern: str, *, recursive: bool) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"corpus root must be a regular directory: {root}")
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and not path.is_symlink()
    )


def _hash_files(paths: Sequence[Path], workers: int) -> dict[Path, str]:
    if workers < 1 or workers > 32:
        raise ValueError("hash_workers must be between 1 and 32")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        digests = executor.map(sha256_file, paths)
        return dict(zip(paths, digests, strict=True))


def _json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON source must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON source must contain an object: {path}")
    return value


VENDOR_ALIASES = {
    "freescale/nxp": "nxp",
    "microchip technology": "microchip",
    "silicon laboratories": "silabs",
    "silicon labs": "silabs",
    "stmicroelectronics": "st",
    "texas instruments": "ti",
}


def canonical_vendor(value: Any) -> str | None:
    raw = str(value or "").strip().casefold()
    if not raw:
        return None
    raw = raw.removesuffix(".com")
    return VENDOR_ALIASES.get(raw, raw)


def _vendor_from_stem(stem: str) -> str | None:
    prefix = stem.split("_", 1)[0]
    return canonical_vendor(prefix)


def _part_key(value: Any) -> str | None:
    raw = str(value or "").strip().upper()
    if not raw:
        return None
    return "".join(character for character in raw if character.isalnum())


def _record_ids(row: Mapping[str, Any], fallback: str) -> list[str]:
    output: list[str] = []
    for key in RECORD_ID_KEYS:
        value = row.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            output.append(f"{key}:{str(value).strip()}")
    return output or [f"path:{fallback}"]


def _document_digest(row: Mapping[str, Any]) -> str | None:
    for key in DOCUMENT_SHA_KEYS:
        value = row.get(key)
        if (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ):
            return value
    return None


def _part_candidates(value: Mapping[str, Any]) -> set[str]:
    output: set[str] = set()
    overview = value.get("overview")
    for candidate in (
        value.get("part_number"),
        value.get("base_mpn"),
        overview.get("part_number") if isinstance(overview, Mapping) else None,
    ):
        key = _part_key(candidate)
        if key:
            output.add(key)
    return output


def _record_summary(
    *,
    path: Path,
    digest: str,
    value: Mapping[str, Any],
    relative_to: Path,
) -> dict[str, Any]:
    metadata = value.get("_meta")
    provenance = metadata.get("provenance") if isinstance(metadata, Mapping) else None
    overview = value.get("overview")
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": digest,
        "record_id": path.stem,
        "part_numbers": sorted(_part_candidates(value)),
        "vendor": (
            canonical_vendor(overview.get("manufacturer"))
            if isinstance(overview, Mapping)
            else None
        )
        or _vendor_from_stem(path.stem),
        "sections": [
            section for section in ACTIVE_GROUND_TRUTH_SECTIONS if section in value
        ],
        "batch_id": (
            metadata.get("batch_id") if isinstance(metadata, Mapping) else None
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


def _verify_evidence_digest(value: Mapping[str, Any], kind: str) -> None:
    expected = value.get("evidence_sha256")
    if expected is None:
        return
    core = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "evidence_sha256"}
    }
    actual = hashlib.sha256(canonical_json(core)).hexdigest()
    if expected != actual:
        raise ValueError(f"{kind} has an invalid evidence_sha256")


def _load_source_audit(path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if path is None:
        return {}, {}
    value = _json_object(path)
    if value.get("schema") != "harness.pinout-vision-source-audit.v1":
        raise ValueError("source audit schema is not supported")
    _verify_evidence_digest(value, "source audit")
    by_pdf: dict[str, Any] = {}
    by_published: dict[str, Any] = {}
    for row in value.get("candidates") or []:
        if not isinstance(row, dict):
            raise ValueError("source audit candidate is malformed")
        pdf_path = row.get("pdf_path")
        published_path = row.get("published_path")
        if isinstance(pdf_path, str):
            if pdf_path in by_pdf:
                # One PDF can legitimately back multiple package records.
                current = by_pdf[pdf_path]
                if isinstance(current, list):
                    current.append(row)
                else:
                    by_pdf[pdf_path] = [current, row]
            else:
                by_pdf[pdf_path] = row
        if isinstance(published_path, str):
            by_published[published_path] = row
    return by_pdf, by_published


def _rows(path: Path, kind: str) -> Iterable[dict[str, Any]]:
    if kind == "json":
        value = _json_object(path)
        records = value.get("records")
        if isinstance(records, list):
            for index, record in enumerate(records, 1):
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{path}: records entry {index} is not an object"
                    )
                yield record
        else:
            yield value
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSONL at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            yield value


def _asset_files(asset: AssetSource) -> tuple[str, list[Path]]:
    path = asset.path.expanduser().resolve(strict=True)
    kind = asset.kind
    if kind == "auto":
        if path.is_dir():
            kind = "directory"
        elif path.suffix.casefold() == ".jsonl":
            kind = "jsonl"
        elif path.suffix.casefold() == ".json":
            kind = "json"
        else:
            raise ValueError(f"cannot infer asset kind for {path}")
    if kind == "directory":
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"asset directory is invalid: {path}")
        files = sorted(
            child.resolve()
            for child in path.rglob("*")
            if child.is_file()
            and not child.is_symlink()
            and child.suffix.casefold() in {".json", ".jsonl"}
        )
    else:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"asset file is invalid: {path}")
        files = [path]
    return kind, files


def _inventory_asset(
    asset: AssetSource,
    *,
    documents: dict[str, _Document],
    part_index: Mapping[tuple[str | None, str], set[str]],
) -> dict[str, Any]:
    kind, files = _asset_files(asset)
    rows_total = 0
    direct_joins = 0
    part_joins = 0
    unresolved = 0
    ambiguous = 0
    unique_ids: set[str] = set()
    file_receipts: list[dict[str, Any]] = []
    for path in files:
        file_digest = sha256_file(path)
        file_receipts.append(
            {
                "path": str(path),
                "sha256": file_digest,
                "bytes": path.stat().st_size,
            }
        )
        row_kind = "jsonl" if path.suffix.casefold() == ".jsonl" else "json"
        for row_number, row in enumerate(_rows(path, row_kind), 1):
            rows_total += 1
            identifiers = _record_ids(row, f"{path.name}:{row_number}")
            unique_ids.update(identifiers)
            document_digest = _document_digest(row)
            candidates: set[str] = set()
            route = "unresolved"
            if document_digest is not None and document_digest in documents:
                candidates = {document_digest}
                route = "direct"
            else:
                part = _part_key(
                    row.get("part_number")
                    or row.get("base_mpn")
                    or row.get("series")
                )
                vendor = canonical_vendor(row.get("vendor") or row.get("domain"))
                if part:
                    candidates = set(part_index.get((vendor, part), set()))
                    if not candidates:
                        candidates = set(part_index.get((None, part), set()))
                if len(candidates) == 1:
                    route = "part"
            if len(candidates) == 1:
                joined_digest = next(iter(candidates))
                documents[joined_digest].asset_memberships[asset.name].update(
                    identifiers
                )
                if route == "direct":
                    direct_joins += 1
                else:
                    part_joins += 1
            elif len(candidates) > 1:
                ambiguous += 1
            else:
                unresolved += 1
    return {
        "name": asset.name,
        "kind": kind,
        "files": file_receipts,
        "rows": rows_total,
        "unique_record_identifiers": len(unique_ids),
        "joins": {
            "document_sha256": direct_joins,
            "unambiguous_part": part_joins,
            "ambiguous": ambiguous,
            "unresolved": unresolved,
        },
    }


def build_corpus_registry(
    inputs: CorpusInputs,
    *,
    hash_workers: int = 4,
) -> dict[str, Any]:
    pdf_root = inputs.pdf_root.expanduser().resolve(strict=True)
    gt_root = inputs.ground_truth_root.expanduser().resolve(strict=True)
    validated_root = inputs.validated_root.expanduser().resolve(strict=True)
    pdf_paths = _regular_files(pdf_root, "*.pdf", recursive=False)
    if (
        inputs.expected_pdf_files is not None
        and len(pdf_paths) != inputs.expected_pdf_files
    ):
        raise ValueError(
            f"expected {inputs.expected_pdf_files} PDFs, found {len(pdf_paths)}"
        )
    gt_paths = _regular_files(gt_root, "*.json", recursive=False)
    validated_paths = [
        path
        for path in _regular_files(validated_root, "*.json", recursive=True)
        if not any(part.startswith("_") for part in path.relative_to(validated_root).parts)
    ]

    all_primary_paths = [*pdf_paths, *gt_paths, *validated_paths]
    hashes = _hash_files(all_primary_paths, hash_workers)
    source_by_pdf, source_by_published = _load_source_audit(inputs.source_audit)

    documents: dict[str, _Document] = {}
    pdf_by_name: dict[str, str] = {}
    pdf_by_stem: dict[str, set[str]] = defaultdict(set)
    for path in pdf_paths:
        digest = hashes[path]
        document = documents.setdefault(
            digest,
            _Document(digest=digest, byte_size=path.stat().st_size),
        )
        relative = path.relative_to(pdf_root).as_posix()
        document.paths.add(relative)
        document.stems.add(path.stem)
        vendor = _vendor_from_stem(path.stem)
        if vendor:
            document.vendors.add(vendor)
        pdf_by_name[path.name] = digest
        pdf_by_stem[path.stem].add(digest)

    expected_hash_mismatches: list[dict[str, str]] = []
    part_index: dict[tuple[str | None, str], set[str]] = defaultdict(set)
    orphan_gt: list[dict[str, Any]] = []
    for path in gt_paths:
        value = _json_object(path)
        summary = _record_summary(
            path=path,
            digest=hashes[path],
            value=value,
            relative_to=gt_root,
        )
        candidates = set(pdf_by_stem.get(path.stem, set()))
        audit_rows = source_by_pdf.get(f"{path.stem}.pdf")
        for audit in audit_rows if isinstance(audit_rows, list) else [audit_rows]:
            if isinstance(audit, dict):
                expected = audit.get("pdf_sha256")
                actual = pdf_by_name.get(str(audit.get("pdf_path") or ""))
                if isinstance(actual, str):
                    candidates.add(actual)
                    if isinstance(expected, str) and expected != actual:
                        expected_hash_mismatches.append(
                            {
                                "path": str(audit.get("pdf_path")),
                                "expected": expected,
                                "actual": actual,
                            }
                        )
        if len(candidates) == 1:
            document_digest = next(iter(candidates))
            documents[document_digest].ground_truth.append(summary)
            vendor = summary["vendor"]
            if vendor:
                documents[document_digest].vendors.add(vendor)
            for part in summary["part_numbers"]:
                part_index[(vendor, part)].add(document_digest)
                part_index[(None, part)].add(document_digest)
        else:
            orphan_gt.append(
                {
                    **summary,
                    "reason": (
                        "ambiguous_pdf_stem" if candidates else "no_matching_pdf_stem"
                    ),
                }
            )

    orphan_published: list[dict[str, Any]] = []
    for path in validated_paths:
        value = _json_object(path)
        summary = _record_summary(
            path=path,
            digest=hashes[path],
            value=value,
            relative_to=validated_root,
        )
        relative = path.relative_to(validated_root).as_posix()
        audit = source_by_published.get(relative)
        candidates = set(pdf_by_stem.get(path.stem, set()))
        if isinstance(audit, dict):
            expected = audit.get("pdf_sha256")
            if isinstance(expected, str) and expected in documents:
                candidates.add(expected)
        if len(candidates) == 1:
            document_digest = next(iter(candidates))
            documents[document_digest].published_pinouts.append(summary)
            vendor = summary["vendor"]
            if vendor:
                documents[document_digest].vendors.add(vendor)
            for part in summary["part_numbers"]:
                part_index[(vendor, part)].add(document_digest)
                part_index[(None, part)].add(document_digest)
        else:
            orphan_published.append(
                {
                    **summary,
                    "reason": (
                        "ambiguous_pdf_stem" if candidates else "no_matching_pdf_stem"
                    ),
                }
            )

    if expected_hash_mismatches:
        raise ValueError(
            "source audit PDF hashes differ from the live corpus: "
            f"{expected_hash_mismatches[:3]}"
        )

    asset_reports = [
        _inventory_asset(
            asset,
            documents=documents,
            part_index=part_index,
        )
        for asset in inputs.assets
    ]

    source_receipts: dict[str, Any] = {
        "pdf_root": str(pdf_root),
        "ground_truth_root": str(gt_root),
        "validated_root": str(validated_root),
    }
    if inputs.source_audit is not None:
        path = inputs.source_audit.expanduser().resolve(strict=True)
        source_receipts["source_audit"] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    if inputs.row_dataset_manifest is not None:
        path = inputs.row_dataset_manifest.expanduser().resolve(strict=True)
        manifest = _json_object(path)
        if manifest.get("schema") != "harness.pinout-vision-row-dataset.v1":
            raise ValueError("row dataset manifest schema is not supported")
        source_receipts["row_dataset_manifest"] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "evidence_sha256": manifest.get("evidence_sha256"),
            "counts": manifest.get("counts"),
        }

    document_rows: list[dict[str, Any]] = []
    for digest, document in sorted(documents.items()):
        record_ids = {
            record["record_id"]
            for record in [*document.ground_truth, *document.published_pinouts]
        }
        document_rows.append(
            {
                "document_sha256": digest,
                "byte_size": document.byte_size,
                "paths": sorted(document.paths),
                "stems": sorted(document.stems),
                "vendors": sorted(document.vendors),
                "record_ids": sorted(record_ids),
                "ground_truth": sorted(
                    document.ground_truth,
                    key=lambda row: (row["path"], row["sha256"]),
                ),
                "published_pinouts": sorted(
                    document.published_pinouts,
                    key=lambda row: (row["path"], row["sha256"]),
                ),
                "asset_memberships": {
                    name: sorted(identifiers)
                    for name, identifiers in sorted(
                        document.asset_memberships.items()
                    )
                },
            }
        )

    section_counts = Counter(
        section
        for document in documents.values()
        for record in document.ground_truth
        for section in record["sections"]
    )
    core: dict[str, Any] = {
        "schema": SCHEMA,
        "policy": {
            "join_keys": [
                "document_sha256",
                "exact_source_stem",
                "unambiguous_vendor_and_part",
            ],
            "active_ground_truth_only": True,
            "active_published_pinouts_only": True,
            "symlinks_followed": False,
            "purpose": "local_training_pair_generation",
        },
        "sources": source_receipts,
        "counts": {
            "pdf_files": len(pdf_paths),
            "unique_pdf_sha256": len(documents),
            "duplicate_pdf_files": len(pdf_paths) - len(documents),
            "active_ground_truth": len(gt_paths),
            "active_published_pinouts": len(validated_paths),
            "documents_with_ground_truth": sum(
                bool(document.ground_truth) for document in documents.values()
            ),
            "documents_with_published_pinouts": sum(
                bool(document.published_pinouts) for document in documents.values()
            ),
            "orphan_ground_truth": len(orphan_gt),
            "orphan_published_pinouts": len(orphan_published),
            "ground_truth_sections": dict(sorted(section_counts.items())),
        },
        "assets": asset_reports,
        "documents": document_rows,
        "orphans": {
            "ground_truth": sorted(
                orphan_gt,
                key=lambda row: (row["reason"], row["path"]),
            ),
            "published_pinouts": sorted(
                orphan_published,
                key=lambda row: (row["reason"], row["path"]),
            ),
        },
    }
    core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **core,
    }


def verify_corpus_registry(value: Mapping[str, Any]) -> None:
    if value.get("schema") != SCHEMA:
        raise ValueError("corpus registry schema is not supported")
    expected = value.get("evidence_sha256")
    core = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "evidence_sha256"}
    }
    core["evidence_sha256"] = expected
    # The digest was calculated before the digest field was attached.
    digest_input = {key: item for key, item in core.items() if key != "evidence_sha256"}
    if expected != hashlib.sha256(canonical_json(digest_input)).hexdigest():
        raise ValueError("corpus registry evidence digest is invalid")
    counts = value.get("counts")
    documents = value.get("documents")
    if not isinstance(counts, Mapping) or not isinstance(documents, list):
        raise ValueError("corpus registry is malformed")
    if counts.get("unique_pdf_sha256") != len(documents):
        raise ValueError("corpus registry document count does not match")
    seen: set[str] = set()
    for document in documents:
        if not isinstance(document, Mapping):
            raise ValueError("corpus registry document is malformed")
        digest = document.get("document_sha256")
        if digest in seen:
            raise ValueError("corpus registry contains duplicate documents")
        seen.add(str(digest))


def write_new_registry(path: Path, value: Mapping[str, Any]) -> None:
    verify_corpus_registry(value)
    destination = path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "AssetSource",
    "CorpusInputs",
    "SCHEMA",
    "build_corpus_registry",
    "canonical_vendor",
    "canonical_json",
    "sha256_file",
    "verify_corpus_registry",
    "write_new_registry",
]
