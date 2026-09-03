#!/usr/bin/env python3
"""Seal a CR exact-OPN join as a hash-verified datasheet source cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from harness.electronics.claims import canonical_json
from harness.electronics.corpus import sha256_file


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"manifest must be a regular non-symlink file: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _identity(path: Path) -> tuple[int, int, int, int]:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"source is not a regular file: {path}")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _verify_pdf(
    task: tuple[Path, str, int],
) -> tuple[Path, tuple[int, int, int, int], str | None]:
    path, expected_sha, expected_bytes = task
    try:
        before = _identity(path)
        if before[2] != expected_bytes:
            raise ValueError(
                f"byte count {before[2]} differs from manifest {expected_bytes}"
            )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            digest = hashlib.sha256()
            prefix = b""
            tail = b""
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    if len(prefix) < 1024:
                        prefix = (prefix + chunk)[:1024]
                    tail = (tail + chunk)[-8192:]
                    digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before != after_identity or before != _identity(path):
            raise ValueError("source changed while hashing")
        if b"%PDF-" not in prefix or b"%%EOF" not in tail:
            raise ValueError("source does not have complete PDF framing")
        if digest.hexdigest() != expected_sha:
            raise ValueError("source SHA-256 differs from manifest")
        return path, before, None
    except Exception as exc:
        return path, (0, 0, 0, 0), f"{type(exc).__name__}: {exc}"


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
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
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--join-manifest", type=Path, required=True)
    parser.add_argument("--join-receipt", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--hash-workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.hash_workers <= 32:
        raise ValueError("--hash-workers must be within 1..32")

    join_path = args.join_manifest.expanduser().resolve(strict=True)
    receipt_path = args.join_receipt.expanduser().resolve(strict=True)
    corpus_root = args.corpus_root.expanduser().resolve(strict=True)
    corpus_manifest_path = args.corpus_manifest.expanduser().resolve(
        strict=True
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if sha256_file(join_path) != receipt.get("join_manifest_sha256"):
        raise ValueError("CR join manifest differs from its receipt")
    if Path(str(receipt.get("corpus_root"))).resolve() != corpus_root:
        raise ValueError("CR join receipt names a different corpus root")

    join_rows = _jsonl(join_path)
    corpus_rows = _jsonl(corpus_manifest_path)
    latest_by_file: dict[str, dict[str, Any]] = {}
    for row in corpus_rows:
        relative = row.get("file")
        if isinstance(relative, str) and relative:
            latest_by_file[relative] = row

    bindings_by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paths_by_sha: dict[str, set[Path]] = defaultdict(set)
    bytes_by_sha: dict[str, set[int]] = defaultdict(set)
    rejections: list[dict[str, Any]] = []
    for index, row in enumerate(join_rows, 1):
        if (
            row.get("bind_status") != "mpn_exact"
            or row.get("bind_method") != "MPN_EXACT"
        ):
            rejections.append(
                {
                    "line": index,
                    "opn": row.get("opn"),
                    "reason": "not_mpn_exact",
                }
            )
            continue
        relative = row.get("manifest_file")
        corpus_row = latest_by_file.get(str(relative))
        if corpus_row is None:
            rejections.append(
                {
                    "line": index,
                    "opn": row.get("opn"),
                    "reason": "missing_latest_corpus_manifest_row",
                }
            )
            continue
        digest = str(row.get("pdf_sha256") or "")
        expected_bytes = int(row.get("pdf_bytes") or 0)
        if (
            corpus_row.get("sha256") != digest
            or int(corpus_row.get("bytes") or 0) != expected_bytes
        ):
            rejections.append(
                {
                    "line": index,
                    "opn": row.get("opn"),
                    "reason": "join_differs_from_latest_corpus_manifest",
                }
            )
            continue
        source = (corpus_root / str(relative)).resolve()
        try:
            source.relative_to(corpus_root)
        except ValueError:
            rejections.append(
                {
                    "line": index,
                    "opn": row.get("opn"),
                    "reason": "corpus_path_escape",
                }
            )
            continue
        paths_by_sha[digest].add(source)
        bytes_by_sha[digest].add(expected_bytes)
        bindings_by_sha[digest].append(
            {
                "opn": row.get("opn"),
                "part_id": row.get("part_id"),
                "vendor": row.get("vendor"),
                "published_pin_count": row.get("published_pin_count"),
                "published_status": row.get("published_status"),
                "withhold_reason": row.get("withhold_reason"),
                "source_file": row.get("source_file"),
                "pin_payload_pointer": row.get("pin_payload_pointer"),
            }
        )

    tasks = []
    selected_by_sha: dict[str, Path] = {}
    for digest, paths in sorted(paths_by_sha.items()):
        byte_counts = bytes_by_sha[digest]
        if len(byte_counts) != 1:
            raise ValueError(f"conflicting byte counts for PDF {digest}")
        existing = sorted(
            path
            for path in paths
            if path.exists() and path.is_file() and not path.is_symlink()
        )
        if not existing:
            rejections.extend(
                {
                    "opn": binding["opn"],
                    "reason": "corpus_pdf_missing",
                }
                for binding in bindings_by_sha[digest]
            )
            continue
        selected = existing[0]
        selected_by_sha[digest] = selected
        tasks.append((selected, digest, next(iter(byte_counts))))

    verified: dict[Path, tuple[int, int, int, int]] = {}
    errors: dict[Path, str] = {}
    with ThreadPoolExecutor(max_workers=args.hash_workers) as executor:
        for path, identity, error in executor.map(_verify_pdf, tasks):
            if error is None:
                verified[path] = identity
            else:
                errors[path] = error

    documents = []
    admitted_bindings = 0
    for digest, source in sorted(selected_by_sha.items()):
        error = errors.get(source)
        if error is not None:
            rejections.extend(
                {
                    "opn": binding["opn"],
                    "reason": "pdf_verification_failed",
                    "detail": error,
                }
                for binding in bindings_by_sha[digest]
            )
            continue
        identity = verified[source]
        bindings = sorted(
            bindings_by_sha[digest],
            key=lambda row: (str(row["opn"]), str(row["part_id"])),
        )
        admitted_bindings += len(bindings)
        observation_payload = {
            "path": str(source),
            "device": identity[0],
            "inode": identity[1],
            "byte_size": identity[2],
            "mtime_ns": identity[3],
        }
        documents.append(
            {
                "observation_id": (
                    "source-"
                    + hashlib.sha256(
                        canonical_json(observation_payload)
                    ).hexdigest()[:32]
                ),
                "source_path": str(source),
                "alternate_paths": [
                    str(path)
                    for path in sorted(paths_by_sha[digest])
                    if path != source
                ],
                "byte_size": identity[2],
                "mtime_ns": identity[3],
                "sha256": digest,
                "bindings": bindings,
            }
        )

    core = {
        "schema": "harness.electronics-incremental-source-snapshot.v1",
        "purpose": "stable_deduplicated_datasheet_intake",
        "cohort_id": args.cohort_id,
        "policy": {
            "binding": "MPN_EXACT_only",
            "family_manuals_admitted": False,
            "file_stability_checked_during_hash": True,
            "latest_corpus_manifest_hash_required": True,
            "pdf_framing_required": True,
        },
        "sources": {
            "join_manifest": {
                "path": str(join_path),
                "sha256": sha256_file(join_path),
            },
            "join_receipt": {
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
            },
            "corpus_manifest": {
                "path": str(corpus_manifest_path),
                "sha256": sha256_file(corpus_manifest_path),
                "rows": len(corpus_rows),
            },
            "corpus_root": str(corpus_root),
        },
        "counts": {
            "documents": len(documents),
            "bindings_seen": len(join_rows),
            "bindings_admitted": admitted_bindings,
            "bindings_rejected": len(rejections),
        },
        "documents": documents,
        "rejections": rejections,
    }
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **core,
        "evidence_sha256": hashlib.sha256(canonical_json(core)).hexdigest(),
    }
    _write_new(args.output, output)
    print(
        json.dumps(
            {
                "status": "sealed",
                "output": str(args.output.expanduser().resolve()),
                "evidence_sha256": output["evidence_sha256"],
                "counts": output["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
