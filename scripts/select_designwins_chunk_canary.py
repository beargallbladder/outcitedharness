#!/usr/bin/env python3
"""Freeze complete chunk groups for the first N DesignWins parent cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select(
    parents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    llama_records: list[dict[str, Any]],
    *,
    parent_cases: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if len(chunks) != len(llama_records):
        raise ValueError("canonical and LLaMA Factory chunk counts differ")
    parent_ids = [str(row.get("pair_id") or "") for row in parents[:parent_cases]]
    if len(parent_ids) != parent_cases or any(not value for value in parent_ids):
        raise ValueError("parent dataset cannot satisfy requested case count")
    selected_ids = set(parent_ids)
    selected_chunks: list[dict[str, Any]] = []
    selected_llama: list[dict[str, Any]] = []
    for chunk, llama_record in zip(chunks, llama_records, strict=True):
        metadata = chunk.get("metadata")
        parent_id = (
            str(metadata.get("parent_pair_id") or "")
            if isinstance(metadata, dict)
            else ""
        )
        if parent_id not in selected_ids:
            continue
        selected_chunks.append(chunk)
        selected_llama.append(llama_record)
    for parent_id in parent_ids:
        rows = [
            row
            for row in selected_chunks
            if row["metadata"]["parent_pair_id"] == parent_id
        ]
        declared = {int(row["metadata"]["chunk_count"]) for row in rows}
        indices = sorted(int(row["metadata"]["chunk_index"]) for row in rows)
        if declared != {len(rows)} or indices != list(range(len(rows))):
            raise ValueError(f"selected parent {parent_id!r} has incomplete chunks")
    return selected_chunks, selected_llama, parent_ids


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", required=True, type=Path)
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--llamafactory", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--parent-cases", type=int, default=8)
    args = parser.parse_args()
    if args.parent_cases < 1:
        parser.error("--parent-cases must be positive")
    if args.destination.exists():
        raise FileExistsError(args.destination)
    parents = _jsonl(args.parents)
    chunks, llama, parent_ids = select(
        parents,
        _jsonl(args.chunks),
        json.loads(args.llamafactory.read_text(encoding="utf-8")),
        parent_cases=args.parent_cases,
    )
    args.destination.mkdir(parents=True)
    canonical = args.destination / "canonical.jsonl"
    llama_path = args.destination / "llamafactory.json"
    parents_path = args.destination / "parents.jsonl"
    _write_jsonl(canonical, chunks)
    _write_json(llama_path, llama)
    _write_jsonl(parents_path, parents[: args.parent_cases])
    manifest = {
        "schema": "harness.designwins-chunk-canary.v1",
        "parent_cases": args.parent_cases,
        "parent_ids": parent_ids,
        "chunks": len(chunks),
        "sources": {
            "parents_sha256": _sha256(args.parents),
            "chunks_sha256": _sha256(args.chunks),
            "llamafactory_sha256": _sha256(args.llamafactory),
        },
        "artifacts": {
            "canonical.jsonl": _sha256(canonical),
            "llamafactory.json": _sha256(llama_path),
            "parents.jsonl": _sha256(parents_path),
        },
    }
    _write_json(args.destination / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
