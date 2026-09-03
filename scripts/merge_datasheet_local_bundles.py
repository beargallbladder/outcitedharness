#!/usr/bin/env python3
"""Merge disjoint extraction shards into one immutable evaluation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "harness.electronics-structural-local-extraction.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _verified_artifact(
    root: Path,
    manifest: dict[str, Any],
    name: str,
) -> Path:
    path = root / name
    receipt = manifest.get("artifacts", {}).get(name)
    if (
        not isinstance(receipt, dict)
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != receipt.get("bytes")
        or _sha256(path) != receipt.get("sha256")
    ):
        raise ValueError(f"bundle artifact differs from manifest: {path}")
    return path


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not value.get("work_id"):
                raise ValueError(f"{path}:{line_number} has no work ID")
            rows.append(value)
    return rows


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"immutable merge output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def merge(
    inputs: list[Path],
    output: Path,
    expected_items: int,
) -> dict[str, Any]:
    if len(inputs) < 2:
        raise ValueError("at least two extraction shards are required")
    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"immutable merge output already exists: {output}")

    source_receipts: list[dict[str, Any]] = []
    source_hashes: dict[str, set[str]] = {
        "structural_queue_sha256": set(),
        "structural_queue_evidence_sha256": set(),
        "page_evidence_sha256": set(),
    }
    models: set[str] = set()
    endpoints: set[str] = set()
    merged: dict[str, dict[str, dict[str, Any]]] = {
        "local-results.jsonl": {},
        "pillar-evidence.jsonl": {},
    }
    counts: Counter[str] = Counter()
    selected = 0

    for raw_root in inputs:
        root = raw_root.expanduser().resolve(strict=True)
        manifest_path = root / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError(f"bundle manifest is absent or unsafe: {root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"unsupported extraction bundle: {root}")
        for key in source_hashes:
            value = manifest.get("sources", {}).get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"bundle lacks source fingerprint {key}: {root}")
            source_hashes[key].add(value)
        model = manifest.get("model", {})
        models.add(str(model.get("model") or ""))
        endpoints.add(str(model.get("base_url") or ""))
        selection = manifest.get("selection", {})
        selected += int(selection.get("work_items") or 0)
        counts.update(
            {
                str(key): int(value)
                for key, value in manifest.get("counts", {}).items()
            }
        )
        source_receipts.append(
            {
                "path": str(root),
                "manifest_sha256": _sha256(manifest_path),
                "offset": selection.get("offset"),
                "limit": selection.get("limit"),
                "work_items": selection.get("work_items"),
            }
        )
        for name in merged:
            for row in _jsonl(_verified_artifact(root, manifest, name)):
                work_id = str(row["work_id"])
                if work_id in merged[name]:
                    raise ValueError(f"duplicate {name} work ID: {work_id}")
                merged[name][work_id] = row

    inconsistent = {
        key: sorted(values)
        for key, values in source_hashes.items()
        if len(values) != 1
    }
    if inconsistent:
        raise ValueError(f"extraction shards used different sources: {inconsistent}")
    if len(models) != 1 or "" in models:
        raise ValueError(f"extraction shards used different models: {models}")
    evidence_ids = set(merged["pillar-evidence.jsonl"])
    result_ids = set(merged["local-results.jsonl"])
    if selected != expected_items or len(evidence_ids) != expected_items:
        raise ValueError(
            "merged extraction shards do not cover the expected work-item count"
        )
    if not result_ids <= evidence_ids:
        raise ValueError("local results exist without corresponding pillar evidence")

    output.mkdir(parents=True)
    artifacts: dict[str, dict[str, Any]] = {}
    for name, rows in merged.items():
        payload = b"".join(
            _canonical(rows[work_id]) + b"\n" for work_id in sorted(rows)
        )
        path = output / name
        _write_new(path, payload)
        artifacts[name] = {
            "bytes": len(payload),
            "sha256": _sha256(path),
        }

    sources = {
        key: next(iter(values)) for key, values in source_hashes.items()
    }
    manifest_core = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "counts": dict(sorted(counts.items())),
        "model": {
            "provider": "local",
            "model": next(iter(models)),
            "base_urls": sorted(endpoints),
        },
        "selection": {
            "work_items": expected_items,
            "results": len(result_ids),
            "shards": len(inputs),
        },
        "sources": sources,
        "source_shards": source_receipts,
    }
    manifest_core["evidence_sha256"] = hashlib.sha256(
        _canonical(manifest_core)
    ).hexdigest()
    _write_new(
        output / "manifest.json",
        json.dumps(
            manifest_core,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode()
        + b"\n",
    )
    os.chmod(output, 0o555)
    return manifest_core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-items", required=True, type=int)
    args = parser.parse_args()
    manifest = merge(args.input, args.output, args.expected_items)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
