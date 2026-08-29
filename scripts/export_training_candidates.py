#!/usr/bin/env python3
"""Export real Harness and greenfield records through provenance gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.training.adapters import (
    load_greenfield_git_candidates,
    load_harness_pass_candidates,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest(
    destination: Path,
    *,
    kind: str,
    count: int,
    rejections: list[dict[str, Any]],
    data_use_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema": "harness.training.candidate-export.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "artifact": {
            "path": destination.name,
            "rows": count,
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        },
        "data_use_counts": data_use_counts,
        "rejections": rejections,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    harness = subparsers.add_parser("harness")
    harness.add_argument("--database", required=True, type=Path)
    harness.add_argument("--cases-root", required=True, type=Path)
    harness.add_argument("--answer-root", type=Path)
    harness.add_argument("--destination", required=True, type=Path)
    harness.add_argument("--approved-model-key", action="append", default=[])
    harness.add_argument("--strict", action="store_true")

    git = subparsers.add_parser("greenfield-git")
    git.add_argument("--database", required=True, type=Path)
    git.add_argument("--runs-root", required=True, type=Path)
    git.add_argument("--destination", required=True, type=Path)
    git.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = args.destination.resolve(strict=False)
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite an existing export")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rejections: list[dict[str, Any]] = []

    if args.command == "harness":
        rows = load_harness_pass_candidates(
            args.database,
            args.cases_root,
            approved_model_keys=frozenset(args.approved_model_key),
            answer_root=args.answer_root,
            destination=destination,
            strict=args.strict,
            rejections=rejections,
        )
        data_use_counts: dict[str, int] = {}
        for row in rows:
            data_use_counts[row.data_use.value] = (
                data_use_counts.get(row.data_use.value, 0) + 1
            )
        kind = "harness-pass"
    else:
        rows = load_greenfield_git_candidates(
            args.database,
            runs_root=args.runs_root,
            destination=destination,
            strict=args.strict,
            rejections=rejections,
        )
        data_use_counts = {"quarantine": len(rows)}
        kind = "greenfield-git"

    manifest = _manifest(
        destination,
        kind=kind,
        count=len(rows),
        rejections=rejections,
        data_use_counts=data_use_counts,
    )
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest["artifact"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
