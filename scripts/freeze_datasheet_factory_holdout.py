#!/usr/bin/env python3
"""Freeze multi-axis document holdouts before broader local training."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from harness.electronics.corpus import verify_corpus_registry
from harness.electronics.holdout import freeze_factory_holdout


def _write_new(path: Path, value: dict) -> None:
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
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-registry", type=Path, required=True)
    parser.add_argument("--row-dataset", type=Path, required=True)
    parser.add_argument("--page-index", type=Path)
    parser.add_argument("--fraction", type=float, default=0.15)
    parser.add_argument("--minimum-documents", type=int, default=100)
    parser.add_argument("--maximum-documents", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(
        args.corpus_registry.expanduser().resolve(strict=True).read_text()
    )
    verify_corpus_registry(registry)
    now = datetime.now(timezone.utc)
    holdout = freeze_factory_holdout(
        registry,
        row_dataset_root=args.row_dataset,
        page_index_root=args.page_index,
        fraction=args.fraction,
        minimum_documents=args.minimum_documents,
        maximum_documents=args.maximum_documents,
        temporal_cutoff=now,
    )
    holdout["created_at"] = now.isoformat()
    _write_new(args.output, holdout)
    print(json.dumps(holdout["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
