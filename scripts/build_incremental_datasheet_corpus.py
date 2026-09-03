#!/usr/bin/env python3
"""Build a corpus registry for one stable, deduplicated download cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.electronics.corpus import write_new_registry
from harness.electronics.incremental_corpus import (
    build_incremental_corpus_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--pdf-root", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--validated-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = build_incremental_corpus_registry(
        args.source_snapshot,
        pdf_root=args.pdf_root,
        ground_truth_root=args.ground_truth_root,
        validated_root=args.validated_root,
    )
    write_new_registry(args.output, registry)
    print(
        json.dumps(
            {
                "status": "sealed",
                "output": str(args.output.expanduser().resolve()),
                "evidence_sha256": registry["evidence_sha256"],
                "counts": registry["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
