#!/usr/bin/env python3
"""Read a document's cover-page scope statement: what does it govern?

Thin CLI over ``harness.electronics.document_scope.read_scope``; see that
module for the full contract (schema harness.electronics-document-scope.v2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.electronics.document_scope import read_scope  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--control-token",
        help=(
            "token that MUST appear in the front matter (vendor family stem, "
            "e.g. STM32); absence marks the row extraction_failed rather "
            "than scope_absent"
        ),
    )
    args = parser.parse_args()

    rows = [read_scope(path, args.control_token) for path in args.pdfs]
    if args.output:
        if args.output.exists():
            raise SystemExit(f"output already exists: {args.output}")
        with args.output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    for row in rows:
        print(json.dumps(row, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
