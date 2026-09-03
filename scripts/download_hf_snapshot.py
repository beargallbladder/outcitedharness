#!/usr/bin/env python3
"""Download a pinned Hugging Face snapshot without persisting its token."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def token_from_stdin() -> str:
    if sys.stdin.isatty():
        raise ValueError("token stdin must be a pipe")
    token = sys.stdin.readline().strip()
    if len(token) < 8 or any(character.isspace() for character in token):
        raise ValueError("token stdin is empty or malformed")
    return token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--token-stdin", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    if (
        "/" not in args.repo
        or len(args.revision) != 40
        or any(character not in "0123456789abcdef" for character in args.revision)
        or not args.local_dir.is_absolute()
        or not 1 <= args.max_workers <= 32
    ):
        parser.error("invalid pinned snapshot request")
    return args


def main() -> int:
    args = parse_args()
    if args.token_stdin:
        os.environ["HF_TOKEN"] = token_from_stdin()
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import snapshot_download

    result = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=args.local_dir,
        max_workers=args.max_workers,
    )
    print(
        json.dumps(
            {
                "repo": args.repo,
                "revision": args.revision,
                "local_dir": str(result),
                "complete": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
