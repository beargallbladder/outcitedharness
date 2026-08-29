#!/usr/bin/env python3
"""Enforce the ASUS1 scratch free-space floor without deleting anything."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


DEFAULT_FLOOR = 250 * 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--min-free-bytes", type=int)
    return parser.parse_args()


def marker_values(path: Path) -> dict[str, str]:
    marker = path / ".harness-training-owner-v1"
    if not marker.is_file() or marker.is_symlink():
        raise ValueError(f"missing regular ownership marker: {marker}")
    values: dict[str, str] = {}
    for line in marker.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def main() -> int:
    args = parse_args()
    try:
        path = args.path.resolve(strict=True)
        if not path.is_dir():
            raise ValueError(f"scratch path is not a directory: {path}")
        marker = marker_values(path)
        if marker.get("role") != "asus1" or marker.get("root") != str(path):
            raise ValueError("scratch ownership marker does not identify this ASUS1 root")
        floor_file = path / ".minimum-free-bytes"
        configured = int(floor_file.read_text().strip()) if floor_file.is_file() else DEFAULT_FLOOR
        floor = args.min_free_bytes if args.min_free_bytes is not None else configured
        if floor < DEFAULT_FLOOR:
            raise ValueError(f"free-space floor cannot be below {DEFAULT_FLOOR} bytes")
        free = shutil.disk_usage(path).free
        if free < floor:
            print(
                f"ASUS1 scratch below floor: free={free} required={floor}",
                file=sys.stderr,
            )
            return 1
        print(f"ASUS1 scratch ready: free={free} floor={floor}")
        return 0
    except (OSError, ValueError) as error:
        print(f"training scratch check: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
