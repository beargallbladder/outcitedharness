#!/usr/bin/env python3
"""Create and verify deterministic SHA-256 manifests for training artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "harness.training.manifest.v1"
CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def regular_files(root: Path, excluded: set[Path]) -> list[Path]:
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for dirname in list(dirnames):
            child = base / dirname
            if child.is_symlink():
                raise ValueError(f"refusing symlinked directory: {child.relative_to(root)}")
        for filename in filenames:
            child = base / filename
            if child in excluded:
                continue
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"refusing symlinked file: {child.relative_to(root)}")
            if not stat.S_ISREG(mode):
                raise ValueError(f"refusing non-regular file: {child.relative_to(root)}")
            files.append(child)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def build_manifest(root: Path, output: Path) -> dict[str, Any]:
    excluded = {output.resolve(strict=False)}
    entries = []
    for path in regular_files(root, excluded):
        relative = path.relative_to(root).as_posix()
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": SCHEMA,
        "algorithm": "sha256",
        "artifact": root.name,
        "files": entries,
    }


def encoded_manifest(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()


def create(root: Path, output: Path) -> int:
    root = root.resolve()
    output = output.resolve(strict=False)
    if not root.is_dir():
        raise ValueError(f"artifact root is not a directory: {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = encoded_manifest(build_manifest(root, output))

    if output.exists():
        if not output.is_file() or output.is_symlink():
            raise ValueError(f"refusing non-regular output: {output}")
        if output.read_bytes() == payload:
            print(f"manifest unchanged: {output}")
            return 0
        raise ValueError(
            f"manifest already exists with different content: {output}; use a new path"
        )

    descriptor, temp_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, 0o640)
        try:
            os.link(temp_path, output)
        except FileExistsError:
            if output.read_bytes() != payload:
                raise ValueError(f"output appeared concurrently with different content: {output}")
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"wrote {len(payload)} bytes to {output}")
    return 0


def load_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"manifest is not a regular file: {path}")
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA or data.get("algorithm") != "sha256":
        raise ValueError("unsupported manifest schema or algorithm")
    if not isinstance(data.get("files"), list):
        raise ValueError("manifest files must be a list")
    return data


def verify(root: Path, manifest_path: Path, allow_extra: bool) -> int:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    if not root.is_dir():
        raise ValueError(f"artifact root is not a directory: {root}")
    manifest = load_manifest(manifest_path)
    expected: dict[str, dict[str, Any]] = {}
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise ValueError("invalid manifest entry")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative or relative.startswith("/"):
            raise ValueError("manifest contains an invalid relative path")
        candidate = (root / relative).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"manifest path escapes artifact root: {relative}") from error
        if relative in expected:
            raise ValueError(f"duplicate manifest path: {relative}")
        expected[relative] = entry

    errors: list[str] = []
    for relative, entry in expected.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            errors.append(f"missing or non-regular: {relative}")
            continue
        actual_size = path.stat().st_size
        if actual_size != entry.get("bytes"):
            errors.append(f"size mismatch: {relative}")
            continue
        if sha256_file(path) != entry.get("sha256"):
            errors.append(f"checksum mismatch: {relative}")

    if not allow_extra:
        excluded = {manifest_path} if manifest_path.is_relative_to(root) else set()
        actual = {
            path.relative_to(root).as_posix()
            for path in regular_files(root, excluded)
        }
        for relative in sorted(actual - expected.keys()):
            errors.append(f"unexpected file: {relative}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"verified {len(expected)} files under {root}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("root", type=Path)
    create_parser.add_argument("output", type=Path)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("root", type=Path)
    verify_parser.add_argument("manifest", type=Path)
    verify_parser.add_argument(
        "--allow-extra",
        action="store_true",
        help="verify listed files without rejecting additional artifact files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "create":
            return create(args.root, args.output)
        return verify(args.root, args.manifest, args.allow_extra)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"training manifest: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
