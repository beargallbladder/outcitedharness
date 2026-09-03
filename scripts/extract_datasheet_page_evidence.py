#!/usr/bin/env python3
"""Extract text blocks and tables from indexed pages without a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.electronics.claims import canonical_json
from harness.electronics.corpus import sha256_file
from harness.electronics.page_evidence import extract_profile_evidence


BUNDLE_SCHEMA = "harness.electronics-page-evidence-bundle.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-index", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--maximum-pages-per-lane", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--maximum-documents", type=int)
    return parser


def _load_profiles(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    path = root / "profiles.jsonl"
    receipt = manifest["artifacts"]["profiles.jsonl"]
    if sha256_file(path) != receipt["sha256"] or path.stat().st_size != receipt["bytes"]:
        raise ValueError("page profiles differ from their manifest")
    profiles = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return manifest, profiles


def _extract_one(
    task: tuple[dict[str, Any], int],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    profile, maximum_pages_per_lane = task
    path = Path(profile["source_path"])
    try:
        if sha256_file(path) != profile["document_sha256"]:
            raise ValueError("PDF differs from sealed corpus identity")
        import pymupdf

        with pymupdf.open(path) as document:
            rows = list(
                extract_profile_evidence(
                    document,
                    profile,
                    maximum_pages_per_lane=maximum_pages_per_lane,
                )
            )
        return rows, None
    except Exception as exc:
        return [], {
            "document_sha256": profile.get("document_sha256"),
            "source_path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("--workers must be between 1 and 32")
    if not 1 <= args.maximum_pages_per_lane <= 100:
        raise ValueError("--maximum-pages-per-lane must be between 1 and 100")
    index_root = args.page_index.expanduser().resolve(strict=True)
    index_manifest, profiles = _load_profiles(index_root)
    if args.maximum_documents is not None:
        if args.maximum_documents < 1:
            raise ValueError("--maximum-documents must be positive")
        profiles = profiles[: args.maximum_documents]
    output = args.output_directory.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    evidence_path = temporary / "page-evidence.jsonl"
    error_path = temporary / "errors.jsonl"
    page_count = 0
    table_count = 0
    lane_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    processed = 0
    try:
        tasks = [
            (profile, args.maximum_pages_per_lane) for profile in profiles
        ]
        with evidence_path.open("xb") as evidence_handle, error_path.open(
            "xb"
        ) as error_handle, ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            for rows, error in executor.map(_extract_one, tasks, chunksize=1):
                processed += 1
                if error is not None:
                    error_handle.write(canonical_json(error) + b"\n")
                    error_counts[error["error"].split(":", 1)[0]] += 1
                for row in rows:
                    evidence_handle.write(canonical_json(row) + b"\n")
                    page_count += 1
                    table_count += len(row["tables"])
                    lane_counts.update(row["lanes"])
                if processed % 100 == 0:
                    print(f"extracted {processed}/{len(tasks)}", flush=True)
            for handle in (evidence_handle, error_handle):
                handle.flush()
                os.fsync(handle.fileno())
        artifacts = {}
        for path in (evidence_path, error_path):
            artifacts[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        core = {
            "schema": BUNDLE_SCHEMA,
            "policy": {
                "extractor": "pymupdf",
                "network_used": False,
                "ocr_used": False,
                "maximum_pages_per_lane": args.maximum_pages_per_lane,
                "model_escalation": "only_after_local_attempt",
            },
            "source": {
                "page_index": str(index_root),
                "page_index_evidence_sha256": index_manifest[
                    "evidence_sha256"
                ],
            },
            "artifacts": artifacts,
            "counts": {
                "documents": len(profiles),
                "processed": processed,
                "pages": page_count,
                "tables": table_count,
                "lane_pages": dict(sorted(lane_counts.items())),
                "errors": sum(error_counts.values()),
                "errors_by_type": dict(sorted(error_counts.items())),
            },
        }
        core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **core,
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("xb") as handle:
            handle.write(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        for path in (evidence_path, error_path, manifest_path):
            os.chmod(path, 0o444)
        os.chmod(temporary, 0o555)
        os.rename(temporary, output)
    except BaseException:
        try:
            os.chmod(temporary, 0o755)
        except OSError:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
