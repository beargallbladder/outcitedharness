#!/usr/bin/env python3
"""Index datasheet evidence pages with PyMuPDF and exact package gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.electronics.corpus import sha256_file, verify_corpus_registry
from harness.electronics.page_index import (
    PAGE_INDEX_SCHEMA,
    index_document,
    package_requests_from_ground_truth,
)


BUNDLE_SCHEMA = "harness.electronics-page-index-bundle.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-registry", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--fallback-maximum-pages",
        type=int,
        help="Optional safety cap for text-heading fallback scans.",
    )
    parser.add_argument(
        "--maximum-documents",
        type=int,
        help="Bounded smoke run; omitted means the full unique corpus.",
    )
    parser.add_argument(
        "--supplemental-bindings",
        type=Path,
        help=(
            "Optional JSONL of {document_sha256, package} rows minted from "
            "verified upstream extraction (for example ordering-page OPN "
            "decodes). Expected pin counts derive from digits printed in "
            "the package name itself; rows without digits are ignored. "
            "Every downstream locator gate still applies unchanged."
        ),
    )
    return parser


def _supplemental_requests(
    path: Path | None,
) -> dict[str, set[tuple[str, int]]]:
    if path is None:
        return {}
    requests: dict[str, set[tuple[str, int]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        package = str(row.get("package") or "").strip()
        digest = str(row.get("document_sha256") or "").strip()
        # Pin count is the trailing digit run in the package name itself
        # (LQFP100 -> 100); no word boundary exists inside such tokens.
        matches = [int(item) for item in re.findall(r"(\d{2,4})", package)]
        if package and digest and matches:
            requests.setdefault(digest, set()).add((package, matches[-1]))
    return requests


def _ground_truth_values(
    registry: dict[str, Any],
    document: dict[str, Any],
) -> list[dict[str, Any]]:
    root = Path(registry["sources"]["ground_truth_root"]).resolve()
    output: list[dict[str, Any]] = []
    for summary in document.get("ground_truth") or []:
        path = (root / summary["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("ground-truth path escapes configured root")
        if sha256_file(path) != summary["sha256"]:
            raise ValueError(f"ground truth changed after corpus seal: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            output.append(value)
    return output


def _index_one(
    task: tuple[str, str, tuple[tuple[str, int], ...], int | None],
) -> dict[str, Any]:
    document_sha256, raw_path, package_requests, fallback_maximum_pages = task
    path = Path(raw_path)
    try:
        import pymupdf

        with pymupdf.open(path) as document:
            return index_document(
                document,
                document_sha256=document_sha256,
                source_path=path,
                package_requests=package_requests,
                fallback_maximum_pages=fallback_maximum_pages,
            )
    except Exception as exc:
        core = {
            "schema": PAGE_INDEX_SCHEMA,
            "document_sha256": document_sha256,
            "source_path": str(path),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "section_entries": [],
            "lane_pages": {},
            "exact_pin_locations": [],
        }
        core["evidence_sha256"] = hashlib.sha256(
            json.dumps(
                core,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return core


def main() -> int:
    args = _parser().parse_args()
    if args.workers < 1 or args.workers > 32:
        raise ValueError("--workers must be between 1 and 32")
    registry_path = args.corpus_registry.expanduser().resolve(strict=True)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    verify_corpus_registry(registry)
    supplemental = _supplemental_requests(args.supplemental_bindings)
    pdf_root = Path(registry["sources"]["pdf_root"]).resolve()
    documents = list(registry["documents"])
    if args.maximum_documents is not None:
        if args.maximum_documents < 1:
            raise ValueError("--maximum-documents must be positive")
        documents = documents[: args.maximum_documents]
    tasks: list[tuple[str, str, tuple[tuple[str, int], ...], int | None]] = []
    for document in documents:
        paths = document.get("paths") or []
        if not paths:
            raise ValueError("corpus document has no source path")
        path = (pdf_root / paths[0]).resolve()
        if not path.is_relative_to(pdf_root):
            raise ValueError("PDF path escapes configured root")
        merged = set(
            package_requests_from_ground_truth(
                _ground_truth_values(registry, document)
            )
        )
        merged.update(supplemental.get(document["document_sha256"], set()))
        requests = tuple(sorted(merged))
        tasks.append(
            (
                document["document_sha256"],
                str(path),
                requests,
                args.fallback_maximum_pages,
            )
        )

    output = args.output_directory.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    profiles_path = temporary / "profiles.jsonl"
    lane_documents: Counter[str] = Counter()
    exact_statuses: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    indexed = 0
    try:
        with profiles_path.open("wb") as handle, ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            for profile in executor.map(_index_one, tasks, chunksize=1):
                handle.write(
                    json.dumps(
                        profile,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                    + b"\n"
                )
                indexed += 1
                for lane, pages in profile.get("lane_pages", {}).items():
                    if pages:
                        lane_documents[lane] += 1
                for result in profile.get("exact_pin_locations") or []:
                    exact_statuses[result["status"]] += 1
                if profile.get("status") == "error":
                    errors[profile["error"].split(":", 1)[0]] += 1
                if indexed % 100 == 0:
                    print(f"indexed {indexed}/{len(tasks)}", flush=True)
            handle.flush()
            os.fsync(handle.fileno())
        artifact = {
            "sha256": sha256_file(profiles_path),
            "bytes": profiles_path.stat().st_size,
        }
        core = {
            "schema": BUNDLE_SCHEMA,
            "policy": {
                "extractor": "pymupdf",
                "fallback_maximum_pages": args.fallback_maximum_pages,
                "package_match": "exact_family_count_and_variant",
                "ambiguous_package_action": "withhold",
            },
            "source": {
                "corpus_registry": str(registry_path),
                "corpus_registry_sha256": sha256_file(registry_path),
                "corpus_evidence_sha256": registry["evidence_sha256"],
            },
            "artifacts": {"profiles.jsonl": artifact},
            "counts": {
                "documents": len(tasks),
                "indexed": indexed,
                "errors": sum(errors.values()),
                "errors_by_type": dict(sorted(errors.items())),
                "documents_with_lane": dict(sorted(lane_documents.items())),
                "exact_pin_location_statuses": dict(sorted(exact_statuses.items())),
            },
        }
        core["evidence_sha256"] = hashlib.sha256(
            json.dumps(
                core,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
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
        for path in (profiles_path, manifest_path):
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
