#!/usr/bin/env python3
"""Census every PDF in the corpus registry for above-OPN grain and bogeys.

Deterministic PyMuPDF pass, no models. Per unique document it records grain
(above_opn / single_opn / not_device_family / unknown / extraction_failed),
the parts and wildcard series the document itself prints, printed-package
pin bogeys, located device-comparison tables with the attributes they
address, and the resulting denominator (parts_or_variants x attributes).

Outputs (refuses to overwrite):
  <out>/census.jsonl   one harness.electronics-family-census.v1 row per doc
  <out>/summary.json   per-vendor and overall counts
  <out>/manifest.json  inputs, counts, evidence sha256 of census.jsonl

Example:
  uv run --python 3.11 python scripts/census_family_documents.py \
    --corpus-registry results/datasheet-corpus-registry-20260901.json \
    --page-index results/datasheet-page-index-20260901/profiles.jsonl \
    --output-directory results/family-census-20260908 --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.electronics.family_census import CENSUS_SCHEMA, census_document, summarize

CATEGORY_PREFIXES = {"mcu", "power", "connector", "battery"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus-registry", type=Path)
    source.add_argument(
        "--inventory",
        type=Path,
        help="jsonl with one document per line: {path, sha256[, title]} (CR cr-pdf-inventory shape)",
    )
    parser.add_argument(
        "--page-index",
        type=Path,
        help="profiles.jsonl from index_datasheet_pages.py; supplies lane pages",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--maximum-documents", type=int)
    parser.add_argument(
        "--vendor",
        action="append",
        help="restrict to documents whose derived vendor matches (repeatable)",
    )
    return parser


def derive_vendor(path_name: str, registry_vendors: list[str]) -> tuple[str, str]:
    """(vendor, category) from the corpus filename convention
    ``<category>_<vendor>_...`` or ``<vendor>_...``."""
    parts = path_name.split("_")
    category = registry_vendors[0] if registry_vendors else "unknown"
    if parts and parts[0].lower() in CATEGORY_PREFIXES and len(parts) > 2:
        return parts[1].lower(), parts[0].lower()
    if parts:
        return parts[0].lower(), category
    return "unknown", category


def _load_lane_pages(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sha = row.get("document_sha256")
            if isinstance(sha, str):
                out[sha] = row.get("lane_pages") or {}
    return out


def _work_items(
    registry: dict[str, Any],
    lane_pages: dict[str, dict[str, Any]],
    vendors: set[str] | None,
    maximum: int | None,
) -> list[dict[str, Any]]:
    pdf_root = Path(registry["sources"]["pdf_root"])
    items: list[dict[str, Any]] = []
    for document in registry.get("documents") or []:
        paths = document.get("paths") or []
        if not paths:
            continue
        vendor, category = derive_vendor(paths[0], document.get("vendors") or [])
        if vendors and vendor not in vendors:
            continue
        sha = document.get("document_sha256")
        items.append(
            {
                "path": str(pdf_root / paths[0]),
                "document_sha256": sha,
                "lane_pages": lane_pages.get(sha),
                "registry_stems": len(document.get("stems") or []),
                "vendor": vendor,
                "category": category,
            }
        )
        if maximum and len(items) >= maximum:
            break
    return items


_INVENTORY_VENDOR_HINTS: tuple[tuple[str, str], ...] = (
    ("stm32", "st"), ("st_", "st"), ("rl78", "renesas"), ("renesas", "renesas"), ("ra_", "renesas"),
    ("rx", "renesas"), ("msp", "ti"), ("tms", "ti"), ("tm4c", "ti"), ("spnu", "ti"), ("spms", "ti"),
    ("ti_", "ti"), ("gd32", "gigadevice"), ("efm", "silabs"), ("efr", "silabs"), ("mcx", "nxp"),
    ("lpc", "nxp"), ("imx", "nxp"), ("kl", "nxp"), ("k8", "nxp"), ("s32", "nxp"), ("nxp", "nxp"),
    ("pic", "microchip"), ("sam", "microchip"), ("atmel", "microchip"), ("psoc", "infineon"),
    ("xmc", "infineon"), ("infineon", "infineon"), ("nrf", "nordic"), ("esp", "espressif"),
)


def _inventory_items(path: Path, vendors: set[str] | None, maximum: int | None) -> list[dict[str, Any]]:
    """Work items from a CR-style inventory (path + sha256). Vendor is a
    filename hint only; the census reads vendor facts from the document."""
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            name = Path(row["path"]).name.lower()
            vendor = next((v for hint, v in _INVENTORY_VENDOR_HINTS if name.startswith(hint) or f"_{hint}" in name), "unknown")
            if vendors and vendor not in vendors:
                continue
            items.append(
                {
                    "path": row["path"],
                    "document_sha256": row.get("sha256") or row.get("document_sha256"),
                    "lane_pages": None,
                    "registry_stems": 0,
                    "vendor": vendor,
                    "category": "mcu",
                }
            )
            if maximum and len(items) >= maximum:
                break
    return items


def _run_one(item: dict[str, Any]) -> dict[str, Any]:
    try:
        row = census_document(
            Path(item["path"]),
            lane_pages=item.get("lane_pages"),
            registry_stems=int(item.get("registry_stems") or 0),
            vendor=item.get("vendor"),
            category=item.get("category"),
        )
    except Exception as error:  # noqa: BLE001 - one bad PDF must not stop the census
        row = {
            "schema": CENSUS_SCHEMA,
            "source_path": item["path"],
            "document_sha256": item.get("document_sha256"),
            "vendor": item.get("vendor"),
            "category": item.get("category"),
            "registry_stems": int(item.get("registry_stems") or 0),
            "scope_status": "extraction_failed",
            "extraction_failed_reason": f"census_crashed: {type(error).__name__}: {error}",
            "grain": "extraction_failed",
            "grain_reasons": ["census_crashed"],
        }
    registry_sha = item.get("document_sha256")
    if registry_sha and row.get("document_sha256") not in (None, registry_sha):
        row["registry_sha256_mismatch"] = registry_sha
    return row


def main() -> int:
    args = _parser().parse_args()
    out = args.output_directory
    if out.exists():
        raise SystemExit(f"output directory already exists: {out}")
    lane_pages = _load_lane_pages(args.page_index)
    vendors = {v.lower() for v in args.vendor} if args.vendor else None
    if args.inventory:
        items = _inventory_items(args.inventory, vendors, args.maximum_documents)
        source_path, source_label = args.inventory, "inventory"
    else:
        registry = json.loads(args.corpus_registry.read_text(encoding="utf-8"))
        items = _work_items(registry, lane_pages, vendors, args.maximum_documents)
        source_path, source_label = args.corpus_registry, "corpus_registry"
    if not items:
        raise SystemExit("no documents selected")
    out.mkdir(parents=True)
    census_path = out / "census.jsonl"

    rows: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    with census_path.open("w", encoding="utf-8") as handle, ProcessPoolExecutor(
        max_workers=max(1, args.workers)
    ) as pool:
        futures = [pool.submit(_run_one, item) for item in items]
        for count, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            if count % 100 == 0 or count == len(items):
                print(f"[{count}/{len(items)}] {row.get('grain')}  {Path(row['source_path']).name}", file=sys.stderr)

    rows.sort(key=lambda r: str(r.get("source_path")))
    with census_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": "harness.electronics-family-census-bundle.v1",
        "created_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "source_kind": source_label,
        "corpus_registry": str(source_path),
        "corpus_registry_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "page_index": str(args.page_index) if args.page_index else None,
        "vendors_filter": sorted(vendors) if vendors else None,
        "maximum_documents": args.maximum_documents,
        "documents": len(rows),
        "grain": summary["overall"]["grain"],
        "evidence_sha256": hashlib.sha256(census_path.read_bytes()).hexdigest(),
        "policy": {
            "deterministic_pymupdf_only": True,
            "models_used": [],
            "wildcards_never_expanded": True,
            "bogeys_held_by_verifier_not_shown_to_extractors": True,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"documents": len(rows), "grain": summary["overall"]["grain"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
