#!/usr/bin/env python3
"""Read family device-comparison tables deterministically (knife lane v1).

Consumes the census (``scripts/census_family_documents.py``) and, for every
document marked ``family_matrix``, emits one family record in the CR
conventions shape plus a corpus summary.

    uv run --python 3.11 python scripts/extract_family_device_matrix.py \
        --census results/family-census-20260908/census.jsonl \
        --vendor st --out results/family-matrix-st-20260908

Outputs:
    <out>/records/<stem>.json     one record per document
    <out>/records.jsonl           all records, one per line
    <out>/summary.json            yield, binding kinds, unknown rate, per attribute
    <out>/manifest.json           inputs, sha256 of outputs, code version
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.electronics.family_device_matrix import MATRIX_SCHEMA, extract_family_matrix  # noqa: E402


def _one(row: dict) -> dict:
    path = Path(row["source_path"])
    try:
        return extract_family_matrix(path, row)
    except Exception as exc:  # noqa: BLE001
        return {
            "schema": MATRIX_SCHEMA,
            "_meta": {"source_path": str(path), "document_sha256": row.get("document_sha256"), "extraction_failed": repr(exc)},
            "overview": {"part_number": None, "family": None, "part_numbers_covered": [], "vendor": row.get("vendor")},
            "shared": {},
            "variants": [],
            "unbound_columns": [],
            "unknown": [],
            "conflicts": [],
            "consistency": [],
            "bogey": {"variant_columns": 0, "variants_bound": 0, "attributes_addressed": [], "cells_total": 0, "cells_typed": 0, "cells_verbatim": 0, "cells_unknown": 0},
        }


def summarize(records: list[dict]) -> dict:
    binding_kinds: Counter[str] = Counter()
    attr_cells: Counter[str] = Counter()
    attr_typed: Counter[str] = Counter()
    attr_unknown: Counter[str] = Counter()
    unknown_reasons: Counter[str] = Counter()
    unbound_reasons: Counter[str] = Counter()
    docs_with_variants = 0
    docs_fully_bound = 0
    docs_failed = 0
    parts_bound: set[str] = set()
    cells = Counter()
    conflicts = 0
    consistency_checked = 0
    consistency_ok = 0
    for record in records:
        if record["_meta"].get("extraction_failed"):
            docs_failed += 1
            continue
        bogey = record["bogey"]
        for key in ("cells_total", "cells_typed", "cells_verbatim", "cells_unknown", "variant_columns", "variants_bound"):
            cells[key] += bogey.get(key, 0)
        if record["variants"] or record["shared"]:
            docs_with_variants += 1
        if bogey["variant_columns"] and bogey["variants_bound"] == bogey["variant_columns"]:
            docs_fully_bound += 1
        for variant in record["variants"]:
            binding_kinds[variant["binding"]] += 1
            if variant["part_number"]:
                parts_bound.add(variant["part_number"])
            for attribute, leaf in variant["attributes"].items():
                attr_cells[attribute] += 1
                if leaf["status"] in ("typed", "boolean"):
                    attr_typed[attribute] += 1
                elif leaf["status"] == "unknown":
                    attr_unknown[attribute] += 1
        for attribute, leaf in record["shared"].items():
            attr_cells[attribute] += 1
            if leaf["status"] in ("typed", "boolean"):
                attr_typed[attribute] += 1
            elif leaf["status"] == "unknown":
                attr_unknown[attribute] += 1
        for item in record["unknown"]:
            unknown_reasons[item.get("reason") or "unspecified"] += 1
        for item in record["unbound_columns"]:
            unbound_reasons[item.get("reason") or "unspecified"] += 1
        conflicts += len(record["conflicts"])
        for check in record["consistency"]:
            consistency_checked += 1
            consistency_ok += int(check["ok"])
    per_attribute = {
        attribute: {
            "cells": attr_cells[attribute],
            "typed_or_boolean": attr_typed[attribute],
            "unknown": attr_unknown[attribute],
            "verbatim_only": attr_cells[attribute] - attr_typed[attribute] - attr_unknown[attribute],
        }
        for attribute in sorted(attr_cells)
    }
    return {
        "schema": "harness.electronics-family-device-matrix-summary.v1",
        "documents": len(records),
        "documents_failed": docs_failed,
        "documents_with_matrix_read": docs_with_variants,
        "documents_all_columns_bound": docs_fully_bound,
        "variant_columns": cells["variant_columns"],
        "variant_columns_bound": cells["variants_bound"],
        "distinct_parts_bound": len(parts_bound),
        "binding_kinds": dict(binding_kinds),
        "unbound_reasons": dict(unbound_reasons),
        "cells": {k: cells[k] for k in ("cells_total", "cells_typed", "cells_verbatim", "cells_unknown")},
        "unknown_reasons": dict(unknown_reasons),
        "conflicts": conflicts,
        "consistency_checks": {"checked": consistency_checked, "ok": consistency_ok},
        "per_attribute": per_attribute,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vendor", action="append", default=None, help="restrict to vendor(s)")
    parser.add_argument("--kind", default="family_matrix", help="census above_opn_kind to read (default family_matrix)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.census.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("above_opn_kind") == args.kind]
    if args.vendor:
        rows = [r for r in rows if r.get("vendor") in set(args.vendor)]
    if args.limit:
        rows = rows[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    records_dir = args.out / "records"
    records_dir.mkdir(exist_ok=True)

    started = time.time()
    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_one, row): row for row in rows}
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            stem = Path(record["_meta"]["source_path"]).stem
            (records_dir / f"{stem}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
            if index % 50 == 0 or index == len(rows):
                print(f"  {index}/{len(rows)}  {time.time() - started:.0f}s", flush=True)

    records.sort(key=lambda r: r["_meta"]["source_path"])
    jsonl = args.out / "records.jsonl"
    jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    summary = summarize(records)
    summary["elapsed_s"] = round(time.time() - started, 1)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "census": str(args.census),
        "census_sha256": hashlib.sha256(args.census.read_bytes()).hexdigest(),
        "vendors": args.vendor,
        "kind": args.kind,
        "documents": len(records),
        "records_jsonl_sha256": hashlib.sha256(jsonl.read_bytes()).hexdigest(),
        "code_commit": commit,
        "schema": MATRIX_SCHEMA,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_attribute"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
