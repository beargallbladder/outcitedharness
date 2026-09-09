#!/usr/bin/env python3
"""Read family-level documents (reference manuals, TRMs, family data sheets)
in full and emit one family record per document.

Input is either a census jsonl (``harness.electronics-family-census.v1``,
filtered to the document classes you want) or an inventory jsonl of
``{path, sha256[, vendor]}`` rows. Output: ``records/<stem>.json`` per
document, ``records.jsonl``, ``summary.json``, ``manifest.json``.

    uv run --python 3.11 python scripts/extract_family_documents.py \
        --census results/family-census-20260908b/census.jsonl \
        --classes reference_manual --min-pages 300 \
        --out results/family-documents-20260908
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.electronics.family_document import SCHEMA, read_family_document  # noqa: E402

SUMMARY_SCHEMA = "harness.electronics-family-document-summary.v1"


def _one(path: str, vendor: str | None, max_pages: int | None) -> dict:
    try:
        return read_family_document(Path(path), vendor=vendor, max_text_pages=max_pages)
    except Exception as exc:  # noqa: BLE001
        return {"schema": SCHEMA, "_meta": {"source_path": path, "vendor": vendor}, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]}


def _load(args: argparse.Namespace) -> list[tuple[str, str | None]]:
    rows = []
    if args.census:
        for line in args.census.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if args.classes and row.get("document_class") not in args.classes:
                continue
            if args.vendor and row.get("vendor") not in args.vendor:
                continue
            if (row.get("page_count") or 0) < args.min_pages:
                continue
            rows.append((row["source_path"], row.get("vendor")))
    else:
        for line in args.inventory.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("pdf_pages") or row.get("page_count") or 10**6) < args.min_pages:
                continue
            path = row["path"]
            if args.path_prefix_map:
                old, new = args.path_prefix_map.split("=", 1)
                path = path.replace(old, new, 1) if path.startswith(old) else path
            if not Path(path).exists():
                continue
            rows.append((path, row.get("vendor")))
    if args.limit:
        rows = rows[: args.limit]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--census", type=Path)
    source.add_argument("--inventory", type=Path)
    parser.add_argument("--classes", nargs="*", default=None, help="census document_class values to keep (census input)")
    parser.add_argument("--vendor", nargs="*", default=None)
    parser.add_argument("--min-pages", type=int, default=0)
    parser.add_argument("--max-text-pages", type=int, default=None, help="cap pages read per document (debug)")
    parser.add_argument("--path-prefix-map", default=None, help="OLD=NEW rewrite for inventory paths mounted elsewhere")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = _load(args)
    out = args.out
    (out / "records").mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    tally: Counter = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_one, path, vendor, args.max_text_pages): path for path, vendor in rows}
        for future in as_completed(futures):
            record = future.result()
            path = futures[future]
            stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(path).stem)[:100]
            (out / "records" / f"{stem}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
            records.append(record)
            if "error" in record:
                tally["error"] += 1
                print(f"  ! {Path(path).name}: {record['error']}", file=sys.stderr)
            else:
                tally["ok"] += 1
                b = record["bogey"]
                print(f"  {Path(path).name[:50]:50s} id={record['identity']['document_id']!s:14s} chapters={b['chapter_entries']:4d} classes={b['peripheral_classes']:2d} inst={b['instances_asserted']:3d} feat={b['features']:3d} chfeat={b['chapter_features']:4d}/{b['chapter_feature_sections']:3d} mem={b['memory_rows']:3d}")
    with (out / "records.jsonl").open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    ok = [r for r in records if "error" not in r]
    class_presence: Counter = Counter()
    instance_classes: Counter = Counter()
    for r in ok:
        class_presence.update(r["chapters"]["peripheral_classes"].keys())
        instance_classes.update(k for k, v in r["peripheral_instances"].items() if v["count_asserted"])
    summary = {
        "schema": SUMMARY_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documents": len(records),
        "tally": dict(tally),
        "by_vendor": dict(Counter(r["_meta"].get("vendor") for r in ok)),
        "document_ids_found": sum(1 for r in ok if r["identity"]["document_id"]),
        "peripheral_class_presence": dict(class_presence.most_common()),
        "instance_class_presence": dict(instance_classes.most_common()),
        "features_total": sum(r["bogey"]["features"] for r in ok),
        "features_qualified": sum(r["bogey"]["features_qualified"] for r in ok),
        "memory_rows_total": sum(r["bogey"]["memory_rows"] for r in ok),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "manifest.json").write_text(json.dumps({"schema": "harness.run-manifest.v1", "script": "scripts/extract_family_documents.py", "args": {k: str(v) for k, v in vars(args).items()}, "documents": len(records), "generated_at": summary["generated_at"]}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("peripheral_class_presence", "instance_class_presence")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
