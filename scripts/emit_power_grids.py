#!/usr/bin/env python3
"""Emit power-datasheet grids (electrical characteristics tables) for scoring.

    uv run --python 3.11 python scripts/emit_power_grids.py \
        --pairs /Volumes/M5_4TB/exports/cr_drops/power-pairs-20260909/pairs-ti.jsonl \
        --pdf-dir /tmp/power-sample/pdf \
        --out results/power-grids-20260909

One grid per distinct pdf_sha256 (CR pair-split-on-sha-20260909); a document
fetched for several part numbers is one document whose scope lists them all.
Only documents present under --pdf-dir are read; the rest are counted.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.electronics.power_datasheet import build_power_grid  # noqa: E402


def _one(job: tuple[str, str, str | None, list[str], str | None]) -> dict:
    path, sha, vendor, parts, aisle = job
    try:
        return build_power_grid(Path(path), vendor=vendor, part_numbers=parts, aisle=aisle, sha256=sha)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-800:], "_meta": {"document_sha256": sha, "source_artifact": Path(path).name}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, action="append", required=True)
    ap.add_argument("--pdf-dir", type=Path, action="append", required=True)
    ap.add_argument("--only-sha", type=Path, default=None, help="json list of pair rows or sha256 strings to restrict to")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    by_sha: dict[str, dict] = {}
    parts: dict[str, list[str]] = defaultdict(list)
    for path in args.pairs:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            by_sha.setdefault(p["pdf_sha256"], p)
            parts[p["pdf_sha256"]].append(p["part_number"])
    restrict = None
    if args.only_sha:
        data = json.loads(args.only_sha.read_text())
        restrict = {d["pdf_sha256"] if isinstance(d, dict) else d for d in data}

    jobs = []
    missing = 0
    for sha, p in by_sha.items():
        if restrict is not None and sha not in restrict:
            continue
        name = Path(p["pdf_path"]).name
        found = next((d / name for d in args.pdf_dir if (d / name).exists()), None)
        if found is None:
            missing += 1
            continue
        jobs.append((str(found), sha, p.get("vendor"), sorted(set(parts[sha])), p.get("aisle")))

    grids, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for g in pool.map(_one, jobs, chunksize=4):
            (errors if "error" in g else grids).append(g)

    with (args.out / "grids.jsonl").open("w") as fh:
        for g in grids:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")
    with (args.out / "grid_rows.jsonl").open("w") as fh:
        for g in grids:
            for r in g["rows"]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (args.out / "errors.jsonl").open("w") as fh:
        for e in errors:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    rows = [r for g in grids for r in g["rows"]]
    summary = {
        "schema": "harness.electronics-power-grid-summary.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documents_in_manifest": len(by_sha),
        "documents_read": len(grids),
        "documents_missing_pdf": missing,
        "documents_errored": len(errors),
        "rows": len(rows),
        "rows_by_group": dict(Counter(r["group"] for r in rows)),
        "rows_by_table_kind": dict(Counter(r["table_kind"] for r in rows)),
        "rows_with_condition": sum(1 for r in rows if r.get("condition_verbatim")),
        "quantity_qualifier_counts": dict(Counter(r.get("quantity_qualifier") for r in rows)),
        "documents_with_zero_rows": sum(1 for g in grids if not g["rows"]),
        "by_aisle_documents": dict(Counter(g["_meta"].get("aisle") for g in grids)),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
