#!/usr/bin/env python3
"""Score a Key Features grid drop against hand-checked gold pages.

Gold files (tests/fixtures/gold/key_features_*.json) list the facts a
document's own Features page prints, one per line of the target grid, keyed
by group and a label regex, optionally with the instance count, value,
quantity_qualifier, qualifier_verbatim and varies_by_part the row must carry.
`forbidden` lists rows that must NOT be on the grid tier.

Recall = expected lines found on the grid tier with every stated field
matching. Each miss is printed with what was found instead, so the fix is
visible. Exit code 1 if any gold document falls under --min-recall.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def match_row(exp: dict, row: dict) -> tuple[bool, str]:
    if row["group"] != exp["group"] or row["tier"] != "grid":
        return False, "group/tier"
    hay = f"{row.get('label') or ''} | {row.get('verbatim') or ''}"
    if not re.search(exp["label"], hay, re.I):
        return False, "label"
    for key in ("instances", "value", "quantity_qualifier", "qualifier_verbatim", "varies_by_part"):
        if key in exp and row.get(key) != exp[key]:
            return False, f"{key}={row.get(key)!r} wanted {exp[key]!r}"
    return True, ""


def score(gold: dict, grid: dict) -> dict:
    rows = grid["rows"]
    hits, misses = [], []
    for exp in gold["expected"]:
        near = []
        found = False
        for row in rows:
            ok, why = match_row(exp, row)
            if ok:
                found = True
                break
            if why not in ("group/tier", "label"):
                near.append(f"{row['label'][:50]} ({why})")
            elif why == "group/tier" and re.search(exp["label"], f"{row.get('label') or ''} | {row.get('verbatim') or ''}", re.I):
                near.append(f"{row['label'][:50]} (in {row['group']}/{row['tier']})")
        (hits if found else misses).append({**exp, "near": near[:3]})
    violations = []
    for forb in gold.get("forbidden", []):
        for row in rows:
            if row["tier"] != "grid" or row["group"] != forb["group"]:
                continue
            if not re.search(forb["label"], f"{row.get('label') or ''} | {row.get('verbatim') or ''}", re.I):
                continue
            if "varies_by_part" in forb and row.get("varies_by_part") != forb["varies_by_part"]:
                continue
            violations.append({"row": row["label"][:80], "why": forb["why"]})
    return {
        "source_artifact": gold["source_artifact"],
        "expected": len(gold["expected"]),
        "found": len(hits),
        "recall": round(len(hits) / max(1, len(gold["expected"])), 3),
        "misses": misses,
        "forbidden_on_grid": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", type=Path, required=True, help="grids.jsonl from emit_key_features_grid.py")
    parser.add_argument("--gold", type=Path, nargs="*", default=sorted(Path("tests/fixtures/gold").glob("key_features_*.json")))
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    grids = {}
    with args.grids.open() as fh:
        for line in fh:
            g = json.loads(line)
            grids[g["_meta"]["document_sha256"]] = g
            grids[g["_meta"]["source_artifact"]] = g

    reports = []
    worst = 1.0
    for gold_path in args.gold:
        gold = json.loads(gold_path.read_text())
        grid = grids.get(gold.get("document_sha256")) or grids.get(gold["source_artifact"])
        if grid is None:
            reports.append({"source_artifact": gold["source_artifact"], "error": "document not in drop"})
            worst = 0.0
            continue
        report = score(gold, grid)
        reports.append(report)
        worst = min(worst, report["recall"])
        print(f"{report['source_artifact']}: {report['found']}/{report['expected']} recall={report['recall']}  forbidden_on_grid={len(report['forbidden_on_grid'])}")
        for miss in report["misses"]:
            print(f"   MISS [{miss['group']}] /{miss['label']}/" + (f"  near: {miss['near']}" if miss["near"] else ""))
        for v in report["forbidden_on_grid"]:
            print(f"   FORBIDDEN [{v['row']}] {v['why']}")
    if args.out:
        args.out.write_text(json.dumps({"schema": "harness.electronics-key-features-gold-report.v1", "documents": reports}, indent=1))
    return 1 if worst < args.min_recall else 0


if __name__ == "__main__":
    sys.exit(main())
