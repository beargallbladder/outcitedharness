#!/usr/bin/env python3
"""Split power-gold misses into reader misses and label/document disagreements.

For every fixture row the lenient scorer could not find, look for a grid row
that names the same quantity (by symbol/parameter vocabulary) with a different
value. If one exists, the document prints a different number than the label
and the row is an adjudication case, not a reader miss; CR pair-split-on-sha-
20260909: "a document cannot owe two different values for one quantity".

    python3 scripts/power_gold_adjudication.py \
        --grids results/power-grids-20260909/grids.jsonl \
        --gold-report results/power-grids-20260909/gold-report-lenient.json \
        --gold-dir /tmp/power-sample/gold --out results/power-grids-20260909/adjudication.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

QUANTITY = {
    "vds_v": re.compile(r"^(?:v\(?br\)?dss|vdss|vds|bvdss)$|drain[- ]to[- ]source\s+(?:breakdown\s+)?voltage", re.I),
    "id_a": re.compile(r"^(?:id|idm|id\d+|id1d2)$|continuous\s+drain(?:[- ]to[- ]drain)?\s+current|operating\s+current", re.I),
    "rds_on_mohm": re.compile(r"^rds\(?on\)?$|on[- ]?resistance", re.I),
    "qg_nc": re.compile(r"^qg$|gate\s+charge\s+total|total\s+gate\s+charge", re.I),
    "vin_min_v": re.compile(r"^vin$|^vi$|^vcc$|\binput\s+voltage|supply\s+voltage", re.I),
    "vin_max_v": re.compile(r"^vin$|^vi$|^vcc$|\binput\s+voltage|supply\s+voltage", re.I),
    "vout_min_v": re.compile(r"^vout$|^vo$|\boutput\s+voltage", re.I),
    "vout_max_v": re.compile(r"^vout$|^vo$|\boutput\s+voltage", re.I),
    "iout_max_a": re.compile(r"^iout$|^io$|\boutput\s+current|load\s+current", re.I),
    "temp_min_c": re.compile(r"^t[aj]$|junction\s+temperature|free-air\s+temperature|ambient\s+temperature|operating\s+temperature", re.I),
    "temp_max_c": re.compile(r"^t[aj]$|junction\s+temperature|free-air\s+temperature|ambient\s+temperature|operating\s+temperature", re.I),
}


def values(row: dict) -> list[float]:
    v = row.get("value")
    if isinstance(v, list):
        return [float(x) for x in v if isinstance(x, (int, float))]
    return [float(v)] if isinstance(v, (int, float)) else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grids", type=Path, required=True)
    ap.add_argument("--gold-report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    grids = {}
    for line in args.grids.read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            grids[g["_meta"]["source_artifact"]] = g
    report = json.loads(args.gold_report.read_text())
    out_rows = []
    kinds: Counter = Counter()
    for doc in report["documents"]:
        grid = grids.get(doc["source_artifact"])
        if not grid or "misses" not in doc:
            continue
        for miss in doc["misses"]:
            field = miss.get("source_field") or ""
            pattern = QUANTITY.get(field)
            same_quantity = []
            if pattern:
                for r in grid["rows"]:
                    hay = f"{r.get('symbol') or ''} {r.get('parameter') or ''}"
                    sym = (r.get("symbol") or "").replace(" ", "")
                    if (sym and pattern.match(sym)) or pattern.search(r.get("parameter") or "") or any(pattern.match(a) for a in r.get("also_printed", [])):
                        if r["group"] in ("thermal",) or re.search(r"storage|shutdown|hysteresis|soldering|lead\s+temp", hay, re.I):
                            continue
                        same_quantity.append(r)
            if same_quantity:
                kind = "label_document_disagree"
                doc_values = sorted({v for r in same_quantity for v in values(r)})
            else:
                kind = "reader_miss"
                doc_values = []
            kinds[kind] += 1
            out_rows.append({
                "source_artifact": doc["source_artifact"],
                "source_field": field,
                "label_value": miss.get("value"),
                "label_unit": miss.get("unit"),
                "label_condition": miss.get("condition_verbatim"),
                "kind": kind,
                "document_values": doc_values[:12],
                "document_rows": [{"label": r["label"][:100], "table_kind": r.get("table_kind"), "quantity_qualifier": r.get("quantity_qualifier"), "pages": r.get("source_pages")} for r in same_quantity[:4]],
            })
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
    by_field = Counter((r["source_field"], r["kind"]) for r in out_rows)
    print(f"misses {len(out_rows)}: {dict(kinds)}")
    for field in sorted({r["source_field"] for r in out_rows}):
        print(f"  {field:14} disagree {by_field[(field, 'label_document_disagree')]:3}  reader_miss {by_field[(field, 'reader_miss')]:3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
