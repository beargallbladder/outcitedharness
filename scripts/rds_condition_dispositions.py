#!/usr/bin/env python3
"""Dispositions for the RDS-condition subset of the ambiguous pool.

CR reship-ack-20260912 sequence: "RDS-condition subset of the ambiguous pool".
For every ambiguous_comparison RDS miss in the relabeled adjudication, read
the full same-quantity grid rows (not just the 8-row payload), match the
label's condition with the lane's containment semantics, and state what the
document actually prints at that condition — value + qualifier + verbatim —
so CR's corrections layer can act without re-opening the PDF.

Dispositions are factual, never guesses:
- doc_max_at_condition      exactly one max-family value at the label's VGS
- ambiguous_multiple_max    several distinct max values at that condition
- doc_typ_only_at_condition only typical rows at that condition (typ/label
                            equality is reported, not decided)
- unqualified_rows_only     rows exist at the condition but carry no qualifier
- no_row_at_label_condition nothing states the label's condition
- label_condition_absent    the selector label itself states no condition

    python3 scripts/rds_condition_dispositions.py \
        --adjudication /tmp/power-gold-relabel-20260912/reports/ifx2-adjudication.jsonl \
        --grids results/power-grids-20260909-ifx2/grids.jsonl \
        --out rds-dispositions-ifx2.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from power_gold_adjudication import (  # noqa: E402
    QUANTITY,
    norm_condition,
    normalize,
)

MAX_FAMILY = {"maximum", "rated", "absolute_maximum"}

_VGS = re.compile(r"vgs\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)\s*v", re.I)


def extract_vgs(row: dict) -> float | None:
    """The RDS condition axis. Some documents state VGS only inside the row
    verbatim or label text, not in condition_verbatim, so every text field
    is searched. First numeric VGS found; None when nowhere stated."""
    for key in ("condition_verbatim", "table_condition", "verbatim", "label"):
        m = _VGS.search(str(row.get(key) or ""))
        if m:
            return float(m.group(1))
    return None


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a))


def same_quantity_rows(grid: dict, field: str) -> list[dict]:
    rows = []
    pattern = QUANTITY.get(field)
    if not pattern:
        return rows
    for r in grid["rows"]:
        hay = f"{r.get('symbol') or ''} {r.get('parameter') or ''}"
        sym = (r.get("symbol") or "").replace(" ", "")
        if (sym and pattern.match(sym)) or pattern.search(r.get("parameter") or "") or any(pattern.match(a) for a in r.get("also_printed", [])):
            if r["group"] in ("thermal",) or re.search(r"storage|shutdown|hysteresis|soldering|lead\s+temp", hay, re.I):
                continue
            rows.append(r)
    return rows


def row_values_canonical(row: dict, field: str) -> list[float]:
    unit = row.get("unit")
    if field == "rds_on_mohm" and unit == "mW":
        unit = "mOhm"
    raw = row.get("value")
    vals = raw if isinstance(raw, list) else [raw]
    out = []
    for v in vals:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            n = normalize(field, unit, v)
            if n is not None:
                out.append(n)
    return out


def disposition_for(miss_row: dict, rows: list[dict]) -> dict:
    field = miss_row["source_field"]
    label_condition = miss_row.get("label_condition")
    if not label_condition:
        return {"disposition": "label_condition_absent", "same_condition_rows": [], "rows_without_vgs": len(rows)}
    want = norm_condition(label_condition)
    label_vgs = _VGS.search(str(label_condition))
    same_condition = []
    without_vgs = 0
    for r in rows:
        cond = r.get("condition_verbatim")
        if label_vgs is not None:
            row_vgs = extract_vgs(r)
            if row_vgs is None:
                without_vgs += 1
                continue
            if not _close(row_vgs, float(label_vgs.group(1))):
                continue
        elif not (cond and want in norm_condition(cond)):
            continue
        same_condition.append(r)
    evidence = [
        {
            "value": r.get("value"),
            "unit": r.get("unit"),
            "canonical_mohm": row_values_canonical(r, field),
            "quantity_qualifier": r.get("quantity_qualifier"),
            "condition_verbatim": r.get("condition_verbatim"),
            "table_kind": r.get("table_kind"),
            "verbatim": (r.get("verbatim") or "")[:200],
        }
        for r in same_condition[:8]
    ]
    if not same_condition:
        return {"disposition": "no_row_at_label_condition", "same_condition_rows": [], "rows_without_vgs": without_vgs}
    max_vals, typ_vals = set(), set()
    for r in same_condition:
        q = r.get("quantity_qualifier")
        vals = set(row_values_canonical(r, field))
        if q in MAX_FAMILY:
            max_vals |= vals
        elif q == "typical":
            typ_vals |= vals
    label_value = miss_row.get("label_value")
    label_mohm = normalize(field, miss_row.get("label_unit"), label_value) if isinstance(label_value, (int, float)) else None
    result = {"same_condition_rows": evidence}
    if len(max_vals) == 1:
        result["disposition"] = "doc_max_at_condition"
        result["doc_max_mohm"] = max_vals.pop()
    elif len(max_vals) > 1:
        result["disposition"] = "ambiguous_multiple_max"
        result["doc_max_values_mohm"] = sorted(max_vals)
    elif typ_vals:
        result["disposition"] = "doc_typ_only_at_condition"
        result["doc_typ_values_mohm"] = sorted(typ_vals)
        if label_mohm is not None and label_mohm in typ_vals:
            result["typ_equals_label"] = True
    else:
        result["disposition"] = "unqualified_rows_only"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adjudication", type=Path, required=True)
    ap.add_argument("--grids", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    grids = {}
    for line in args.grids.read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            grids[g["_meta"]["source_artifact"]] = g

    out_rows = []
    tally: Counter = Counter()
    for line in args.adjudication.read_text().splitlines():
        if not line.strip():
            continue
        miss_row = json.loads(line)
        if miss_row.get("kind") != "ambiguous_comparison" or miss_row.get("source_field") != "rds_on_mohm":
            continue
        grid = grids.get(miss_row["source_artifact"])
        rows = same_quantity_rows(grid, "rds_on_mohm") if grid else []
        result = disposition_for(miss_row, rows)
        tally[result["disposition"]] += 1
        out_rows.append({
            "source_artifact": miss_row["source_artifact"],
            "source_field": "rds_on_mohm",
            "label_value": miss_row.get("label_value"),
            "label_unit": miss_row.get("label_unit"),
            "label_condition": miss_row.get("label_condition"),
            "label_quantity_qualifier": miss_row.get("label_quantity_qualifier"),
            "fixture_source_qualifier": miss_row.get("fixture_source_qualifier"),
            **result,
        })

    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
    print(f"ambiguous RDS rows: {len(out_rows)}")
    for key in sorted(tally):
        print(f"  {key:32} {tally[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
