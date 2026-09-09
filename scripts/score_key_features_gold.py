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
from typing import Any


_SI_PREFIX = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3, "M": 1e6, "G": 1e9}
_UNIT_BASE = {"ohm": "ohm", "ω": "ohm", "v": "v", "a": "a", "c": "c", "ºc": "c", "hz": "hz", "w": "w", "f": "f", "h": "h", "s": "s", "j": "j", "%": "%"}


def to_base(value: Any, unit: str | None) -> tuple[float, str] | None:
    """(6.8, 'mΩ') -> (0.0068, 'ohm'); units the table does not know return None."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    u = (unit or "").strip().replace("Ohms", "Ohm").replace("ohms", "ohm").replace("°", "")
    if not u:
        return (float(value), "")
    if u.lower() in _UNIT_BASE:
        return (float(value), _UNIT_BASE[u.lower()])
    if u[0] in _SI_PREFIX and u[1:].lower() in _UNIT_BASE:
        base = _UNIT_BASE[u[1:].lower()]
        if base == "c":
            base = "coulomb"  # nC is charge; bare C in these fixtures is Celsius
        return (float(value) * _SI_PREFIX[u[0]], base)
    return None


def values_equal(exp: dict, row: dict) -> bool:
    """Exact after SI normalisation when both sides carry a unit; a range row
    ([lo, hi]) satisfies a fixture that asks for either bound."""
    want = exp["value"]
    got = row.get("value")
    if isinstance(got, list) and len(got) == 2 and not isinstance(want, list):
        return any(values_equal({**exp, "value": want}, {**row, "value": g}) for g in got)
    if "unit" in exp and exp["unit"] and row.get("unit"):
        a, b = to_base(want, exp["unit"]), to_base(got, row.get("unit"))
        if a and b and a[1] == b[1]:
            if abs(a[0] - b[0]) <= 1e-9 * max(1.0, abs(a[0])):
                return True
            # P-channel VDS/ID print negative; vendor selectors are unsigned.
            if a[1] in {"v", "a"} and abs(abs(a[0]) - abs(b[0])) <= 1e-9 * max(1.0, abs(a[0])):
                return True
            return False
    return got == want


def norm_condition(text: str) -> str:
    return re.sub(r"\s+|[:=]", "", str(text)).lower().replace("ω", "ohm").replace("Ω", "ohm").replace("°", "")


def match_row(exp: dict, row: dict) -> tuple[bool, str]:
    if row["group"] != exp["group"] or row["tier"] != "grid":
        return False, "group/tier"
    # A fact printed twice is one row; the other spelling is kept in
    # also_printed and counts as present ("12-bit ADC modules" under
    # "12-bit Analog-to-Digital Converters (ADC)").
    hay = " | ".join([row.get("label") or "", row.get("verbatim") or "", *(row.get("also_printed") or [])])
    if not re.search(exp["label"], hay, re.I):
        return False, "label"
    if "value" in exp and not values_equal(exp, row):
        return False, f"value={row.get('value')!r} {row.get('unit') or ''} wanted {exp['value']!r} {exp.get('unit') or ''}"
    for key in ("instances", "quantity_qualifier", "qualifier_verbatim", "varies_by_part"):
        if key in exp and row.get(key) != exp[key]:
            return False, f"{key}={row.get(key)!r} wanted {exp[key]!r}"
    # Third axis (CR power-gold-fixtures-20260909): the test condition is
    # independent of which bound and which hedge. Compared after normalising
    # whitespace, case and the =/: spelling.
    # The document usually states more of the condition than the label does
    # ("VGS = 10 V, ID = 18 A" against "VGS = 10 V"), so the label's condition
    # must be contained in the row's, not equal to it.
    if "condition_verbatim" in exp and norm_condition(exp["condition_verbatim"]) not in norm_condition(row.get("condition_verbatim") or ""):
        return False, f"condition_verbatim={row.get('condition_verbatim')!r} wanted {exp['condition_verbatim']!r}"
    return True, ""


def score(gold: dict, grid: dict, ignore_keys: tuple[str, ...] = ()) -> dict:
    rows = grid["rows"]
    hits, misses = [], []
    for exp in gold["expected"]:
        exp = {k: v for k, v in exp.items() if k not in ignore_keys}
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
    parser.add_argument("--ignore-key", action="append", default=[], help="fixture key to drop before matching (e.g. quantity_qualifier) to report a lenient recall beside the strict one")
    parser.add_argument("--quiet", action="store_true", help="totals only")
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
            print(f"{gold['source_artifact']}: NOT IN DROP")
            worst = 0.0
            continue
        report = score(gold, grid, tuple(args.ignore_key))
        reports.append(report)
        worst = min(worst, report["recall"])
        if not args.quiet:
            print(f"{report['source_artifact']}: {report['found']}/{report['expected']} recall={report['recall']}  forbidden_on_grid={len(report['forbidden_on_grid'])}")
            for miss in report["misses"]:
                print(f"   MISS [{miss['group']}] /{miss['label']}/" + (f"  near: {miss['near']}" if miss["near"] else ""))
            for v in report["forbidden_on_grid"]:
                print(f"   FORBIDDEN [{v['row']}] {v['why']}")
    scored = [r for r in reports if "error" not in r]
    expected = sum(r["expected"] for r in scored)
    found = sum(r["found"] for r in scored)
    by_field: dict[str, list[int]] = {}
    for r in scored:
        for m in r["misses"]:
            by_field.setdefault(m.get("source_field") or m["group"], [0, 0])[1] += 1
    print(f"TOTAL documents {len(scored)} (not in drop: {len(reports) - len(scored)})  rows {found}/{expected} = {round(found / expected, 4) if expected else None}" + (f"  ignoring {args.ignore_key}" if args.ignore_key else ""))
    if by_field:
        print("misses by source_field: " + ", ".join(f"{k} {v[1]}" for k, v in sorted(by_field.items(), key=lambda kv: -kv[1][1])))
    if args.out:
        args.out.write_text(json.dumps({"schema": "harness.electronics-key-features-gold-report.v1", "documents": reports}, indent=1))
    return 1 if worst < args.min_recall else 0


if __name__ == "__main__":
    sys.exit(main())
