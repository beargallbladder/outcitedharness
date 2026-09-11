#!/usr/bin/env python3
"""Split power-gold misses into reader misses and label/document disagreements.

For every fixture row the lenient scorer could not find, look for a grid row
that names the same quantity (by symbol/parameter vocabulary) with a different
value. If one exists, the document prints a different number than the label
and the row is an adjudication case, not a reader miss; CR pair-split-on-sha-
20260909: "a document cannot owe two different values for one quantity".

CR adjudicator-fixes-20260911 patch: the old comparator inflated the disagree
piles ~5x because (1) it diffed label values against document values as
printed, never normalising Omega<->milliohm (Infineon prints mOhm numerals
while fixture labels carry Ohm; the extractor's Omega->W glyph confusion makes
some rows print mW), and (2) for id_a it unioned every same-quantity row, so
the document max grabbed pulsed IDM rows, output-characteristic points and
silicon-limited (chip-limited) currents. This version normalises every row
into the field's canonical unit before diffing and admits only spec-grade rows
for id_a (absolute-maximum plus prose continuous-drain, never pulsed/peak/
silicon-limited). A label whose numeral matches a normalised document value is
a unit/identity match, not a disagreement; for rds_on_mohm the label is also
tried as a plain mOhm numeral because ROHM selector exports are known to
carry mOhm numerals under an "Ohm" unit string.

Kinds: unit_identity_match | label_document_disagree | no_spec_row |
reader_miss. document_values is normalised to the canonical unit;
document_values_raw keeps the as-printed numbers for traceability.

    python3 scripts/power_gold_adjudication.py \
        --grids results/power-grids-20260909/grids.jsonl \
        --gold-report results/power-grids-20260909/gold-report-lenient.json \
        --out results/power-grids-20260909/adjudication.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

QUANTITY = {
    "vds_v": re.compile(r"^(?:v\(?br\)?dss|vdss|vds|bvdss)$|drain[- ]to[- ]source\s+(?:breakdown\s+)?voltage", re.I),
    "id_a": re.compile(r"^(?:id|id\d+|id1d2)$|continuous\s+drain(?:[- ]to[- ]drain)?\s+current|operating\s+current", re.I),
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

CANONICAL_UNIT = {
    "rds_on_mohm": "mOhm",
    "id_a": "A",
    "iout_max_a": "A",
    "vds_v": "V",
    "vin_min_v": "V",
    "vin_max_v": "V",
    "vout_min_v": "V",
    "vout_max_v": "V",
    "qg_nc": "nC",
    "temp_min_c": "C",
    "temp_max_c": "C",
}

# Rows that are a different spec than the continuous, package-limited drain
# current the selector labels describe: pulsed/peak (IDM), silicon- or
# chip-limited. CR adjudicator-fixes-20260911: "chip limited is a different
# spec than package-limited ID".
NOT_CONTINUOUS_ID = re.compile(r"pulsed|peak\s+drain|silicon\s*limited|chip\s*limited", re.I)
PROSE_DRAIN = re.compile(r"continuous\s+drain|drain\s+current", re.I)

# Strip private-use padding, NULs, control characters and spaces from unit
# strings before parsing ("mΩ\uf020" -> "mΩ").
_UNIT_NOISE = re.compile(r"[\uf000-\uf0ff\x00-\x20]+")


def _clean_unit(unit: str | None) -> str:
    return _UNIT_NOISE.sub("", unit or "")


def _ohm_kind(unit: str) -> str | None:
    """Classify a resistance unit string as 'milli' or 'plain' (Ohm).

    The Omega glyph reaches us in several encodings (Ω, Ω, mojibake Â) and the
    extractor sometimes reads it as W, so inside an RDS(on) row mW means mΩ
    and W means Ω. A bare 'm' (or m# / m\\x00) is a milli prefix whose Omega
    glyph was eaten.
    """
    u = _clean_unit(unit).lower().replace("µ", "u")
    if not u:
        return None
    u = u.replace("ω", "ohm").replace("ω", "ohm").replace("â", "ohm")
    if u.startswith("m"):
        rest = u[1:]
        if not rest or "ohm" in rest or rest in ("w", "#"):
            return "milli"
        return None
    if "ohm" in u or u in ("w",):
        return "plain"
    return None


_METRIC = {
    ("A", "a"): 1.0, ("A", "ma"): 1e-3, ("A", "ua"): 1e-6, ("A", "ka"): 1e3,
    ("V", "v"): 1.0, ("V", "mv"): 1e-3, ("V", "kv"): 1e3,
    ("nC", "nc"): 1.0, ("nC", "uc"): 1e3, ("nC", "mc"): 1e6, ("nC", "pc"): 1e-3, ("nC", "c"): 1e9,
}


def normalize(field: str, unit: str | None, value: float) -> float | None:
    """Convert one as-printed number into the field's canonical unit."""
    canonical = CANONICAL_UNIT.get(field)
    if canonical is None or not isinstance(value, (int, float)):
        return None
    u = _clean_unit(unit).lower().replace("µ", "u").replace("°", "")
    if canonical == "mOhm":
        kind = _ohm_kind(unit or "")
        if kind == "milli":
            return float(value)
        if kind == "plain":
            return float(value) * 1000.0
        return None
    if canonical == "C":
        if u in ("c", "k"):
            return float(value) if u == "c" else float(value) - 273.15
        if u == "f":
            return (float(value) - 32.0) * 5.0 / 9.0
        return None
    scale = _METRIC.get((canonical, u))
    return float(value) * scale if scale is not None else None


def label_readings(field: str, unit: str | None, value) -> list[tuple[str, float]]:
    """Plausible canonical readings of a fixture label value.

    Every field gets its proper conversion. rds_on_mohm additionally gets the
    raw numeral read as mOhm, because ROHM selector exports carry mOhm
    numerals under an "Ohm" unit string (CR adjudicator-fixes-20260911).
    """
    if not isinstance(value, (int, float)):
        return []
    proper = normalize(field, unit, value)
    readings = []
    if proper is not None:
        readings.append(("as_printed", proper))
    if field == "rds_on_mohm" and isinstance(value, (int, float)):
        numeral = float(value)
        if not any(math.isclose(numeral, v, rel_tol=1e-9) for _, v in readings):
            readings.append(("as_mohm_numeral", numeral))
    return readings


def values(row: dict) -> list[float]:
    v = row.get("value")
    if isinstance(v, list):
        return [float(x) for x in v if isinstance(x, (int, float))]
    return [float(v)] if isinstance(v, (int, float)) else []


def spec_grade(field: str, row: dict) -> bool:
    """Is this row spec-grade evidence for the field's quantity?

    Only id_a needs row selection (CR adjudicator-fixes-20260911): pulsed
    IDM, output-characteristic and silicon-limited rows are a different spec
    than the continuous package-limited drain current. For id_a only
    absolute-maximum rows and prose continuous-drain rows qualify; symbols
    whose multi-column concatenation contains IDM are excluded because their
    value lists cannot be split reliably.
    """
    if field != "id_a":
        return True
    symbol = row.get("symbol") or ""
    hay = f"{symbol} {row.get('parameter') or ''} {row.get('label') or ''}"
    if "IDM" in symbol.upper():
        return False
    if NOT_CONTINUOUS_ID.search(hay):
        return False
    kind = row.get("table_kind")
    if kind == "absolute_maximum":
        return True
    if kind == "prose" and PROSE_DRAIN.search(hay):
        return True
    return False


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


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
            if not same_quantity:
                kind = "reader_miss"
                doc_values: list[float] = []
                doc_raw: list[float] = []
                payload_rows = []
                label_match = None
            else:
                spec_rows = [r for r in same_quantity if spec_grade(field, r)]
                payload_rows = [
                    {
                        "label": r["label"][:100],
                        "table_kind": r.get("table_kind"),
                        "quantity_qualifier": r.get("quantity_qualifier"),
                        "pages": r.get("source_pages"),
                        "value": r.get("value"),
                        "unit": r.get("unit"),
                        "spec_grade": spec_grade(field, r),
                    }
                    for r in (spec_rows + [r for r in same_quantity if r not in spec_rows])[:4]
                ]
                norm: set[float] = set()
                raw: set[float] = set()
                for r in spec_rows:
                    for v in values(r):
                        raw.add(v)
                        n = normalize(field, r.get("unit"), v)
                        if n is not None:
                            norm.add(n)
                doc_values = sorted(norm)
                doc_raw = sorted(raw)
                readings = label_readings(field, miss.get("unit"), miss.get("value"))
                label_match = None
                for name, n in readings:
                    if any(close(n, d) for d in norm):
                        label_match = name
                        break
                if label_match is not None:
                    kind = "unit_identity_match"
                elif norm:
                    kind = "label_document_disagree"
                else:
                    kind = "no_spec_row"
            kinds[kind] += 1
            row = {
                "source_artifact": doc["source_artifact"],
                "source_field": field,
                "label_value": miss.get("value"),
                "label_unit": miss.get("unit"),
                "label_condition": miss.get("condition_verbatim"),
                "kind": kind,
                "canonical_unit": CANONICAL_UNIT.get(field),
                "document_values": doc_values[:12],
                "document_values_raw": doc_raw[:12],
                "document_rows": payload_rows,
            }
            if label_match is not None:
                row["label_reading_matched"] = label_match
            out_rows.append(row)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
    by_field = Counter((r["source_field"], r["kind"]) for r in out_rows)
    print(f"misses {len(out_rows)}: {dict(kinds)}")
    for field in sorted({r["source_field"] for r in out_rows}):
        parts = [f"{k.split('_')[0]} {by_field[(field, k)]}" for k in ("unit_identity_match", "label_document_disagree", "no_spec_row", "reader_miss") if by_field[(field, k)]]
        print(f"  {field:14} {'  '.join(parts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
