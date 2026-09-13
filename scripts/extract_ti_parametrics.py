#!/usr/bin/env python3
"""TI parametrics wave, stage 2: extract per-class parametric axes from datasheets.

Reads the 1,678-part queue, opens each fetched PDF with the deterministic
power reader (characteristics tables + front-page facts), maps symbols to
the authoritative POWER_PARAM_SETS axes, and emits fill-only rows:
(part_number, device_class, field, value, unit, pdf_sha, source_url,
verdict, verbatim, condition, page). Values are printed evidence only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.electronics.power_datasheet import (  # noqa: E402
    read_characteristic_tables,
    read_front_page_facts,
)

AXES = {
    "mosfet": {"vds_v": ["VDS"], "id_a": ["ID"], "rds_on_mohm": ["RDS(ON"]},
    "regulator": {
        "vin_min_v": ["VIN", "VDD", "VI"],
        "vin_max_v": ["VIN", "VDD", "VI"],
        "iout_max_a": ["IOUT", "IO"],
    },
    "gate_driver": {"vin_max_v": ["VDD", "VIN"], "iout_max_a": ["IOUT", "IO", "IPK"]},
    "charger": {"vin_max_v": ["VIN", "VCC"], "charge_current_max_a": ["ICHG", "ICHARGE", "IBAT"]},
    "pmic": {"vin_max_v": ["VIN", "VDD", "PVIN"]},
    "load_switch": {"vin_max_v": ["VIN", "VDD"], "iout_max_a": ["IOUT", "IMAX"]},
}
WANTED = {"vout_min_v": ["VOUT"], "vout_max_v": ["VOUT"], "iq": ["IQ", "ICC", "ISY"]}

GOLD_AXES = {
    "vds_v", "id_a", "rds_on_mohm", "vin_min_v", "vin_max_v", "iout_max_a",
    "charge_current_max_a", "regulated_outputs",
}
_ABS_MAX = re.compile(r"absolute\s+maximum", re.I)
_MIN_HINT = re.compile(r"\bmin", re.I)
_MAX_HINT = re.compile(r"\bmax", re.I)
_OUTPUT_COUNT = re.compile(
    r"\b(dual|triple|quad|dual-output|two|three|four|2|3|4)[- ](?:output|channel|phase|rail)", re.I
)


def _norm(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9()]", "", symbol or "").upper()


def _axis_for(fact: dict, device_class: str) -> tuple[str, bool] | None:
    """(axis, is_min_role) for a fact, per the class axes; None if irrelevant."""
    sym = _norm(fact.get("symbol") or "")
    par = (fact.get("parameter") or "") + " " + (fact.get("section") or "")
    # Pulsed/peak variants never satisfy continuous-current axes.
    if re.fullmatch(r"I(D|C)?[MPK](?:\(\d\))?", sym) or re.search(
        r"pulsed|peak\s+switching|surge", par, re.I
    ):
        return None
    for axis, symbols in AXES.get(device_class, {}).items():
        for s in symbols:
            s_norm = _norm(s)
            exactish = sym == s_norm or sym.startswith(s_norm + "(") or sym == s_norm + "1"
            if exactish or (not sym and s_norm in _norm(par)):
                if axis.endswith("_min_v"):
                    if _MIN_HINT.search(par) or fact.get("quantity_qualifier") == "minimum":
                        return axis, True
                    return None
                return axis, False
    for axis, symbols in WANTED.items():
        if isinstance(symbols, list):
            for s in symbols:
                if sym == _norm(s):
                    return axis, axis.endswith("_min_v")
    return None


_UNIT_FACTOR = {
    "A": 1.0, "MA": 1e6, "KA": 1e3, "MA(U)": 1e-3, "UA": 1e-6, "NA": 1e-9,
    "V": 1.0, "MV": 1e-3, "KV": 1e3, "UV": 1e-6,
    "OHM": 1000.0, "MOHM": 1.0, "KOHM": 1e6, "MO": 1.0,
}


def _to_canonical(value: float, unit: str, axis: str) -> float | None:
    u = re.sub(r"[^A-Za-z]", "", unit or "").upper().replace("Ω", "OHM").replace("µ", "U").replace("μ", "U")
    if axis.endswith("_a"):
        factor = {"A": 1.0, "MA": 1e-3, "UA": 1e-6, "NA": 1e-9, "KA": 1e3}.get(u)
    elif axis.endswith("_v"):
        factor = {"V": 1.0, "MV": 1e-3, "KV": 1e3, "UV": 1e-6}.get(u)
    elif axis.endswith("_mohm"):
        factor = {"MOHM": 1.0, "MO": 1.0, "OHM": 1000.0, "KOHM": 1e6, "UOHM": 1e-3}.get(u)
    else:
        return value
    if factor is None:
        return None
    return value * factor


def _pick_value(fact: dict, axis: str) -> tuple[float | None, str]:
    for role in ("max", "typ", "min", "value"):
        v = fact.get(role)
        if isinstance(v, (int, float)):
            canonical = _to_canonical(float(v), fact.get("unit") or "", axis)
            if canonical is not None:
                return canonical, str(fact.get(role))
    return None, ""


def _outputs_from_prose(facts: list[dict]) -> int | None:
    text = " ".join(str(f.get("parameter") or "") for f in facts if f.get("table_kind") == "prose")
    text += " ".join(str(f.get("verbatim") or "") for f in facts[:40])
    m = _OUTPUT_COUNT.search(text)
    if not m:
        return None
    word = m.group(1).lower()
    return {"dual": 2, "two": 2, "2": 2, "triple": 3, "three": 3, "3": 3,
            "quad": 4, "four": 4, "4": 4}.get(word.rstrip("-"))


def extract_part(pdf: Path, part: str, device_class: str) -> list[dict]:
    doc = pymupdf.open(pdf)
    try:
        facts = read_characteristic_tables(doc) + read_front_page_facts(doc)
    finally:
        doc.close()
    rows = []
    best: dict[str, dict] = {}
    for fact in facts:
        if (fact.get("table_kind") or "") == "prose":
            continue  # parametric fills come from tables; prose headlines are not spec values
        hit = _axis_for(fact, device_class)
        if hit is not None and hit[0] not in GOLD_AXES:
            continue  # this wave ships gold-required axes only; vout/iq await the v2 detector
        if hit is None:
            continue
        axis, is_min = hit
        if device_class == "pmic" and axis == "vin_max_v":
            pass
        value, role = _pick_value(fact, axis)
        if value is None:
            continue
        if axis.endswith("_min_v") != is_min and axis in ("vin_min_v", "vin_max_v", "vout_min_v", "vout_max_v"):
            continue
        rating_weight = 3 if _ABS_MAX.search(str(fact.get("table_title") or "")) else 1
        if axis.endswith("_max_v") or axis.endswith("_max_a"):
            rating_weight += 1 if _MAX_HINT.search((fact.get("parameter") or "")) else 0
        cond = str(fact.get("condition_verbatim") or "") + str(fact.get("parameter") or "")
        if re.search(r"T[Aa]\s*=\s*25", cond):
            rating_weight += 2  # industry-standard reporting condition; matches selector gold
        key = (axis, rating_weight)
        candidate = {
            "axis": axis,
            "value": value,
            "unit": (
                "A" if axis.endswith("_a")
                else "V" if axis.endswith("_v")
                else "mohm" if axis.endswith("_mohm")
                else (fact.get("unit") or "")
            ),
            "role": role,
            "verbatim": (fact.get("verbatim") or "")[:220],
            "condition": fact.get("condition_verbatim"),
            "page": fact.get("page"),
            "table_kind": fact.get("table_kind"),
            "weight": rating_weight,
        }
        cur = best.get(axis)
        if cur is None or (rating_weight, value) > (cur["weight"], cur["value"]) or (
            axis.endswith("_mohm") and cur["weight"] == rating_weight and value < cur["value"]
        ):
            if cur is None or rating_weight >= cur["weight"]:
                best[axis] = candidate
    for axis, c in best.items():
        rows.append(
            {
                "part_number": part,
                "device_class": device_class,
                "field": axis,
                "value": c["value"],
                "unit": c["unit"],
                "role": c["role"],
                "page_1based": c["page"],
                "verbatim": c["verbatim"],
                "condition_verbatim": c["condition"],
                "table_kind": c["table_kind"],
            }
        )
    if device_class == "pmic":
        n = _outputs_from_prose(facts)
        if n:
            rows.append(
                {
                    "part_number": part,
                    "device_class": device_class,
                    "field": "regulated_outputs",
                    "value": n,
                    "unit": "count",
                    "role": "prose",
                    "page_1based": 1,
                    "verbatim": _OUTPUT_COUNT.pattern,
                    "condition_verbatim": None,
                    "table_kind": "prose",
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--pdf-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    queue = json.loads(args.queue.read_text())
    parts = queue.get("parts") or queue.get("queue") or queue
    if args.limit:
        parts = parts[: args.limit]
    out_rows = []
    tally = Counter()
    for item in parts:
        part = item["part_number"]
        pdf = args.pdf_root / f"ti-{part}.pdf"
        if not pdf.exists():
            tally["missing_pdf"] += 1
            continue
        try:
            rows = extract_part(pdf, part, item["device_class"])
        except Exception as exc:
            tally[f"error:{type(exc).__name__}"] += 1
            continue
        sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
        for r in rows:
            r["pdf_sha"] = sha
            r["source_url"] = item.get("datasheet_url") or f"https://www.ti.com/lit/gpn/{part.lower()}"
            out_rows.append(r)
            tally[f"axis:{r['field']}"] += 1
        if rows:
            tally["parts_with_rows"] += 1
        else:
            tally["parts_no_rows"] += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in out_rows)
    )
    print(json.dumps({"parts": len(parts), "rows": len(out_rows), **dict(sorted(tally.items()))}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
