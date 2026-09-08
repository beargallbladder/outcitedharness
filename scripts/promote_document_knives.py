#!/usr/bin/env python3
"""Promote document-derived values to knife rows, under the two-source rule.

Tier A  document value agrees with the vendor-feed referee            -> promote
Tier B  document value agrees with the ordering-code decode           -> promote
        (ST memory letter for flash; package pin count vs pin letter)
Tier C  single printed statement, no second source                    -> content only

Only attributes the operator has approved are promoted. Every promoted value
carries its sources and the document receipt. Disagreements are not touched
here; they live in the agreement report's adjudication file.

    uv run --python 3.11 python scripts/promote_document_knives.py \
        --records results/family-matrix-st-20260908/records.jsonl \
        --referee .../st_selector_mcu_knives_v0.json --pins .../pin_count_knives_v1.json \
        --attributes code_flash_kb freq_mhz can_count usb pin_count \
        --approved-by "Samson Kim 2026-09-08 10:47" \
        --out results/family-matrix-st-20260908/promoted
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from harness.electronics.family_device_matrix import st_flash_code_kb  # noqa: E402
from knife_agreement_report import MAPPING, _compare, _load_referee, _our_value  # noqa: E402

# Our attribute -> CR knife field(s) and how to render the value.
EMIT = {
    "code_flash_kb": ("flash_kb", "number"),
    "sram_kb": ("ram_kb", "number"),
    "freq_mhz": ("freq_mhz", "number"),
    "can_count": ("n_can", "number"),
    "usb": ("has_usb", "boolean"),
    "pin_count": ("pin_counts", "set"),
    "spi_count": ("n_spi", "number"),
    "i2c_count": ("n_i2c", "number"),
    "usart_count": ("n_usart", "number"),
    "uart_count": ("n_uart", "number"),
    "i2s_count": ("n_i2s", "number"),
    "operating_voltage": ("vdd", "range"),
    "temp_range": ("temp_c", "range"),
}

# ST pin-count letter (public ordering scheme), used only as a second source.
ST_PIN_LETTER = {"D": 14, "Y": 20, "F": 20, "G": 28, "K": 32, "T": 36, "S": 44, "C": 48, "U": 63, "R": 64, "M": 81, "O": 90, "V": 100, "Q": 132, "Z": 144, "I": 176, "A": 169, "B": 208, "N": 216, "X": 240}


def _tier_b(attribute: str, part: str, ours) -> str | None:
    """Ordering-code corroboration where the vendor scheme prints it."""
    if attribute == "code_flash_kb":
        expected = st_flash_code_kb(part)
        if expected is not None and ours == expected:
            return "ordering_code_memory_letter"
    if attribute == "pin_count":
        import re
        match = re.match(r"^STM32(?:[A-Z]\d[A-Z0-9]\d|[A-Z]{2}\d{2}|[A-Z]{3}\d{1,2})([A-Z])", part)
        if match and ST_PIN_LETTER.get(match.group(1)) == ours:
            return "ordering_code_pin_letter"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--referee", type=Path, required=True)
    parser.add_argument("--pins", type=Path, default=None)
    parser.add_argument("--attributes", nargs="+", required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--domain", default="st.com")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    referee = _load_referee(args.referee)
    if args.pins:
        for part, row in _load_referee(args.pins).items():
            if row.get("domain") and args.domain not in str(row.get("domain")):
                continue
            referee.setdefault(part, {}).update({k: v for k, v in row.items() if k in ("pin_counts", "gpio_count")})

    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    by_part: dict[str, dict] = {}
    tally: Counter[str] = Counter()
    held: list[dict] = []

    for record in records:
        meta = record["_meta"]
        for variant in record["variants"]:
            part = (variant.get("part_number") or "").upper()
            if not part:
                continue
            attributes = dict(record["shared"])
            attributes.update(variant["attributes"])
            ref = referee.get(part, {})
            for attribute in args.attributes:
                leaf = attributes.get(attribute)
                if not leaf:
                    continue
                ours = _our_value(leaf)
                if ours is None:
                    tally[f"{attribute}:not_typed"] += 1
                    continue
                sources = ["document"]
                tier = None
                if attribute in MAPPING and ref:
                    fields, kind = MAPPING[attribute]
                    verdict, _ = _compare(kind, ours, leaf, ref, fields)
                    if verdict == "agree":
                        sources.append("vendor_feed:" + args.referee.name)
                        tier = "A"
                    elif verdict == "disagree":
                        tally[f"{attribute}:disagree_held"] += 1
                        continue
                if tier is None:
                    second = _tier_b(attribute, part, ours)
                    if second:
                        sources.append(second)
                        tier = "B"
                if tier is None:
                    tally[f"{attribute}:single_source_held"] += 1
                    held.append({"part_number": part, "attribute": attribute, "value": ours if not isinstance(ours, tuple) else list(ours), "document": meta["source_path"].rsplit("/", 1)[-1], "page": leaf["receipt"]["page"]})
                    continue
                field, render = EMIT.get(attribute, (attribute, "number"))
                row = by_part.setdefault(part, {"part_number": part, "domain": args.domain, "_provenance": {}})
                if field in row and row[field] != ours and render != "set":
                    tally[f"{attribute}:cross_document_conflict_held"] += 1
                    row.pop(field, None)
                    row["_provenance"].pop(field, None)
                    continue
                if render == "set":
                    row.setdefault(field, [])
                    if ours not in row[field]:
                        row[field].append(ours)
                elif render == "range":
                    row[field + "_min"], row[field + "_max"] = ours
                else:
                    row[field] = ours
                row["_provenance"][field] = {
                    "tier": tier,
                    "sources": sources,
                    "document_sha256": meta["document_sha256"],
                    "document": meta["source_path"].rsplit("/", 1)[-1],
                    "page": leaf["receipt"]["page"],
                    "row_label": leaf["receipt"]["row_label"],
                    "verbatim": leaf.get("verbatim"),
                    "merged_cell": leaf["receipt"].get("merged_cell", False),
                    "applies_to": leaf["receipt"].get("applies_to", "variant_column"),
                }
                tally[f"{attribute}:promoted_tier_{tier}"] += 1

    rows = [r for r in by_part.values() if any(k not in ("part_number", "domain", "_provenance") for k in r)]
    rows.sort(key=lambda r: r["part_number"])
    fill = Counter(k for r in rows for k in r if k not in ("part_number", "domain", "_provenance"))
    out = {
        "contract": "fae-document-derived-knives-v0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": args.approved_by,
        "semantics": (
            "Document-derived, two-source rule. Tier A: document agrees with the vendor feed. "
            "Tier B: document agrees with the vendor's printed ordering-code scheme. Values printed as "
            "common to the family are attached only to parts the document names. flash/ram in binary KB. "
            "n_can = CAN instances of any kind as printed in the device table. pin_counts = set of package "
            "pin counts printed for the part. has_usb from the device table row. Every value has _provenance."
        ),
        "source_records": str(args.records),
        "source_records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "referee": args.referee.name,
        "referee_sha256": hashlib.sha256(args.referee.read_bytes()).hexdigest(),
        "attributes_approved": args.attributes,
        "n_parts": len(rows),
        "fill": dict(fill),
        "tally": dict(sorted(tally.items())),
        "by_part": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "document_derived_knives_st_v0.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    (args.out / "held_single_source.jsonl").write_text("".join(json.dumps(h) + "\n" for h in held))
    print(json.dumps({"n_parts": len(rows), "fill": dict(fill), "tally": dict(sorted(tally.items()))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
