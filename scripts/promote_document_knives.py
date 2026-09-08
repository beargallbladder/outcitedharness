#!/usr/bin/env python3
"""Promote document-derived values to knife rows, under the two-source rule.

Tier A  document value agrees with the vendor-feed referee              -> promote
Tier B  document value agrees with an independent printed statement:
          - the vendor's ordering-code scheme decoded from the part number
            (harness.electronics.ordering_codes; validated 100 % against the
            documents by scripts/validate_ordering_codes.py), or
          - a second document (different sha256) printing the same value    -> promote
Tier C  single printed statement, no second source                        -> content only

Only attributes the operator has approved are promoted. Every promoted value
carries its sources and the document receipt. Disagreements with the referee
are held (they live in the agreement report's adjudication file); a
disagreement with the ordering code is held too.

    uv run --python 3.11 python scripts/promote_document_knives.py \
        --records results/family-matrix-st-20260908/records.jsonl \
        --referee .../st_selector_mcu_knives_v0.json --pins .../pin_count_knives_v1.json \
        --attributes code_flash_kb freq_mhz can_count usb pin_count \
        --approved-by "Samson Kim 2026-09-08 10:47" --domain st.com \
        --out results/family-matrix-st-20260908/promoted
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from harness.electronics.ordering_codes import decode, source_label  # noqa: E402
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
ORDERING_KEY = {"code_flash_kb": ("flash_kb",), "pin_count": ("pin_count", "pin_count_nominal")}


def _ordering_code(attribute: str, part: str, ours) -> tuple[str | None, bool]:
    """(source label if the ordering code corroborates, True if it contradicts)."""
    decoded = decode(part)
    for key in ORDERING_KEY.get(attribute, ()):
        if key in decoded:
            if decoded[key] == ours:
                return source_label(decoded, key), False
            if key != "pin_count_nominal":
                return None, True
    return None, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--referee", type=Path, default=None, help="vendor-feed knives (optional; without it only Tier B promotes)")
    parser.add_argument("--pins", type=Path, default=None)
    parser.add_argument("--attributes", nargs="+", required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    referee: dict[str, dict] = _load_referee(args.referee) if args.referee else {}
    if args.pins:
        for part, row in _load_referee(args.pins).items():
            if row.get("domain") and args.domain not in str(row.get("domain")):
                continue
            referee.setdefault(part, {}).update({k: v for k, v in row.items() if k in ("pin_counts", "gpio_count")})

    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]

    # Pass 1: every typed statement per (part, attribute), keyed by document.
    statements: dict[tuple[str, str], list[dict]] = defaultdict(list)
    tally: Counter[str] = Counter()
    for record in records:
        meta = record["_meta"]
        for variant in record["variants"]:
            part = (variant.get("part_number") or "").upper()
            if not part:
                continue
            attributes = dict(record["shared"])
            attributes.update(variant["attributes"])
            for attribute in args.attributes:
                leaf = attributes.get(attribute)
                if not leaf:
                    continue
                ours = _our_value(leaf)
                if leaf.get("qualifier"):
                    # "Up to N" is a bound, not this part's value. It belongs
                    # to the family-grain set (elimination-only), never here.
                    tally[f"{attribute}:qualified_bound_not_a_value"] += 1
                    continue
                statements[(part, attribute)].append({"value": ours, "leaf": leaf, "meta": meta})

    by_part: dict[str, dict] = {}
    held: list[dict] = []

    def _promote(part: str, attribute: str, typed: list[dict], field: str, render: str) -> None:
        first = typed[0]
        ours, leaf, meta = first["value"], first["leaf"], first["meta"]
        sources = ["document:" + meta["document_sha256"][:12]]
        tier = None

        ref = referee.get(part, {})
        if attribute in MAPPING and ref:
            fields, kind = MAPPING[attribute]
            verdict, _ = _compare(kind, ours, leaf, ref, fields)
            if verdict == "agree":
                sources.append("vendor_feed:" + args.referee.name if args.referee else "vendor_feed")
                tier = "A"
            elif verdict == "disagree":
                tally[f"{attribute}:disagree_referee_held"] += 1
                return

        label, contradicts = _ordering_code(attribute, part, ours)
        if contradicts:
            tally[f"{attribute}:disagree_ordering_code_held"] += 1
            held.append({"part_number": part, "attribute": attribute, "reason": "ordering_code_disagrees", "value": ours, "decoded": decode(part), "document": meta["source_path"].rsplit("/", 1)[-1], "page": leaf["receipt"]["page"]})
            return
        if label:
            sources.append(label)
            tier = tier or "B"
        other_docs = {r["meta"]["document_sha256"] for r in typed} - {meta["document_sha256"]}
        if other_docs:
            sources.extend("second_document:" + sha[:12] for sha in sorted(other_docs))
            tier = tier or "B"

        if tier is None:
            tally[f"{attribute}:single_source_held"] += 1
            held.append({"part_number": part, "attribute": attribute, "reason": "single_source", "value": ours if not isinstance(ours, tuple) else list(ours), "document": meta["source_path"].rsplit("/", 1)[-1], "page": leaf["receipt"]["page"]})
            return

        row = by_part.setdefault(part, {"part_number": part, "domain": args.domain, "_provenance": {}})
        key = field
        if render == "set":
            row.setdefault(field, [])
            if ours not in row[field]:
                row[field].append(ours)
            row[field].sort()
            key = f"{field}:{ours}"
        elif render == "range":
            row[field + "_min"], row[field + "_max"] = ours
        else:
            row[field] = ours
        row["_provenance"][key] = {
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

    for (part, attribute), rows in statements.items():
        typed = [r for r in rows if r["value"] is not None]
        if not typed:
            tally[f"{attribute}:not_typed"] += len(rows)
            continue
        field, render = EMIT.get(attribute, (attribute, "number"))
        values = sorted({json.dumps(r["value"], sort_keys=True) for r in typed})
        if len(values) > 1 and render != "set":
            tally[f"{attribute}:cross_document_conflict_held"] += 1
            held.append({"part_number": part, "attribute": attribute, "reason": "documents_disagree", "values": [{"value": r["value"], "document": r["meta"]["source_path"].rsplit("/", 1)[-1], "page": r["leaf"]["receipt"]["page"]} for r in typed]})
            continue
        # Set-valued attributes (pin_counts): each distinct printed value is
        # its own statement (one part, several packages).
        groups = [typed] if render != "set" else [[r for r in typed if json.dumps(r["value"], sort_keys=True) == v] for v in values]
        for group in groups:
            _promote(part, attribute, group, field, render)


    rows = [r for r in by_part.values() if any(k not in ("part_number", "domain", "_provenance") for k in r)]
    rows.sort(key=lambda r: r["part_number"])
    fill = Counter(k for r in rows for k in r if k not in ("part_number", "domain", "_provenance"))
    tiers = Counter(p["tier"] for r in rows for p in r["_provenance"].values())
    out = {
        "contract": "fae-document-derived-knives-v0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": args.approved_by,
        "domain": args.domain,
        "semantics": (
            "Document-derived, two-source rule. Tier A: document agrees with the vendor feed. "
            "Tier B: document agrees with the vendor's printed ordering-code scheme or with a second "
            "document. Values printed as common to the family are attached only to parts the document "
            "names. flash/ram in binary KB. n_can = CAN instances of any kind as printed in the device "
            "table. pin_counts = set of package pin counts printed for the part. has_usb from the "
            "device table row. Every value has _provenance."
        ),
        "source_records": str(args.records),
        "source_records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "referee": args.referee.name if args.referee else None,
        "referee_sha256": hashlib.sha256(args.referee.read_bytes()).hexdigest() if args.referee else None,
        "attributes_approved": args.attributes,
        "n_parts": len(rows),
        "fill": dict(fill),
        "values_by_tier": dict(tiers),
        "tally": dict(sorted(tally.items())),
        "by_part": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.domain.split(".")[0]
    path = args.out / f"document_derived_knives_{stem}_v0.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    (args.out / "held.jsonl").write_text("".join(json.dumps(h) + "\n" for h in held))
    print(json.dumps({"n_parts": len(rows), "fill": dict(fill), "values_by_tier": dict(tiers), "tally": dict(sorted(tally.items()))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
