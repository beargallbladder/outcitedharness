#!/usr/bin/env python3
"""Above-OPN facts, emitted at the grain the document prints them.

Contract (CR, `family-grain-contract-correction-20260908`): extraction reports
what the document says, at the grain it says it, with receipts, and does not
interpret. CR decides direction, floors and what becomes a knife.

  grain               part | series | family | group — as PRINTED
  qualifier_verbatim  the exact words bounding the number ("Up to", "max."),
                      or null when the number is bare
  value               typed, with unit
  scope               what the document says the value covers:
                        {"kind": "named_parts", "parts": [...]}      value printed per part column
                        {"kind": "series_token", "token": "STM32F411xC"}  wildcard column header
                        {"kind": "family_label", "label": "..."}      one cell spanning the table
  _meta               document sha256, page, row label, verbatim, merged_cell

Nothing is expanded to members and nothing is withheld for failing to bind:
every typed cell in a wildcard column is handed over, qualifier or not.

    uv run --python 3.11 python scripts/emit_family_grain_knives.py \
        --records results/family-matrix-*-20260908/records.jsonl \
        --out /Volumes/M5_4TB/exports/family-matrix-20260908/above_opn_facts_v0.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.electronics.ordering_codes import ST_MEMORY_LETTER_KB  # noqa: E402

_ST_GD_WILDCARD_FLASH = re.compile(r"^(?:STM32|GD32)[A-Z]{1,2}\d{2,3}x([0-9A-Z])(?:x|$)")
_QUALIFIER_WORDS = re.compile(r"\b(up\s+to|max(?:imum|\.)?|min(?:imum|\.)?|typ(?:ical|\.)?|at\s+least|approx(?:imately|\.)?)\b", re.I)


def _wildcard_flash_kb(token: str) -> int | None:
    match = _ST_GD_WILDCARD_FLASH.match(token or "")
    return ST_MEMORY_LETTER_KB.get(match.group(1)) if match else None


def _typed(leaf: dict) -> dict | None:
    status = leaf.get("status")
    if status == "typed" and "typ" in leaf:
        return {"value": leaf["typ"], "unit": leaf.get("unit")}
    if status == "typed":
        return {"value": {"min": leaf.get("min"), "max": leaf.get("max"), "grades": leaf.get("grades")}, "unit": leaf.get("unit")}
    if status == "boolean":
        return {"value": leaf["value"], "unit": None}
    return None


def _qualifier_verbatim(leaf: dict) -> str | None:
    """The exact printed words, taken from the cell first, then the row label."""
    for text in (leaf.get("verbatim") or "", leaf["receipt"].get("row_label") or ""):
        match = _QUALIFIER_WORDS.search(text)
        if match:
            return match.group(1)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    tally: Counter[str] = Counter()

    def emit(grain: str, scope: dict, attribute: str, leaf: dict, meta: dict, vendor: str | None, extra: dict | None = None) -> None:
        typed = _typed(leaf)
        if typed is None:
            tally["not_typed_skipped"] += 1
            return
        row = {
            "grain": grain,
            "attribute": attribute,
            **typed,
            "qualifier_verbatim": _qualifier_verbatim(leaf),
            "scope": scope,
            "vendor": vendor,
            "_meta": {
                "document_sha256": meta["document_sha256"],
                "document": meta["source_path"].rsplit("/", 1)[-1],
                "page": leaf["receipt"]["page"],
                "row_label": leaf["receipt"]["row_label"],
                "verbatim": leaf.get("verbatim"),
                "merged_cell": leaf["receipt"].get("merged_cell", False),
                "note": leaf.get("note"),
            },
        }
        if extra:
            row.update(extra)
        if attribute == "code_flash_kb" and scope.get("kind") == "series_token":
            decoded = _wildcard_flash_kb(scope["token"])
            if decoded is not None:
                row["_meta"]["ordering_code_memory_letter_kb"] = decoded
        rows.append(row)
        tally[f"{grain}:{'qualified' if row['qualifier_verbatim'] else 'bare'}"] += 1

    for path in args.records:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            meta, overview = record["_meta"], record["overview"]
            vendor = overview.get("vendor")
            # The parts the device table itself binds, not every token the
            # document mentions (register names slip into parts_named).
            named = sorted({v["part_number"] for v in record["variants"] if v.get("part_number")})
            series_tokens = list(overview.get("series_tokens") or [])

            # Values common to every bound column. Printed per part column ->
            # grain part over the named parts; printed once in a cell spanning
            # the table -> the grain the header states.
            for attribute, leaf in record["shared"].items():
                if leaf["receipt"].get("merged_cell") and series_tokens:
                    emit("family", {"kind": "family_label", "label": series_tokens[0]}, attribute, leaf, meta, vendor)
                elif named:
                    emit("part", {"kind": "named_parts", "parts": named}, attribute, leaf, meta, vendor)
                elif series_tokens:
                    emit("family", {"kind": "family_label", "label": series_tokens[0]}, attribute, leaf, meta, vendor)

            # Wildcard columns that never bound to a part: handed over as the
            # series the header prints, whatever the cell says.
            for variant in record["variants"]:
                if variant.get("part_number"):
                    continue
                token = variant.get("family_token")
                if not token:
                    continue
                for attribute, leaf in variant["attributes"].items():
                    scope = {"kind": "series_token", "token": token, "page": variant.get("page"), "column_index": variant.get("column_index")}
                    emit("series", scope, attribute, leaf, meta, vendor, {"column_binding": variant.get("reason") or variant.get("binding")})

    rows.sort(key=lambda r: (r["vendor"] or "", r["grain"], json.dumps(r["scope"], sort_keys=True), r["attribute"]))
    payload = {
        "schema": "harness.electronics-above-opn-facts.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contract": "CR family-grain-contract-correction-20260908: grain as printed, qualifier verbatim or null, scope as stated, no expansion, no withholding, no direction",
        "n_records": len(rows),
        "tally": dict(sorted(tally.items())),
        "records": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"n_records": len(rows), "tally": payload["tally"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
