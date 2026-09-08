#!/usr/bin/env python3
"""Agreement report: document-derived device matrix vs CR's vendor-feed knives.

The document is primary and the feed is a second source; this report scores
the overlap only, per attribute, with n and exclusions, and lists every
disagreement for adjudication. Nothing here promotes a value.

    uv run --python 3.11 python scripts/knife_agreement_report.py \
        --records results/family-matrix-st-20260908/records.jsonl \
        --referee /Volumes/M5_4TB/exports/cr_requests/knives-referee-20260908/st_selector_mcu_knives_v0.json \
        --pins /Volumes/M5_4TB/exports/cr_requests/knives-referee-20260908/pin_count_knives_v1.json \
        --out results/family-matrix-st-20260908/agreement

Referee tri-state (per CR contract): explicit 0/False is ST's "-" and is
compared; a missing key is blank/unknown and is excluded as no_referee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ours -> (referee field(s), comparison kind)
MAPPING: dict[str, tuple[tuple[str, ...], str]] = {
    "code_flash_kb": (("flash_kb",), "number"),
    "sram_kb": (("ram_kb",), "number"),
    "freq_mhz": (("freq_mhz",), "number"),
    "spi_count": (("n_spi",), "count"),
    "i2c_count": (("n_i2c",), "count"),
    "usart_count": (("n_usart",), "count"),
    "uart_count": (("n_uart",), "count"),
    "i2s_count": (("n_i2s",), "count"),
    "can_count": (("n_can_20", "n_can_fd"), "count_sum"),
    "usb": (("has_usb",), "boolean"),
    "operating_voltage": (("vdd_min", "vdd_max"), "range"),
    "temp_range": (("temp_min_c", "temp_max_c"), "range"),
    "gpio_count": (("gpio_count",), "count"),
    "pin_count": (("pin_counts",), "member"),
}


def _load_referee(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = data["by_part"] if isinstance(data, dict) and "by_part" in data else data
    if isinstance(rows, dict):
        rows = list(rows.values())
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        part = str(row.get("part_number", "")).upper()
        if part:
            out.setdefault(part, {}).update({k: v for k, v in row.items() if v is not None and v != ""})
    return out


def _our_value(leaf: dict[str, Any]) -> Any:
    status = leaf.get("status")
    if status == "typed":
        if "typ" in leaf:
            return leaf["typ"]
        return (leaf.get("min"), leaf.get("max"))
    if status == "boolean":
        return leaf["value"]
    return None


def _compare(kind: str, ours: Any, leaf: dict[str, Any], referee: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, Any]:
    """Returns (verdict, referee_value) with verdict in agree/disagree/no_referee."""
    if kind in ("number", "count"):
        if fields[0] not in referee:
            return "no_referee", None
        theirs = referee[fields[0]]
        if isinstance(ours, bool):
            # We hold a presence flag; compare against count > 0.
            try:
                return ("agree" if (float(theirs) > 0) == ours else "disagree"), theirs
            except (TypeError, ValueError):
                return "no_referee", theirs
        try:
            return ("agree" if abs(float(theirs) - float(ours)) < 1e-6 else "disagree"), theirs
        except (TypeError, ValueError):
            return "no_referee", theirs
    if kind == "count_sum":
        present = [f for f in fields if f in referee]
        if not present:
            if "has_can" in referee and isinstance(ours, (int, float)) and not isinstance(ours, bool):
                return ("agree" if bool(referee["has_can"]) == (ours > 0) else "disagree"), referee["has_can"]
            return "no_referee", None
        theirs = sum(float(referee[f]) for f in present)
        if isinstance(ours, bool):
            return ("agree" if (theirs > 0) == ours else "disagree"), theirs
        return ("agree" if abs(theirs - float(ours)) < 1e-6 else "disagree"), theirs
    if kind == "boolean":
        if fields[0] not in referee:
            return "no_referee", None
        theirs = bool(referee[fields[0]])
        if isinstance(ours, bool):
            return ("agree" if theirs == ours else "disagree"), theirs
        return "no_referee", theirs
    if kind == "range":
        if not all(f in referee for f in fields):
            return "no_referee", None
        theirs = (float(referee[fields[0]]), float(referee[fields[1]]))
        if not isinstance(ours, tuple):
            return "no_referee", theirs
        candidates = [tuple(g) for g in leaf.get("grades") or []] or [ours]
        ok = any(abs(theirs[0] - c[0]) < 1e-6 and abs(theirs[1] - c[1]) < 1e-6 for c in candidates)
        return ("agree" if ok else "disagree"), theirs
    if kind == "member":
        if fields[0] not in referee:
            return "no_referee", None
        theirs = referee[fields[0]]
        if not isinstance(theirs, list) or not theirs:
            return "no_referee", theirs
        return ("agree" if ours in theirs else "disagree"), theirs
    return "no_referee", None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--referee", type=Path, required=True)
    parser.add_argument("--pins", type=Path, default=None, help="pin_count knives (pin_counts set, gpio_count)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    referee = _load_referee(args.referee)
    if args.pins:
        for part, row in _load_referee(args.pins).items():
            if row.get("domain") and "st.com" not in str(row.get("domain")):
                continue
            referee.setdefault(part, {}).update({k: v for k, v in row.items() if k in ("pin_counts", "gpio_count")})

    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]

    tally: dict[str, Counter[str]] = defaultdict(Counter)
    adjudication: list[dict[str, Any]] = []
    parts_ours: set[str] = set()
    parts_overlap: set[str] = set()
    values_compared_by_part: Counter[str] = Counter()

    for record in records:
        doc = record["_meta"]["source_path"].rsplit("/", 1)[-1]
        sha = record["_meta"].get("document_sha256")
        for variant in record["variants"]:
            part = variant.get("part_number")
            if not part:
                continue
            part = part.upper()
            parts_ours.add(part)
            ref = referee.get(part)
            if ref is None:
                continue
            parts_overlap.add(part)
            attributes = dict(record["shared"])
            attributes.update(variant["attributes"])
            for attribute, leaf in attributes.items():
                if attribute not in MAPPING:
                    continue
                fields, kind = MAPPING[attribute]
                ours = _our_value(leaf)
                if ours is None:
                    tally[attribute]["ours_unknown" if leaf.get("status") == "unknown" else "ours_verbatim_only"] += 1
                    continue
                verdict, theirs = _compare(kind, ours, leaf, ref, fields)
                tally[attribute][verdict] += 1
                if verdict != "no_referee":
                    values_compared_by_part[part] += 1
                if verdict == "disagree":
                    adjudication.append(
                        {
                            "part_number": part,
                            "attribute": attribute,
                            "document": {"file": doc, "sha256": sha, "page": leaf["receipt"]["page"], "row_label": leaf["receipt"]["row_label"], "verbatim": leaf.get("verbatim"), "typed": ours if not isinstance(ours, tuple) else list(ours), "note": leaf.get("note"), "merged_cell": leaf["receipt"].get("merged_cell", False), "applies_to": leaf["receipt"].get("applies_to")},
                            "referee": {"fields": list(fields), "value": theirs, "source_files": ref.get("source_files")},
                            "resolution": "unknown",
                        }
                    )

    per_attribute: dict[str, dict[str, Any]] = {}
    total_compared = total_agree = 0
    for attribute in MAPPING:
        counts = tally.get(attribute, Counter())
        compared = counts["agree"] + counts["disagree"]
        total_compared += compared
        total_agree += counts["agree"]
        per_attribute[attribute] = {
            "compared": compared,
            "agree": counts["agree"],
            "disagree": counts["disagree"],
            "agreement_rate": round(counts["agree"] / compared, 4) if compared else None,
            "excluded_no_referee": counts["no_referee"],
            "excluded_ours_unknown": counts["ours_unknown"],
            "excluded_ours_verbatim_only": counts["ours_verbatim_only"],
            "passes_95": (counts["agree"] / compared >= 0.95) if compared else None,
        }

    disagreement_kinds = Counter((a["attribute"]) for a in adjudication)
    report = {
        "schema": "harness.electronics-knife-agreement-report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": str(args.records),
        "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "referee": str(args.referee),
        "referee_sha256": hashlib.sha256(args.referee.read_bytes()).hexdigest(),
        "referee_parts": len(referee),
        "parts": {
            "ours_bound": len(parts_ours),
            "overlap_with_referee": len(parts_overlap),
            "ours_not_in_referee_net_new": len(parts_ours - set(referee)),
            "referee_not_in_ours": len(set(referee) - parts_ours),
            "overlap_with_at_least_one_comparison": sum(1 for p in parts_overlap if values_compared_by_part[p]),
        },
        "overall": {
            "compared": total_compared,
            "agree": total_agree,
            "agreement_rate": round(total_agree / total_compared, 4) if total_compared else None,
        },
        "per_attribute": per_attribute,
        "disagreements_by_attribute": dict(disagreement_kinds),
        "adjudication_count": len(adjudication),
        "method": "Overlap only: a value is compared when we typed it from the document and the referee has the key (explicit 0/False counts). Exact equality for numbers and ranges; booleans vs has_*; our presence flags in count rows compared against count>0; pin_count as membership in the referee's pin_counts set. Nothing is promoted by this report.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "agreement-report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "adjudication.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in adjudication))

    lines = [
        f"# Agreement report: document matrix vs referee ({args.referee.name})",
        "",
        f"Parts bound from documents: {report['parts']['ours_bound']}; in referee: {report['parts']['overlap_with_referee']}; "
        f"net new (not in referee): {report['parts']['ours_not_in_referee_net_new']}; referee parts we did not bind: {report['parts']['referee_not_in_ours']}.",
        "",
        f"Overall: {total_agree}/{total_compared} agree ({report['overall']['agreement_rate']}). Adjudication rows: {len(adjudication)}.",
        "",
        "| attribute | compared | agree | disagree | rate | no referee | ours unknown | ours verbatim | >=95% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for attribute, row in per_attribute.items():
        lines.append(
            f"| {attribute} | {row['compared']} | {row['agree']} | {row['disagree']} | {row['agreement_rate']} | {row['excluded_no_referee']} | {row['excluded_ours_unknown']} | {row['excluded_ours_verbatim_only']} | {row['passes_95']} |"
        )
    (args.out / "agreement-report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
