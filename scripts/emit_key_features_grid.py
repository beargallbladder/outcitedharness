#!/usr/bin/env python3
"""Emit ten-group Key Features grid rows from family-document records.

    uv run --python 3.11 python scripts/emit_key_features_grid.py \
        --records results/family-documents-cr-20260908/records.jsonl \
        --out results/key-features-grid-cr-20260908
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.electronics.key_features_grid import GROUPS, SCHEMA, build_grid  # noqa: E402


def _quantiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    s = sorted(values)
    pick = lambda q: s[min(len(s) - 1, int(q * (len(s) - 1)))]  # noqa: E731
    return {"p25": pick(0.25), "median": pick(0.5), "p75": pick(0.75), "max": s[-1]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True, action="append")
    parser.add_argument("--vendor-inventory", type=Path, action="append", default=[], help="jsonl with {sha256, vendor}; joins vendor for records that lack one")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    vendor_by_sha: dict[str, str] = {}
    for path in args.vendor_inventory:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("vendor"):
                    vendor_by_sha[row["sha256"]] = row["vendor"]

    grids = []
    seen_sha: set[str] = set()
    for path in args.records:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if "error" in record or record["_meta"]["document_sha256"] in seen_sha:
                continue
            seen_sha.add(record["_meta"]["document_sha256"])
            grids.append(build_grid(record, vendor_by_sha))

    rows_path = args.out / "grid_rows.jsonl"
    with rows_path.open("w") as fh:
        for grid in grids:
            for row in grid["rows"]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out / "grids.jsonl").open("w") as fh:
        for grid in grids:
            fh.write(json.dumps(grid, ensure_ascii=False) + "\n")

    populated = Counter(g["groups_populated"] for g in grids)
    by_group = Counter()
    for g in grids:
        by_group.update(k["key"] for k in g["groups"] if k["grid_rows"])
    tiers = Counter(r["tier"] for g in grids for r in g["rows"])
    varies = sum(1 for g in grids for r in g["rows"] if r["varies_by_part"] and r["tier"] == "grid")
    summary = {
        "schema": SCHEMA + ".summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documents": len(grids),
        "meets_six_of_ten": sum(1 for g in grids if g["meets_six_of_ten"]),
        "groups_populated_histogram": {str(k): v for k, v in sorted(populated.items())},
        "documents_with_group_populated": {k: by_group.get(k, 0) for k, _ in GROUPS},
        "rows_by_tier": dict(tiers),
        "grid_rows_varies_by_part": varies,
        "by_vendor_meets_bar": dict(Counter(g["_meta"]["vendor"] for g in grids if g["meets_six_of_ten"])),
        "by_vendor_documents": dict(Counter(g["_meta"]["vendor"] for g in grids)),
        "vendor_basis": dict(Counter(g["_meta"].get("vendor_basis") or "null" for g in grids)),
        "vendor_null_artifacts": sorted(g["_meta"]["source_artifact"] for g in grids if not g["_meta"]["vendor"])[:60],
        "scope_empty": sum(1 for g in grids if not g["_meta"]["scope_as_printed"]),
        "grid_rows_per_document": _quantiles([sum(1 for r in g["rows"] if r["tier"] == "grid") for g in grids]),
        "rows_with_also_printed": sum(1 for g in grids for r in g["rows"] if r.get("also_printed")),
    }
    # Typed facts for the six primary groups, and the progress ledger.
    typed_by_group: Counter = Counter()
    typed_kinds: Counter = Counter()
    docs_with_typed: dict[str, set] = {}
    for g in grids:
        for k, v in g.get("typed_facts_by_group", {}).items():
            typed_by_group[k] += v
            docs_with_typed.setdefault(k, set()).add(g["_meta"]["document_sha256"])
        typed_kinds.update(g.get("typed_fact_kinds", {}))
    qq = Counter()
    for g in grids:
        qq.update({k: v for k, v in g.get("quantity_qualifiers", {}).items() if k})
    summary["typed_facts_by_group"] = dict(typed_by_group)
    summary["documents_with_typed_facts_by_group"] = {k: len(v) for k, v in docs_with_typed.items()}
    summary["typed_fact_kinds"] = dict(typed_kinds.most_common())
    summary["quantity_qualified_grid_rows"] = dict(qq)
    summary["io_by_package_rows"] = sum(1 for g in grids for r in g["rows"] if r.get("section") == "io_by_package")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))

    ledger = args.out.parent / "key-features-grid-progress.jsonl"
    measures = {
        "documents": summary["documents"],
        "meets_six_of_ten": summary["meets_six_of_ten"],
        "grid_rows": summary["rows_by_tier"].get("grid", 0),
        "median_grid_rows": summary["grid_rows_per_document"].get("median"),
        "typed_facts": sum(typed_by_group.values()),
        "typed_facts_by_group": dict(typed_by_group),
        "quantity_qualified_rows": sum(qq.values()),
        "io_by_package_rows": summary["io_by_package_rows"],
        "scope_empty": summary["scope_empty"],
    }
    previous = None
    if ledger.exists():
        lines = [l for l in ledger.read_text().splitlines() if l.strip()]
        if lines:
            previous = json.loads(lines[-1])["measures"]
    delta = {}
    if previous:
        for k, v in measures.items():
            if isinstance(v, (int, float)) and isinstance(previous.get(k), (int, float)):
                delta[k] = v - previous[k]
    entry = {"generated_at": summary["generated_at"], "out": str(args.out), "measures": measures, "delta_vs_previous": delta}
    with ledger.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    summary["progress_delta"] = delta
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
