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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True, action="append")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grids = []
    for path in args.records:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if "error" in record:
                continue
            grids.append(build_grid(record))

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
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
