#!/usr/bin/env python3
"""Check every ordering-code decoder against the documents it claims to decode.

For each bound variant with a typed document value for flash or pin count,
compare with ``ordering_codes.decode``. A decoder family below the threshold
is reported and must be fixed or removed; it is never used for promotion.

    uv run --python 3.11 python scripts/validate_ordering_codes.py \
        --records results/family-matrix-*-20260908/records.jsonl --out /tmp/oc.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.electronics.ordering_codes import decode  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    stats: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"agree": 0, "disagree": 0}))
    examples: dict[str, list[dict]] = defaultdict(list)
    decodable = 0
    for path in args.records:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for variant in record["variants"]:
                part = variant.get("part_number")
                if not part:
                    continue
                decoded = decode(part)
                if not decoded:
                    continue
                decodable += 1
                attrs = dict(record["shared"])
                attrs.update(variant["attributes"])
                family = str(decoded.get("scheme"))
                for key, attribute in (("flash_kb", "code_flash_kb"), ("pin_count", "pin_count"), ("pin_count_nominal", "pin_count")):
                    if key not in decoded:
                        continue
                    leaf = attrs.get(attribute)
                    if not leaf or leaf.get("status") != "typed":
                        continue
                    ours = leaf.get("typ")
                    ok = ours == decoded[key]
                    stats[family][key]["agree" if ok else "disagree"] += 1
                    if not ok and len(examples[f"{family}:{key}"]) < 5:
                        examples[f"{family}:{key}"].append({"part": part, "document": ours, "decoded": decoded[key], "verbatim": leaf.get("verbatim"), "file": record["_meta"]["source_path"].rsplit("/", 1)[-1], "page": leaf["receipt"]["page"]})

    report = {"decodable_variants": decodable, "families": {}, "below_threshold": [], "examples": examples}
    print(f"| family | key | agree | disagree | rate | ok |\n|---|---|---:|---:|---:|---|")
    for family in sorted(stats):
        for key, c in sorted(stats[family].items()):
            n = c["agree"] + c["disagree"]
            rate = c["agree"] / n if n else 0.0
            ok = rate >= args.threshold and n >= 5
            report["families"][f"{family}:{key}"] = {**c, "rate": round(rate, 4), "ok": ok}
            if not ok:
                report["below_threshold"].append(f"{family}:{key}")
            print(f"| {family} | {key} | {c['agree']} | {c['disagree']} | {rate:.4f} | {ok} |")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    for key, rows in examples.items():
        print(f"\n{key}")
        for row in rows:
            print("  ", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
