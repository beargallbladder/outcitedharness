#!/usr/bin/env python3
"""Count verified admitted training pairs and report the pin training gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

PIN_CAPABILITIES = {"pin_or_ball", "pin_semantics"}
TRAINING_GATE = 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--gate", type=int, default=TRAINING_GATE)
    args = parser.parse_args()

    sft = Counter()
    dpo = Counter()
    sources = []
    for pairs_path in sorted(
        args.results_root.glob("**/training-pairs.jsonl")
    ):
        if pairs_path.is_symlink():
            continue
        counted = 0
        with pairs_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("disposition") != "admitted":
                    continue
                sft[str(row.get("capability"))] += 1
                counted += 1
        preference = pairs_path.with_name("preference-training-pairs.jsonl")
        if preference.is_file() and not preference.is_symlink():
            with preference.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        dpo[str(row.get("capability"))] += 1
        if counted:
            sources.append({"path": str(pairs_path), "admitted_sft": counted})

    pin_sft = sum(sft[name] for name in PIN_CAPABILITIES)
    pin_dpo = sum(dpo[name] for name in PIN_CAPABILITIES)
    print(
        json.dumps(
            {
                "admitted_sft_by_capability": dict(sorted(sft.items())),
                "dpo_by_capability": dict(sorted(dpo.items())),
                "pin_sft_pairs": pin_sft,
                "pin_dpo_pairs": pin_dpo,
                "pin_training_gate": args.gate,
                "pin_training_authorized": pin_sft >= args.gate,
                "sources": sources,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
