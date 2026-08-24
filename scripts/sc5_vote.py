#!/usr/bin/env python3
"""Majority-vote a --seeds 5 run (27B-SC5)."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from harness.storage.db import Store
from harness.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", default="dgx_qwen_sc5")
    args = parser.parse_args()
    cfg = load_config()
    store = Store(cfg.settings.db_path)
    rows = [
        r
        for r in store.model_results(args.run)
        if r["model_key"] == args.model
    ]
    by_case: dict[str, list] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)

    print(f"SC5 vote on {args.run} model={args.model}")
    wins = 0
    for case_id, samples in sorted(by_case.items()):
        n = len(samples)
        need = n // 2 + 1
        evaluator = samples[0]["evaluator"]
        if evaluator == "keyword_rubric":
            group_hits: dict[int, int] = defaultdict(int)
            total_groups = 0
            for sample in samples:
                detail = json.loads(sample["evaluation_detail"] or "{}")
                inner = detail.get("detail") or {}
                if isinstance(inner, str):
                    inner = json.loads(inner)
                total_groups = inner.get("groups_total") or total_groups
                for hit in inner.get("hit") or []:
                    group_hits[hit["index"]] += 1
            majority = [i for i in range(total_groups) if group_hits.get(i, 0) >= need]
            verdict = "PASS" if len(majority) == total_groups and total_groups else "PARTIAL"
            print(f"  {case_id}: {verdict}  groups {len(majority)}/{total_groups} majority of {n}")
        else:
            counts = Counter(s["verdict"] for s in samples)
            verdict, k = counts.most_common(1)[0]
            if k < need:
                verdict = "PARTIAL"
            print(f"  {case_id}: {verdict}  modal {counts} n={n}")
        if verdict == "PASS":
            wins += 1
    print(f"SC5 full solves: {wins}/{len(by_case)}")


if __name__ == "__main__":
    main()
