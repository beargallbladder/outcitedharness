#!/usr/bin/env python3
"""3-Spark chair report. 27B is control. Monsters are on trial.

PASS only. PARTIAL is a miss. Latest result per (case, model) wins.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = yaml.safe_load((ROOT / "config" / "chair.yaml").read_text())
DB = ROOT / "results" / "harness.db"


def latest_verdicts() -> dict[tuple[str, str], str]:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT case_id, model_key, verdict, started_at
        FROM model_results
        ORDER BY started_at ASC, id ASC
        """
    ).fetchall()
    con.close()
    out: dict[tuple[str, str], str] = {}
    for row in rows:
        out[(row["case_id"], row["model_key"])] = row["verdict"]
    return out


def is_pass(verdict: str | None) -> bool:
    return verdict == "PASS"


def main() -> None:
    latest = latest_verdicts()
    control = SPEC["control"]
    challengers = list(SPEC["challengers"])
    frontier = SPEC["frontier"]
    local2 = SPEC.get("local_second")
    buckets: dict[str, list[str]] = SPEC["buckets"]
    cases = [c for group in buckets.values() for c in group]

    def verdict(case: str, model: str) -> str | None:
        return latest.get((case, model))

    control_pass = [c for c in cases if is_pass(verdict(c, control))]
    control_fail = [c for c in cases if not is_pass(verdict(c, control))]
    n = len(cases)

    print("# THREE-SPARK CHAIR REPORT")
    print("27B is CONTROL and is not on trial. PASS only. PARTIAL = miss.")
    print(f"Cases in pack: {n}  (VISION pack: empty — no image cases yet)")
    print(f"27B_PASS_RATE: {len(control_pass)}/{n} = {len(control_pass)/n:.0%}")
    print(f"27B misses ({len(control_fail)}): {', '.join(control_fail)}")
    print()

    print(f"{'model':22} {'Total':8} {'27B-fail solved':18} {'residual':10} {'unique'}")
    rows_out = []
    solved_sets: dict[str, set[str]] = {}
    for key in [control, local2] + challengers + [frontier]:
        if not key:
            continue
        wins = {c for c in cases if is_pass(verdict(c, key))}
        solved_sets[key] = wins
        incr = wins & set(control_fail)
        residual = len(incr) / len(control_fail) if control_fail else 0
        others = set()
        for other, s in solved_sets.items():
            if other != key:
                others |= s
        unique = wins - others - set(control_pass) if key != control else set()
        # unique among challengers+frontier on control-fail
        print(
            f"{key:22} {len(wins):2}/{n:<4} {len(incr):2}/{len(control_fail):<3} "
            f"({len(incr)/n:.0%} incr)  {residual:6.0%}     {len(unique)}"
        )
        rows_out.append((key, wins, incr, unique, residual))

    print()
    print("## By bucket (PASS / n)")
    print(f"{'model':22} " + "  ".join(f"{b:10}" for b in buckets))
    for key in [control] + challengers + [frontier]:
        cells = []
        for b, group in buckets.items():
            if not group:
                cells.append(f"{'n/a':10}")
                continue
            p = sum(1 for c in group if is_pass(verdict(c, key)))
            cells.append(f"{p}/{len(group):<8}")
        print(f"{key:22} " + "  ".join(cells))

    print()
    print("## 27B misses: who solved them")
    for case in control_fail:
        hits = []
        for key in challengers + [frontier, local2]:
            if key and is_pass(verdict(case, key)):
                hits.append(key)
            elif key and verdict(case, key) is None:
                hits.append(f"{key}=GAP")
        print(f"  {case:32} 27B={verdict(case, control) or 'GAP':7}  {', '.join(hits) or 'nobody yet'}")

    print()
    print("## Gaps (no row yet)")
    gaps = []
    for case in cases:
        for key in [control] + challengers + [frontier]:
            if verdict(case, key) is None:
                gaps.append((case, key))
    if not gaps:
        print("  none")
    else:
        for case, key in gaps:
            print(f"  {case}  {key}")

    print()
    print("## Chair rule")
    print("A monster earns THREE Sparks only if residual on 27B-fails is clearly")
    print("above Flash-class (2-box) AND unique wins are not just phrase luck.")
    print("Fitting a 3-node recipe is not enough.")
    print()

    # Recommendation: do not auto-pick. State the numbers.
    best = None
    best_residual = -1.0
    for key, wins, incr, unique, residual in rows_out:
        if key in challengers and residual > best_residual:
            best_residual = residual
            best = key
    print("DO NOT AUTO-DECLARE. Highest residual among cloud monsters:")
    print(f"  {best} residual={best_residual:.0%}  (see table; 3-Spark chair is a human call)")

    # Human-facing writeup lives in results/chair_report.md (do not clobber).


if __name__ == "__main__":
    main()
