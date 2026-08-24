from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (p / 100)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


@dataclass
class TierStats:
    key: str
    display_name: str
    short_name: str
    tier: int
    solved: int = 0
    incremental: int = 0
    already_solved_cheaper: int = 0
    still_need_escalation: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)

    @property
    def median_latency_s(self) -> float | None:
        value = median(self.latencies_ms)
        return None if value is None else value / 1000

    @property
    def p95_latency_s(self) -> float | None:
        value = percentile(self.latencies_ms, 95)
        return None if value is None else value / 1000

    @property
    def average_cost(self) -> float | None:
        if not self.costs:
            return None
        return sum(self.costs) / len(self.costs)

    @property
    def total_cost(self) -> float | None:
        if not self.costs:
            return None
        return sum(self.costs)


def tournament_tier_stats(cfg, store, run_id: str) -> tuple[list[TierStats], dict[str, int], int]:
    results = store.model_results(run_id)
    case_runs = store.case_runs(run_id)
    case_ids = sorted({row["case_id"] for row in case_runs})
    n_cases = len(case_ids) or len({row["case_id"] for row in results})

    by_case: dict[str, list] = defaultdict(list)
    for row in results:
        by_case[row["case_id"]].append(row)

    models = sorted(cfg.enabled_models(), key=lambda m: (m.tier, m.key))
    stats = {
        m.key: TierStats(
            key=m.key,
            display_name=m.display_name,
            short_name=m.short_name,
            tier=m.tier,
        )
        for m in models
    }

    min_counts: dict[str, int] = defaultdict(int)
    for case_id, rows in by_case.items():
        passed_keys = {r["model_key"] for r in rows if r["verdict"] == "PASS"}
        cheapest = None
        for model in models:
            row = next((r for r in rows if r["model_key"] == model.key), None)
            if row is None:
                continue
            bucket = stats[model.key]
            if row["latency_ms"] is not None:
                bucket.latencies_ms.append(row["latency_ms"])
            if row["estimated_cost"] is not None:
                bucket.costs.append(row["estimated_cost"])
            if row["verdict"] == "PASS":
                bucket.solved += 1
                cheaper_passed = any(
                    other.key in passed_keys and other.tier < model.tier for other in models
                )
                if cheaper_passed:
                    bucket.already_solved_cheaper += 1
                else:
                    bucket.incremental += 1
                    if cheapest is None:
                        cheapest = model
            else:
                bucket.still_need_escalation += 1
        if cheapest:
            min_counts[cheapest.short_name] += 1
        else:
            min_counts["NONE"] += 1

    # Prefer case_runs if present (handles escalation, where not every model ran).
    if case_runs:
        min_counts = defaultdict(int)
        name_by_key = {m.key: m.short_name for m in models}
        for row in case_runs:
            label = row["minimum_model_that_solved"] or "NONE"
            min_counts[name_by_key.get(label, label)] += 1
        n_cases = len(case_runs)

    return list(stats.values()), dict(min_counts), n_cases


def latency_list(rows: Iterable) -> list[float]:
    return [float(r["latency_ms"]) for r in rows if r["latency_ms"] is not None]
