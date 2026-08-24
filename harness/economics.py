from __future__ import annotations

from rich.console import Console
from rich.table import Table

from harness.config import AppConfig
from harness.report import fmt_money, fmt_seconds
from harness.stats import latency_list, median, percentile
from harness.storage.db import Store


def _run_metrics(cfg: AppConfig, store: Store, run_id: str) -> dict:
    run = store.get_run(run_id)
    if not run:
        raise KeyError(f"Unknown run {run_id}")
    case_runs = store.case_runs(run_id)
    results = store.model_results(run_id)
    frontier_keys = {m.key for m in cfg.models.values() if m.tier >= 4}
    frontier_rows = [r for r in results if r["model_key"] in frontier_keys]
    solved = sum(1 for r in case_runs if r["minimum_model_that_solved"] not in {None, "NONE"})
    n = len(case_runs) or 1
    costs = [r["estimated_cost"] for r in results if r["estimated_cost"] is not None]
    case_latencies = [r["total_escalation_latency_ms"] for r in case_runs if r["total_escalation_latency_ms"] is not None]
    if run["mode"] == "baseline":
        case_latencies = latency_list(results)
    return {
        "run_id": run_id,
        "mode": run["mode"],
        "cases": len(case_runs),
        "success_rate": solved / n,
        "total_cost": sum(costs) if costs else None,
        "median_latency_ms": median(case_latencies),
        "p95_latency_ms": percentile(case_latencies, 95),
        "frontier_calls": len(frontier_rows),
        "frontier_tokens": sum(
            (r["input_tokens"] or 0) + (r["output_tokens"] or 0) for r in frontier_rows
        ),
    }


def compare_economics(
    cfg: AppConfig,
    store: Store,
    console: Console,
    baseline_run_id: str | None = None,
    escalate_run_id: str | None = None,
) -> None:
    baseline_row = store.get_run(baseline_run_id) if baseline_run_id else store.latest_run("baseline")
    escalate_row = store.get_run(escalate_run_id) if escalate_run_id else store.latest_run("escalate")
    if not baseline_row:
        raise RuntimeError("No baseline run found. Run: harness baseline cases/")
    if not escalate_row:
        raise RuntimeError("No escalation run found. Run: harness escalate cases/")

    left = _run_metrics(cfg, store, baseline_row["run_id"])
    right = _run_metrics(cfg, store, escalate_row["run_id"])

    console.print("\n[bold]DIRECT FRONTIER  vs  ESCALATION LADDER[/bold]")
    console.print("=" * 44)
    table = Table()
    table.add_column("Metric")
    table.add_column("Direct frontier")
    table.add_column("Escalation ladder")
    table.add_column("Delta")

    def row(label: str, a, b, fmt) -> None:
        delta = None
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = b - a
        table.add_row(label, fmt(a), fmt(b), fmt(delta) if delta is not None else "-")

    row("Run", left["run_id"], right["run_id"], str)
    row("Success rate", left["success_rate"], right["success_rate"], lambda v: f"{v*100:.1f}%" if v is not None else "-")
    row("Total cost", left["total_cost"], right["total_cost"], fmt_money)
    row("Median latency", left["median_latency_ms"], right["median_latency_ms"], fmt_seconds)
    row("p95 latency", left["p95_latency_ms"], right["p95_latency_ms"], fmt_seconds)
    row("Frontier calls", left["frontier_calls"], right["frontier_calls"], lambda v: "-" if v is None else str(v))
    row("Frontier tokens", left["frontier_tokens"], right["frontier_tokens"], lambda v: "-" if v is None else str(v))
    console.print(table)

    if left["total_cost"] is not None and right["total_cost"] is not None:
        savings = left["total_cost"] - right["total_cost"]
        console.print(f"\nDirect-frontier baseline: {fmt_money(left['total_cost'])}")
        console.print(f"Escalation ladder:        {fmt_money(right['total_cost'])}")
        console.print(f"Savings:                  {fmt_money(savings)}")
    else:
        console.print("\nCost comparison unavailable until config/pricing.yaml is filled.")

    if left["median_latency_ms"] is not None and right["median_latency_ms"] is not None:
        added = (right["median_latency_ms"] - left["median_latency_ms"]) / 1000
        sign = "+" if added >= 0 else ""
        console.print(f"Added latency (median):   {sign}{added:.1f}s")
        console.print("Do not hide latency behind token savings.")
