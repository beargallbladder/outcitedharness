from __future__ import annotations

from rich.console import Console
from rich.table import Table

from harness.config import AppConfig
from harness.health import HealthRow
from harness.runner import ModelAttempt
from harness.stats import median, tournament_tier_stats
from harness.storage.db import Store


def fmt_seconds(ms: float | None) -> str:
    if ms is None:
        return "-"
    return f"{ms / 1000:.1f}s"


def fmt_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:.4f}"


def print_health(rows: list[HealthRow], console: Console) -> None:
    table = Table(title="Service health")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Endpoint")
    table.add_column("Detail")
    for row in rows:
        style = {
            "OK": "green",
            "DISABLED": "dim",
            "UNCONFIGURED": "yellow",
            "MISSING_KEY": "yellow",
            "FAIL": "red",
            "ERROR": "red",
        }.get(row.status, "")
        table.add_row(row.display_name, f"[{style}]{row.status}[/{style}]", row.endpoint, row.detail)
    console.print(table)


def print_attempts(case_id: str, attempts: list[ModelAttempt], console: Console, preview: int) -> None:
    table = Table(title=f"{case_id}")
    table.add_column("Model")
    table.add_column("Verdict")
    table.add_column("Correct")
    table.add_column("Format")
    table.add_column("Rubric")
    table.add_column("Latency")
    table.add_column("In tok")
    table.add_column("Out tok")
    table.add_column("Cost")
    table.add_column("Error")
    for attempt in attempts:
        result = attempt.result
        ev = attempt.evaluation
        correct = "-" if ev.correctness_ok is None else ("yes" if ev.correctness_ok else "no")
        fmt = "-" if ev.format_ok is None else ("clean" if ev.format_ok else "chatty")
        rubric = ""
        if ev.detail.get("groups_total"):
            rubric = f"{ev.detail.get('groups_hit', 0)}/{ev.detail['groups_total']}"
        table.add_row(
            attempt.model.display_name,
            ev.verdict,
            correct,
            fmt,
            rubric,
            fmt_seconds(result.latency_ms),
            "-" if result.input_tokens is None else str(result.input_tokens),
            "-" if result.output_tokens is None else str(result.output_tokens),
            fmt_money(attempt.estimated_cost),
            (result.error or "")[:80],
        )
    console.print(table)
    for attempt in attempts:
        console.print(f"\n[bold]{attempt.model.display_name}[/bold]  {attempt.evaluation.reason}")
        text = (attempt.result.text or "").strip()
        if text:
            shown = text if len(text) <= preview else text[:preview] + "\n..."
            console.print(shown)
        elif attempt.result.error:
            console.print(f"[red]{attempt.result.error}[/red]")


def print_tournament_summary(cfg: AppConfig, store: Store, run_id: str, console: Console) -> None:
    stats, min_counts, n_cases = tournament_tier_stats(cfg, store, run_id)
    console.print("\n[bold]MODEL TOURNAMENT[/bold]")
    console.print("=" * 33)
    console.print(f"Cases: {n_cases}\n")

    table = Table()
    table.add_column("Model")
    table.add_column("Solved", justify="right")
    table.add_column("Incremental", justify="right")
    table.add_column("Already cheaper", justify="right")
    table.add_column("Still escalate", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("Avg cost", justify="right")
    table.add_column("Total cost", justify="right")
    for row in stats:
        table.add_row(
            row.display_name,
            f"{row.solved}/{n_cases}",
            str(row.incremental),
            str(row.already_solved_cheaper),
            str(row.still_need_escalation),
            "-" if row.median_latency_s is None else f"{row.median_latency_s:.1f}s",
            "-" if row.p95_latency_s is None else f"{row.p95_latency_s:.1f}s",
            fmt_money(row.average_cost),
            fmt_money(row.total_cost),
        )
    console.print(table)

    console.print("\n[bold]Minimum tier[/bold]")
    order = [s.short_name for s in stats] + ["NONE"]
    seen = set()
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        count = min_counts.get(name, 0)
        pct = (count / n_cases * 100) if n_cases else 0
        if count or name == "NONE":
            console.print(f"{name:<16} {count}/{n_cases}  {pct:5.1f}%")


def print_escalation_summary(cfg: AppConfig, store: Store, run_id: str, console: Console) -> None:
    case_runs = store.case_runs(run_id)
    results = store.model_results(run_id)
    n = len(case_runs)
    frontier_keys = {m.key for m in cfg.enabled_models() if m.tier >= 4}
    frontier_short = {m.short_name for m in cfg.enabled_models() if m.tier >= 4}

    frontier_required = sum(
        1
        for row in case_runs
        if row["minimum_model_that_solved"] in frontier_short
        or row["minimum_model_that_solved"] in frontier_keys
    )
    solved = sum(1 for row in case_runs if row["minimum_model_that_solved"] not in {None, "NONE"})
    avoided = solved - frontier_required
    time_to_sol = [r["total_escalation_latency_ms"] for r in case_runs if r["minimum_model_that_solved"] not in {None, "NONE"}]
    waste = [
        r["wasted_latency_before_success_ms"]
        for r in case_runs
        if r["minimum_model_that_solved"] not in {None, "NONE"}
        and r["wasted_latency_before_success_ms"] is not None
    ]

    console.print("\n[bold]ESCALATION ECONOMICS[/bold]")
    console.print("=" * 33)
    console.print(f"Frontier required:        {frontier_required} / {n}")
    console.print(f"Frontier avoided:         {max(avoided, 0)} / {n}")
    console.print(f"Unsolved:                 {n - solved} / {n}")
    med_sol = median(time_to_sol)
    med_waste = median(waste)
    console.print(f"Median time to solution:  {fmt_seconds(med_sol)}")
    console.print(f"Median wasted pre-tier:   {fmt_seconds(med_waste)}")

    console.print("\nCloud calls avoided because a cheaper tier already solved:")
    for model in cfg.enabled_models():
        if model.tier < 2:
            continue
        skipped = sum(
            1
            for row in case_runs
            if row["minimum_model_that_solved"] not in {None, "NONE"}
            and (
                (row["successful_tier"] is not None and row["successful_tier"] < model.tier)
            )
            and model.key not in {
                r["model_key"]
                for r in results
                if r["case_id"] == row["case_id"]
            }
        )
        console.print(f"  {model.display_name}: {skipped}  (ran on {sum(1 for r in results if r['model_key']==model.key)} cases)")

    costs = [r["total_escalation_cost"] for r in case_runs if r["total_escalation_cost"] is not None]
    if costs:
        console.print(f"\nEstimated ladder cost: {fmt_money(sum(costs))}")
    else:
        console.print("\nEstimated ladder cost: n/a (fill config/pricing.yaml)")


def print_runs(store: Store, console: Console, limit: int = 20) -> None:
    rows = store.list_runs(limit)
    if not rows:
        console.print("No runs stored yet.")
        return
    table = Table(title="Recent runs")
    table.add_column("Run ID")
    table.add_column("Mode")
    table.add_column("Cases")
    table.add_column("Started")
    table.add_column("Finished")
    for row in rows:
        table.add_row(
            row["run_id"],
            row["mode"],
            str(row["case_count"]),
            row["started_at"] or "",
            row["finished_at"] or "",
        )
    console.print(table)


def print_optimize_report(report, console: Console) -> None:
    from harness.optimize import OptimizeReport, WorkerShot

    if not isinstance(report, OptimizeReport):
        raise TypeError("expected OptimizeReport")

    health = Table(title=f"Optimize {report.run_id} — health")
    health.add_column("Box")
    health.add_column("Status")
    for key, detail in report.health.items():
        style = "green" if detail == "ok" else "red"
        health.add_row(key, f"[{style}]{detail}[/{style}]")
    console.print(health)

    table = Table(title="Worker shots")
    table.add_column("Case")
    table.add_column("Box")
    table.add_column("ms")
    table.add_column("in/out")
    table.add_column("tok/s")
    table.add_column("tools")
    table.add_column("hit")
    table.add_column("err")
    for outcome in report.outcomes:
        for shot in outcome.shots:
            tokens = f"{shot.result.input_tokens or 0}/{shot.result.output_tokens or 0}"
            rate = f"{shot.tokens_per_sec:.1f}" if shot.tokens_per_sec else "-"
            table.add_row(
                outcome.case.id,
                shot.model_key,
                f"{shot.result.latency_ms:.0f}",
                tokens,
                rate,
                ",".join(shot.tool_names) or "-",
                "yes" if shot.tool_hit else "no",
                (shot.result.error or "")[:40],
            )
    console.print(table)

    totals = Table(title="Box totals")
    totals.add_column("Box")
    totals.add_column("hits")
    totals.add_column("mean ms")
    totals.add_column("mean tok/s")
    totals.add_column("out tok")
    by_key: dict[str, list[WorkerShot]] = {}
    for outcome in report.outcomes:
        for shot in outcome.shots:
            by_key.setdefault(shot.model_key, []).append(shot)
    for key, shots in by_key.items():
        hits = sum(1 for s in shots if s.tool_hit)
        mean_ms = sum(s.result.latency_ms for s in shots) / len(shots)
        rates = [s.tokens_per_sec for s in shots if s.tokens_per_sec]
        mean_rate = (sum(rates) / len(rates)) if rates else None
        out = sum(s.result.output_tokens or 0 for s in shots)
        totals.add_row(
            key,
            f"{hits}/{len(shots)}",
            f"{mean_ms:.0f}",
            f"{mean_rate:.1f}" if mean_rate else "-",
            str(out),
        )
    console.print(totals)

    for outcome in report.outcomes:
        winner = outcome.winner or "-"
        console.print(
            f"{outcome.case.id}: winner=[bold]{winner}[/bold]  "
            f"ranks={outcome.ranks or '-'}  {outcome.reason[:120]}"
        )
    if report.senior_text:
        console.print("\n[bold]Senior[/bold]")
        console.print(report.senior_text)
    if report.json_path:
        console.print(f"\nStored [bold]{report.json_path}[/bold]")


def print_dispatch_report(report, console: Console) -> None:
    from harness.dispatch import DispatchReport

    if not isinstance(report, DispatchReport):
        raise TypeError("expected DispatchReport")

    health = Table(title=f"Dispatch {report.run_id} — health")
    health.add_column("Box")
    health.add_column("Status")
    for key, detail in report.health.items():
        style = "green" if detail == "ok" else "red"
        health.add_row(key, f"[{style}]{detail}[/{style}]")
    console.print(health)
    console.print(f"intent: {report.intent}")
    if report.slice_error:
        console.print(f"[red]QA FAIL closed:[/red] {report.slice_error}")
    console.print(f"packets: {len(report.packets)}  shots: {len(report.shots)}")

    table = Table(title="Pool shots")
    table.add_column("Packet")
    table.add_column("Worker")
    table.add_column("ms")
    table.add_column("in/out")
    table.add_column("tok/s")
    table.add_column("tools")
    table.add_column("hit")
    table.add_column("qa")
    table.add_column("err")
    for shot in report.shots:
        tokens = f"{shot.result.input_tokens or 0}/{shot.result.output_tokens or 0}"
        rate = f"{shot.tokens_per_sec:.1f}" if shot.tokens_per_sec else "-"
        table.add_row(
            shot.packet.id,
            shot.worker_id,
            f"{shot.result.latency_ms:.0f}",
            tokens,
            rate,
            ",".join(shot.tool_names) or "-",
            "yes" if shot.tool_hit else "no",
            "yes" if shot.qa_pass else "no",
            (shot.result.error or "")[:40],
        )
    console.print(table)
    qa = sum(1 for s in report.shots if s.qa_pass)
    console.print(f"qa: {qa}/{len(report.shots)} pass  verdict: {report.critic_verdict or 'python'}")
    if report.critic_text:
        console.print("\n[bold]Critic[/bold]")
        console.print(report.critic_text[:800])
    if report.senior_text:
        console.print("\n[bold]Senior[/bold]")
        console.print(report.senior_text)
    if report.json_path:
        console.print(f"\nStored [bold]{report.json_path}[/bold]")
