from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from harness.baseline import run_baseline
from harness.cases.loader import discover_cases
from harness.config import load_config
from harness.economics import compare_economics
from harness.escalation import run_escalation
from harness.health import check_all
from harness.report import (
    fmt_money,
    fmt_seconds,
    print_attempts,
    print_escalation_summary,
    print_health,
    print_runs,
    print_tournament_summary,
)
from harness.storage.db import Store
from harness.rescue import PACKET_TEMPLATE, PacketError, run_rescue
from harness.serial import discover_tickets, run_serial
from harness.task.models import AttemptRecord, Decision
from harness.task.search import search_code
from harness.task.service import TaskService
from harness.tournament import run_tournament
from harness.workers.registry import load_registry


app = typer.Typer(help="Local/cloud model tournament harness", no_args_is_help=True)
console = Console()


def _cfg():
    return load_config()


def _path(value: str) -> Path:
    return Path(value).expanduser()


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, help="Bind host (default from config/cline.yaml)"),
    port: Optional[int] = typer.Option(None, help="Bind port (default 8787)"),
) -> None:
    """OpenAI-compatible gateway for Cline / VS Code. Does not start model servers."""
    from harness.gateway.server import serve as serve_gateway

    serve_gateway(host=host, port=port)


@app.command()
def health() -> None:
    """Probe configured model endpoints. Never modifies running services."""
    cfg = _cfg()
    rows = asyncio.run(check_all(cfg))
    print_health(rows, console)


@app.command()
def workers() -> None:
    """Show the worker registry. Disabled future nodes report unavailable."""
    cfg = _cfg()
    registry = load_registry(cfg.root)
    table = Table(title="Worker registry")
    table.add_column("Worker")
    table.add_column("Status")
    table.add_column("Model")
    table.add_column("Endpoint")
    table.add_column("Detail")
    for row in registry.summary():
        status = row["status"]
        style = {"healthy": "green", "unavailable": "dim", "unconfigured": "yellow"}.get(status, "")
        table.add_row(
            row["id"],
            f"[{style}]{status}[/{style}]" if style else status,
            row["model_key"] or "-",
            row["endpoint"] or "-",
            row["detail"] or "",
        )
    console.print(table)
    chain = registry.failover_keys()
    if chain:
        console.print("auto failover: " + " → ".join(chain))


@app.command()
def tournament(
    path: str = typer.Argument(..., help="Case directory or folder of cases"),
    seeds: int = typer.Option(1, help="Independent repeats with seed=0..N-1"),
    only: Optional[str] = typer.Option(
        None,
        help="Comma-separated model keys to run (disabled keys allowed)",
    ),
) -> None:
    """Send the same task packet to every enabled model independently."""
    cfg = _cfg()
    cases = discover_cases(_path(path))
    seed_list = list(range(seeds)) if seeds > 1 else None
    only_keys = [k.strip() for k in only.split(",") if k.strip()] if only else None
    outcome = asyncio.run(run_tournament(cfg, cases, seeds=seed_list, only=only_keys))
    store = Store(cfg.settings.db_path)
    for case in outcome.cases:
        print_attempts(
            case.id,
            outcome.attempts[case.id],
            console,
            cfg.settings.max_answer_preview_chars,
        )
        summary = store.case_runs(outcome.run_id)
        row = next(r for r in summary if r["case_id"] == case.id)
        console.print(
            f"\nminimum_model_that_solved: [bold]{row['minimum_model_that_solved']}[/bold]"
        )
    print_tournament_summary(cfg, store, outcome.run_id, console)
    console.print(f"\nStored run [bold]{outcome.run_id}[/bold]")


@app.command()
def serial(
    path: str = typer.Argument(..., help="Serial ticket directory or folder of tickets"),
    only: Optional[str] = typer.Option(
        None,
        help="Comma-separated model keys (disabled keys allowed)",
    ),
) -> None:
    """Multi-turn read/edit/run loop against an isolated repo checkout."""
    cfg = _cfg()
    tickets = discover_tickets(_path(path))
    only_keys = [k.strip() for k in only.split(",") if k.strip()] if only else None
    run_id = asyncio.run(run_serial(cfg, tickets, only=only_keys))
    store = Store(cfg.settings.db_path)
    table = Table(title=f"Serial {run_id}")
    table.add_column("Ticket")
    table.add_column("Model")
    table.add_column("Verdict")
    table.add_column("Turns")
    table.add_column("Reason")
    for row in store.model_results(run_id):
        detail = row["evaluation_detail"]
        if isinstance(detail, str):
            import json

            detail = json.loads(detail or "{}")
        table.add_row(
            row["case_id"],
            row["model_key"],
            row["verdict"],
            str((detail or {}).get("turns", "")),
            str((detail or {}).get("reason", ""))[:80],
        )
    console.print(table)
    for row in store.case_runs(run_id):
        console.print(
            f"{row['case_id']}: minimum_model_that_solved="
            f"[bold]{row['minimum_model_that_solved']}[/bold]"
        )
    console.print(f"\nStored run [bold]{run_id}[/bold]")


@app.command()
def rescue(
    packet: Optional[str] = typer.Argument(None, help="Markdown packet path"),
    model: str = typer.Option("frontier", help="Senior model key"),
    template: bool = typer.Option(False, "--template", help="Print the packet skeleton"),
) -> None:
    """Send a constructed fail packet to frontier. Do not paste a Cline thread."""
    if template or not packet:
        console.print(PACKET_TEMPLATE)
        raise typer.Exit(code=0)
    cfg = _cfg()
    try:
        outcome = asyncio.run(run_rescue(cfg, _path(packet), model_key=model))
    except PacketError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    if outcome.error:
        console.print(f"[red]{outcome.error}[/red]")
        raise typer.Exit(code=1)
    console.print(outcome.text)
    console.print(
        f"\nStored run [bold]{outcome.run_id}[/bold]  "
        f"{outcome.model_key}  {outcome.latency_ms/1000:.1f}s  "
        f"{outcome.answer_path}"
    )


@app.command()
def escalate(
    path: str = typer.Argument(..., help="Case directory or folder of cases"),
) -> None:
    """Walk the model ladder and stop at the first evaluator PASS."""
    cfg = _cfg()
    cases = discover_cases(_path(path))
    outcome = asyncio.run(run_escalation(cfg, cases))
    store = Store(cfg.settings.db_path)
    for case in outcome.cases:
        print_attempts(
            case.id,
            outcome.attempts[case.id],
            console,
            cfg.settings.max_answer_preview_chars,
        )
        row = next(r for r in store.case_runs(outcome.run_id) if r["case_id"] == case.id)
        console.print(
            f"\nminimum_model_that_solved: [bold]{row['minimum_model_that_solved']}[/bold]"
        )
        console.print(
            f"local_waste_ms: {row['wasted_latency_before_success_ms']:.0f}  "
            f"failed_tiers: {row['failed_tiers']}"
        )
    print_escalation_summary(cfg, store, outcome.run_id, console)
    console.print(f"\nStored run [bold]{outcome.run_id}[/bold]")


@app.command()
def benchmark(
    path: str = typer.Argument("cases", help="Folder of cases"),
) -> None:
    """Run a tournament across every case in a folder and print the distribution."""
    cfg = _cfg()
    cases = discover_cases(_path(path))
    outcome = asyncio.run(run_tournament(cfg, cases, mode="benchmark"))
    store = Store(cfg.settings.db_path)
    print_tournament_summary(cfg, store, outcome.run_id, console)
    console.print(f"\nStored run [bold]{outcome.run_id}[/bold]")


@app.command()
def baseline(
    path: str = typer.Argument("cases", help="Folder of cases"),
) -> None:
    """Send every case directly to the configured frontier-tier model."""
    cfg = _cfg()
    cases = discover_cases(_path(path))
    outcome = asyncio.run(run_baseline(cfg, cases))
    store = Store(cfg.settings.db_path)
    print_tournament_summary(cfg, store, outcome.run_id, console)
    console.print(f"\nStored run [bold]{outcome.run_id}[/bold]")


@app.command()
def results() -> None:
    """List recent stored runs."""
    cfg = _cfg()
    print_runs(Store(cfg.settings.db_path), console)


@app.command()
def inspect(run_id: str = typer.Argument(..., help="Run ID from harness results")) -> None:
    """Show one stored run in detail."""
    cfg = _cfg()
    store = Store(cfg.settings.db_path)
    run = store.get_run(run_id)
    if not run:
        raise typer.BadParameter(f"Unknown run {run_id}")
    console.print(f"[bold]{run['run_id']}[/bold]  mode={run['mode']}  cases={run['case_count']}")
    if run["mode"] in {"tournament", "benchmark"}:
        print_tournament_summary(cfg, store, run_id, console)
    if run["mode"] == "escalate":
        print_escalation_summary(cfg, store, run_id, console)

    table = Table(title="Case outcomes")
    table.add_column("Case")
    table.add_column("Min model")
    table.add_column("Waste")
    table.add_column("Failed tiers")
    table.add_column("Total time")
    for row in store.case_runs(run_id):
        table.add_row(
            row["case_id"],
            row["minimum_model_that_solved"] or "NONE",
            fmt_seconds(row["wasted_latency_before_success_ms"]),
            str(row["failed_tiers"] if row["failed_tiers"] is not None else "-"),
            fmt_seconds(row["total_escalation_latency_ms"]),
        )
    console.print(table)

    detail = Table(title="Model results")
    detail.add_column("Case")
    detail.add_column("Model")
    detail.add_column("Verdict")
    detail.add_column("Latency")
    detail.add_column("Tokens")
    detail.add_column("Cost")
    detail.add_column("Error")
    for row in store.model_results(run_id):
        tokens = "-"
        if row["input_tokens"] is not None or row["output_tokens"] is not None:
            tokens = f"{row['input_tokens'] or 0}/{row['output_tokens'] or 0}"
        detail.add_row(
            row["case_id"],
            row["model_key"],
            row["verdict"] or "",
            fmt_seconds(row["latency_ms"]),
            tokens,
            fmt_money(row["estimated_cost"]),
            (row["error"] or "")[:60],
        )
    console.print(detail)


@app.command()
def compare(case_id: str = typer.Argument(..., help="Case ID, e.g. example_001")) -> None:
    """Compare every stored attempt for one case."""
    cfg = _cfg()
    store = Store(cfg.settings.db_path)
    rows = store.results_for_case(case_id)
    if not rows:
        console.print(f"No stored results for {case_id}")
        raise typer.Exit(code=1)
    table = Table(title=f"Compare {case_id}")
    table.add_column("Run")
    table.add_column("Model")
    table.add_column("Verdict")
    table.add_column("Latency")
    table.add_column("Tokens")
    table.add_column("Cost")
    for row in rows:
        tokens = "-"
        if row["input_tokens"] is not None or row["output_tokens"] is not None:
            tokens = f"{row['input_tokens'] or 0}/{row['output_tokens'] or 0}"
        table.add_row(
            row["run_id"],
            row["model_key"],
            row["verdict"] or "",
            fmt_seconds(row["latency_ms"]),
            tokens,
            fmt_money(row["estimated_cost"]),
        )
    console.print(table)


@app.command()
def economics(
    baseline_run: Optional[str] = typer.Option(None, help="Baseline run ID"),
    escalate_run: Optional[str] = typer.Option(None, help="Escalation run ID"),
) -> None:
    """Compare direct-frontier baseline against the escalation ladder."""
    cfg = _cfg()
    compare_economics(cfg, Store(cfg.settings.db_path), console, baseline_run, escalate_run)


@app.command()
def judge(
    run_id: str = typer.Argument(...),
    case_id: str = typer.Argument(...),
    model_key: str = typer.Argument(...),
    verdict: str = typer.Argument(..., help="PASS, PARTIAL, or FAIL"),
    reason: str = typer.Option("", help="Optional reviewer note"),
) -> None:
    """Assign a human verdict and recompute minimum_model_that_solved."""
    verdict = verdict.upper()
    if verdict not in {"PASS", "PARTIAL", "FAIL"}:
        raise typer.BadParameter("verdict must be PASS, PARTIAL, or FAIL")
    cfg = _cfg()
    store = Store(cfg.settings.db_path)
    store.update_verdict(run_id, case_id, model_key, verdict, reason)
    short_names = {m.key: m.short_name for m in cfg.models.values()}
    store.recompute_case_run(run_id, case_id, short_names=short_names)
    row = next(r for r in store.case_runs(run_id) if r["case_id"] == case_id)
    console.print(
        f"Updated {run_id}/{case_id}/{model_key} -> {verdict}. "
        f"minimum_model_that_solved={row['minimum_model_that_solved']}"
    )


task_app = typer.Typer(help="Task / evidence log for real jobs (not a chat dump).")


@task_app.command("start")
def task_start(intent: str = typer.Argument(..., help="What you are actually doing")) -> None:
    cfg = _cfg()
    task = TaskService(Store(cfg.settings.db_path)).start(intent)
    console.print(task.task_id)
    console.print(task.intent)


@task_app.command("list")
def task_list(limit: int = typer.Option(20)) -> None:
    cfg = _cfg()
    rows = TaskService(Store(cfg.settings.db_path)).list_tasks(limit=limit)
    table = Table(title="Tasks")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Intent")
    for t in rows:
        table.add_row(t.task_id, t.status, t.intent[:80])
    console.print(table)


@task_app.command("show")
def task_show(task_id: str) -> None:
    import json

    cfg = _cfg()
    svc = TaskService(Store(cfg.settings.db_path))
    task = svc.get(task_id)
    attempts = svc.attempts(task_id)
    console.print(f"{task.task_id}  {task.status}  {task.intent}")
    for rec in attempts:
        console.print(json.dumps(rec.to_evidence_json(), indent=2))


@task_app.command("packet")
def task_packet(
    task_id: str,
    worker: str = typer.Option("primary_coder"),
) -> None:
    cfg = _cfg()
    packet = TaskService(Store(cfg.settings.db_path)).packet(task_id, worker)
    console.print(packet.to_markdown())


@task_app.command("record")
def task_record(
    task_id: str,
    worker: str = typer.Option("primary_coder"),
    result: str = typer.Option("success"),
    files: str = typer.Option("", help="Comma-separated paths"),
    command: str = typer.Option("", help="Command that was run"),
    passed: Optional[int] = typer.Option(None),
    failed: Optional[int] = typer.Option(None),
    tool_calls: Optional[int] = typer.Option(None),
) -> None:
    cfg = _cfg()
    rec = AttemptRecord(
        task_id=task_id,
        attempt=0,
        worker=worker,
        result=result,
        files_changed=[p.strip() for p in files.split(",") if p.strip()],
        commands=[command] if command else [],
        tests_passed=passed,
        tests_failed=failed,
        tool_calls=tool_calls,
    )
    saved = TaskService(Store(cfg.settings.db_path)).record(rec)
    console.print(json_dumps(saved.to_evidence_json()))


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, indent=2)


@task_app.command("decide")
def task_decide(
    task_id: str,
    text: str,
    actor: str = typer.Option("fallback_reasoner"),
    accepted: bool = typer.Option(True),
) -> None:
    cfg = _cfg()
    TaskService(Store(cfg.settings.db_path)).add_decision(
        Decision(task_id=task_id, actor=actor, text=text, accepted=accepted)
    )


@app.command("search")
def search_cmd(
    query: str,
    repo: str = typer.Option(".", help="Repo root"),
    mode: str = typer.Option("auto", help="auto|grep|ast|semantic|hybrid"),
) -> None:
    result = search_code(query, _path(repo), mode=mode)  # type: ignore[arg-type]
    console.print(f"{result.backend}  {result.detail or 'ok'}  hits={len(result.hits)}")
    for hit in result.hits[:40]:
        console.print(f"{hit.path}:{hit.line}:{hit.text}")


app.add_typer(task_app, name="task")
