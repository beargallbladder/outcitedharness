from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from harness.baseline import run_baseline
from harness.cases.loader import discover_cases
from harness.checkpoints import CheckpointError, CheckpointStore, RollbackConflict
from harness.config import load_config
from harness.economics import compare_economics
from harness.escalation import run_escalation
from harness.fleet import validate_fleet
from harness.health import check_all
from harness.dispatch import run_dispatch
from harness.optimize import run_optimize
from harness.promote import promote_task
from harness.report import (
    fmt_money,
    fmt_seconds,
    print_attempts,
    print_dispatch_report,
    print_escalation_summary,
    print_health,
    print_optimize_report,
    print_runs,
    print_tournament_summary,
)
from harness.rescue import PACKET_TEMPLATE, PacketError, run_rescue
from harness.storage.db import Store
from harness.serial import discover_tickets, run_serial
from harness.task.models import AttemptRecord, Decision, Evidence
from harness.task.search import search_code
from harness.task.service import TaskService
from harness.tournament import run_tournament
from harness.workers.registry import load_registry


app = typer.Typer(help="Local/cloud model tournament harness", no_args_is_help=True)
fleet_app = typer.Typer(help="Validate the manually configured model fleet")
app.add_typer(fleet_app, name="fleet")
gci_app = typer.Typer(help="Global Code Intelligence service and repository registry")
app.add_typer(gci_app, name="gci")
console = Console()


def _cfg():
    return load_config()


def _path(value: str) -> Path:
    return Path(value).expanduser()


@app.command("rollback-task")
def rollback_task(
    task_id: str = typer.Argument(..., help="Task whose latest checkpoint should be restored"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive confirmation"),
) -> None:
    """Restore only task-attributed files to their pre-task state."""
    cfg = _cfg()
    svc = TaskService(Store(cfg.settings.db_path))
    try:
        svc.get(task_id)
    except KeyError:
        raise typer.BadParameter(f"Unknown task {task_id}") from None
    states = svc.evidence(task_id, kind="orch_loop")
    payload = states[-1].payload if states else None
    if (
        not isinstance(payload, dict)
        or not payload.get("checkpoint_run_id")
        or not payload.get("checkpoint_available")
    ):
        console.print(f"[red]No rollback checkpoint is available for {task_id}.[/red]")
        raise typer.Exit(code=1)
    run_id = str(payload["checkpoint_run_id"])
    store = CheckpointStore(
        cfg.settings.results_dir / "checkpoints",
        max_file_bytes=cfg.settings.checkpoint_max_file_bytes,
    )
    try:
        preview = store.rollback_preview(task_id, run_id)
    except CheckpointError as exc:
        svc.add_evidence(
            Evidence(
                task_id=task_id,
                kind="task_rollback",
                payload={"status": "refused", "run_id": run_id, "reason": str(exc)},
            )
        )
        console.print(f"[red]Rollback refused: {exc}[/red]")
        raise typer.Exit(code=1) from None
    if preview.conflicts:
        reason = "post-checkpoint changes: " + ", ".join(preview.conflicts)
        store.record_rollback_refusal(
            task_id,
            run_id,
            reason=reason,
            conflicts=list(preview.conflicts),
        )
        svc.add_evidence(
            Evidence(
                task_id=task_id,
                kind="task_rollback",
                payload={
                    "status": "refused",
                    "run_id": run_id,
                    "reason": reason,
                    "conflicts": list(preview.conflicts),
                },
            )
        )
        console.print(f"[red]Rollback refused: {reason}[/red]")
        raise typer.Exit(code=1)

    console.print(f"Task: [bold]{task_id}[/bold]  checkpoint run: {run_id}")
    console.print("Restore: " + (", ".join(preview.restore) or "(none)"))
    console.print("Remove: " + (", ".join(preview.remove) or "(none)"))
    if not yes and not typer.confirm("Rollback these task-attributed paths?"):
        console.print("Rollback cancelled.")
        raise typer.Exit(code=1)
    try:
        result = store.rollback(task_id, run_id)
    except RollbackConflict as exc:
        reason = "post-checkpoint changes: " + ", ".join(exc.paths)
        svc.add_evidence(
            Evidence(
                task_id=task_id,
                kind="task_rollback",
                payload={
                    "status": "refused",
                    "run_id": run_id,
                    "reason": reason,
                    "conflicts": exc.paths,
                },
            )
        )
        console.print(f"[red]Rollback refused: {reason}[/red]")
        raise typer.Exit(code=1) from None
    except CheckpointError as exc:
        svc.add_evidence(
            Evidence(
                task_id=task_id,
                kind="task_rollback",
                payload={"status": "refused", "run_id": run_id, "reason": str(exc)},
            )
        )
        console.print(f"[red]Rollback failed: {exc}[/red]")
        raise typer.Exit(code=1) from None
    svc.add_evidence(
        Evidence(
            task_id=task_id,
            kind="task_rollback",
            payload={
                "status": "success",
                "run_id": run_id,
                "restored": list(result.restored),
                "removed": list(result.removed),
                "removed_dirs": list(result.removed_dirs),
                "audit_path": result.audit_path,
            },
        )
    )
    console.print(
        f"[green]Rollback complete.[/green] restored={len(result.restored)} "
        f"removed={len(result.removed)}"
    )


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
    table.add_column("Role")
    table.add_column("Status")
    table.add_column("Model")
    table.add_column("Endpoint")
    table.add_column("Detail")
    for row in registry.summary():
        status = row["status"]
        style = {"healthy": "green", "unavailable": "dim", "unconfigured": "yellow"}.get(status, "")
        table.add_row(
            row["id"],
            row.get("role") or "-",
            f"[{style}]{status}[/{style}]" if style else status,
            row["model_key"] or "-",
            row["endpoint"] or "-",
            row["detail"] or "",
        )
    console.print(table)
    chain = registry.failover_keys()
    if chain:
        console.print("auto failover: " + " → ".join(chain))
    pool = [w.id for w in registry.pool("coder")]
    if pool:
        console.print("coder pool: " + ", ".join(pool))


@fleet_app.command("validate")
def fleet_validate() -> None:
    """Probe enabled workers and validate config before restarting the gateway."""
    rows = asyncio.run(validate_fleet(_cfg()))
    table = Table(title="Fleet validation")
    table.add_column("Worker")
    table.add_column("Role")
    table.add_column("Model")
    table.add_column("Result")
    table.add_column("Detail")
    for row in rows:
        table.add_row(
            row.worker_id,
            row.role,
            row.model_key or "-",
            "[green]PASS[/green]" if row.ok else "[red]FAIL[/red]",
            row.detail,
        )
    console.print(table)
    if not rows or any(not row.ok for row in rows):
        raise typer.Exit(code=1)


@app.command()
def optimize(
    workers: Optional[str] = typer.Option(
        None,
        help="Comma-separated model keys (default: four GB10 coder boxes)",
    ),
    only: Optional[str] = typer.Option(None, help="Comma-separated case ids"),
    direct: bool = typer.Option(False, "--direct", help="Skip M5 packet/rank; same prompt to every box"),
    senior: bool = typer.Option(False, "--senior", help="Ask Claude for a short tune note (costs money)"),
) -> None:
    """M5 foreman → three GB10 workers in parallel. Measures latency, tokens, tool calls."""
    cfg = _cfg()
    worker_keys = [k.strip() for k in workers.split(",") if k.strip()] if workers else None
    only_ids = [k.strip() for k in only.split(",") if k.strip()] if only else None
    report = asyncio.run(
        run_optimize(
            cfg,
            worker_keys=worker_keys,
            use_foreman=not direct,
            use_senior=senior,
            only=only_ids,
        )
    )
    print_optimize_report(report, console)


@app.command()
def dispatch(
    intent: str = typer.Argument(..., help="What you want done. M5 slices this into packets."),
    workers: Optional[str] = typer.Option(None, help="Comma-separated model keys (default: enabled coder pool)"),
    senior: bool = typer.Option(False, "--senior", help="Ask Claude after the pool (costs money)"),
) -> None:
    """M5 carves packets; idle GB10 coders take them. Tester scores tools/latency/tokens."""
    cfg = _cfg()
    worker_keys = [k.strip() for k in workers.split(",") if k.strip()] if workers else None
    report = asyncio.run(run_dispatch(cfg, intent, worker_keys=worker_keys, use_senior=senior))
    print_dispatch_report(report, console)


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
def promote(
    task_id: str = typer.Argument(..., help="Verified frontier-rescue task id"),
    pack: str = typer.Option("cases/learned", help="Destination case pack"),
) -> None:
    """Turn a verified local-fail/frontier-pass task into a regression case."""
    cfg = _cfg()
    try:
        case_dir = promote_task(
            Store(cfg.settings.db_path),
            task_id,
            _path(pack),
        )
    except (KeyError, ValueError, FileExistsError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"Promoted [bold]{task_id}[/bold] to {case_dir}")


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


@task_app.command("current")
def task_current() -> None:
    """Show the latest open Cline session and its last 5 attempts."""
    cfg = _cfg()
    svc = TaskService(Store(cfg.settings.db_path))
    task = svc.latest_session()
    if not task:
        console.print("no open cline session")
        raise typer.Exit(0)

    console.print(f"Task: {task.task_id}  {task.status}  {task.intent}")
    attempts = svc.attempts(task.task_id)
    if not attempts:
        console.print("No attempts yet")
        return

    recent = list(reversed(attempts))[:5]
    table = Table(title="Last 5 attempts")
    table.add_column("Attempt")
    table.add_column("Worker")
    table.add_column("Result")
    table.add_column("Started")
    table.add_column("Finished")
    table.add_column("Tokens")
    for rec in recent:
        tokens = f"{rec.input_tokens or 0}→{rec.output_tokens or 0}"
        table.add_row(
            str(rec.attempt),
            rec.worker,
            rec.result,
            rec.started_at[:19] if rec.started_at else "-",
            rec.finished_at[:19] if rec.finished_at else "-",
            tokens,
        )
    console.print(table)


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


@app.command("index")
def index_cmd(
    repo: Optional[list[str]] = typer.Option(None, "--repo", help="Extra repo to index"),
) -> None:
    """Incrementally embed active trees on the M5. Uses :8800 /v1/embeddings only."""
    from harness.task.code_index import DEFAULT_REPOS, default_index_path, index_repos

    cfg = _cfg()
    repos = [Path(p) for p in (cfg.settings.code_index_repos or list(DEFAULT_REPOS))]
    for extra in repo or []:
        repos.append(_path(extra))
    db = cfg.settings.code_index_path or default_index_path(cfg.root)
    stats = index_repos(repos, db)
    console.print(
        f"code index {db} files={stats['files']} unchanged={stats['unchanged']} "
        f"chunks={stats['chunks']} embedded={stats['embedded']}"
    )


@app.command("retrieve")
def retrieve_cmd(
    query: str,
    limit: int = typer.Option(8, help="Top chunks"),
    workspace: Optional[str] = typer.Option(
        None,
        help="Active workspace root (default: cwd). Other indexed repos are not searched.",
    ),
) -> None:
    """Query the M5 code index. Does not hit the CR category search endpoint."""
    from harness.task.code_index import default_index_path, query_index

    cfg = _cfg()
    db = cfg.settings.code_index_path or default_index_path(cfg.root)
    root = _path(workspace) if workspace else Path.cwd()
    hits = query_index(query, db, repo_root=root, limit=limit)
    if not hits:
        console.print(f"no hits in {db} workspace={root}")
        raise typer.Exit(code=1)
    for hit in hits:
        console.print(
            f"{hit.score:.3f}  {hit.path}:{hit.start_line}-{hit.end_line}"
        )
        console.print(hit.text.splitlines()[0][:160] if hit.text else "")


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


def _gci_client(cfg=None):
    from harness.gci.client import GCIClient

    cfg = cfg or _cfg()
    token = os.environ.get(cfg.settings.gci_token_env, "")
    return GCIClient(
        cfg.settings.gci_url,
        token=token,
        timeout=cfg.settings.gci_timeout_s,
    )


@gci_app.command("serve")
def gci_serve(
    host: Optional[str] = typer.Option(None, help="Bind host (default: Spark Tailscale IP)"),
    port: Optional[int] = typer.Option(None, help="Bind port (default: 8810)"),
    db: Optional[str] = typer.Option(None, help="Independent GCI SQLite path"),
) -> None:
    """Run the isolated :8810 service. Requires HARNESS_GCI_TOKEN."""
    from harness.gci.api import serve

    serve(host=host, port=port, db_path=_path(db) if db else None)


@gci_app.command("status")
def gci_status() -> None:
    """Show service health and registered repositories."""
    client = _gci_client()
    health = client.health()
    console.print(
        f"ready={health.get('ready')} paused={health.get('paused')} "
        f"queue_depth={health.get('queue_depth')}"
    )
    for row in client.repos():
        console.print(
            f"{row['repo_id']} {row['source_host']}:{row['repo_root']} "
            f"files={row['file_count']} state={str(row['state_hash'])[:12]}",
            markup=False,
        )


def _gci_scan(repos: list[str] | None, *, wait: bool) -> None:
    from harness.gci.scanner import build_snapshot, repo_id

    cfg = _cfg()
    approved = cfg.settings.code_index_repos
    selected = repos or approved
    if not selected:
        raise typer.BadParameter("No approved repositories configured in code_index_repos")
    client = _gci_client(cfg)
    source_host = socket.gethostname()
    for value in selected:
        root = _path(value).resolve()
        rid = repo_id(source_host, root)
        previous = client.manifest(rid)
        snapshot = build_snapshot(
            root,
            approved_roots=approved,
            previous_files=previous.get("files") or {},
            source_host=source_host,
        )
        if snapshot.state_hash == previous.get("state_hash"):
            console.print(f"{root}: unchanged")
            continue
        job_id = client.submit(snapshot, refresh=bool(previous.get("state_hash")))
        console.print(
            f"{root}: queued {job_id} changed={len(snapshot.documents)} "
            f"deleted={len(snapshot.deleted)} files={len(snapshot.file_hashes)}"
        )
        if wait:
            row = client.wait_job(job_id)
            console.print(f"{root}: {row['state']}")
            if row["state"] != "complete":
                raise typer.Exit(code=1)


@gci_app.command("scan")
def gci_scan(
    repo: Optional[list[str]] = typer.Option(None, "--repo", help="Approved repo root"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for each index job"),
) -> None:
    """Register and index approved repositories from this machine."""
    _gci_scan(repo, wait=wait)


@gci_app.command("refresh")
def gci_refresh(
    repo: Optional[list[str]] = typer.Option(None, "--repo", help="Approved repo root"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for each index job"),
) -> None:
    """Push only changed source documents and deletions."""
    _gci_scan(repo, wait=wait)


@gci_app.command("search")
def gci_search(
    query: str = typer.Argument(...),
    mode: str = typer.Option("semantic", help="semantic|exact|symbol"),
    limit: int = typer.Option(8, min=1, max=100),
    workspace: Optional[str] = typer.Option(None, help="Optional exact repo-root filter"),
) -> None:
    """Search registered repositories without granting filesystem access."""
    hits = _gci_client().search(
        query,
        mode=mode,
        limit=limit,
        repo_root=str(_path(workspace).resolve()) if workspace else None,
    )
    for hit in hits:
        console.print(
            f"{hit.score:.3f} {hit.source_host}:{hit.repo_root}/{hit.path}:"
            f"{hit.start_line}-{hit.end_line} [{hit.match_type}]",
            markup=False,
        )


@gci_app.command("pause")
def gci_pause() -> None:
    """Pause background indexing after the current bounded encoder batch."""
    _gci_client().pause(True)
    console.print("GCI indexing paused")


@gci_app.command("resume")
def gci_resume() -> None:
    """Resume background indexing."""
    _gci_client().pause(False)
    console.print("GCI indexing resumed")


app.add_typer(task_app, name="task")
