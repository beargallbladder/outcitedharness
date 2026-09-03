from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.table import Table

from harness.config import AppConfig
from harness.report import fmt_money, fmt_seconds
from harness.stats import latency_list, median, percentile
from harness.storage.db import Store
from harness.training.security import assert_no_secrets


@dataclass(frozen=True)
class ReplacementObservation:
    observation_id: str
    task_class: str
    route: str
    verified_success: bool
    first_pass: bool
    repair_cycles: int
    critical_regression: bool
    frontier_escalated: bool
    created_at: datetime
    event_id: str | None = None
    evaluation_id: str | None = None
    time_to_green_ms: float | None = None
    pinout_exact: float | None = None
    pinout_leaf_f1: float | None = None
    actual_cost: float | None = None
    direct_frontier_cost: float | None = None
    gpu_hours: float = 0

    def __post_init__(self) -> None:
        if not self.observation_id or not self.task_class or not self.route:
            raise ValueError("observation identity, task class, and route are required")
        for name in (
            "observation_id",
            "task_class",
            "route",
            "event_id",
            "evaluation_id",
        ):
            value = getattr(self, name)
            if value is not None:
                assert_no_secrets(value, field=f"replacement {name}")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        if self.repair_cycles < 0 or self.gpu_hours < 0:
            raise ValueError("repair_cycles and gpu_hours cannot be negative")
        if self.first_pass and not self.verified_success:
            raise ValueError("first_pass requires verified_success")
        for name in ("pinout_exact", "pinout_leaf_f1"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        for name in ("time_to_green_ms", "actual_cost", "direct_frontier_cost"):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} cannot be negative")
        if not math.isfinite(self.gpu_hours):
            raise ValueError("gpu_hours must be finite")


def record_replacement_observation(
    store: Store,
    observation: ReplacementObservation,
) -> None:
    payload = asdict(observation)
    payload["created_at"] = observation.created_at.astimezone(
        timezone.utc
    ).isoformat(timespec="seconds")
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute(
            """
            SELECT * FROM replacement_observations
            WHERE observation_id = ?
            """,
            (observation.observation_id,),
        ).fetchone()
        if current:
            comparable = {
                key: current[key]
                for key in (
                    "observation_id",
                    "event_id",
                    "evaluation_id",
                    "task_class",
                    "route",
                    "time_to_green_ms",
                    "repair_cycles",
                    "pinout_exact",
                    "pinout_leaf_f1",
                    "actual_cost",
                    "direct_frontier_cost",
                    "gpu_hours",
                    "created_at",
                )
            }
            comparable.update(
                {
                    "verified_success": bool(current["verified_success"]),
                    "first_pass": bool(current["first_pass"]),
                    "critical_regression": bool(current["critical_regression"]),
                    "frontier_escalated": bool(current["frontier_escalated"]),
                }
            )
            if comparable != payload:
                raise ValueError(
                    f"replacement observation {observation.observation_id!r} "
                    "is immutable"
                )
            return
        if observation.verified_success:
            if observation.event_id is None:
                raise ValueError(
                    "verified replacement requires an admitted learning event"
                )
            proof = conn.execute(
                """
                SELECT 1
                FROM learning_admissions AS admission
                JOIN learning_verifications AS verification
                  ON verification.verification_id = admission.verification_id
                WHERE admission.event_id = ?
                  AND admission.decision = 'eligible'
                  AND verification.event_id = admission.event_id
                  AND verification.status = 'pass'
                """,
                (observation.event_id,),
            ).fetchone()
            if proof is None:
                raise ValueError(
                    "verified replacement lacks admitted mechanical proof"
                )
        try:
            conn.execute(
                """
                INSERT INTO replacement_observations (
                    observation_id, event_id, evaluation_id, task_class, route,
                    verified_success, first_pass, time_to_green_ms, repair_cycles,
                    pinout_exact, pinout_leaf_f1, critical_regression,
                    frontier_escalated, actual_cost, direct_frontier_cost,
                    gpu_hours, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.observation_id,
                    observation.event_id,
                    observation.evaluation_id,
                    observation.task_class,
                    observation.route,
                    int(observation.verified_success),
                    int(observation.first_pass),
                    observation.time_to_green_ms,
                    observation.repair_cycles,
                    observation.pinout_exact,
                    observation.pinout_leaf_f1,
                    int(observation.critical_regression),
                    int(observation.frontier_escalated),
                    observation.actual_cost,
                    observation.direct_frontier_cost,
                    observation.gpu_hours,
                    payload["created_at"],
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("observation references unknown immutable evidence") from exc


def learning_factory_metrics(store: Store) -> dict:
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM replacement_observations ORDER BY created_at"
        ).fetchall()
    return {
        "schema": "harness.learning-factory-metrics.v1",
        "overall": _replacement_metrics(rows),
        "by_task_class": {
            task_class: _replacement_metrics(
                [row for row in rows if row["task_class"] == task_class]
            )
            for task_class in sorted({row["task_class"] for row in rows})
        },
    }


def _replacement_metrics(rows) -> dict:
    count = len(rows)
    verified = [row for row in rows if row["verified_success"]]
    first_pass = [row for row in verified if row["first_pass"]]
    local_successes = [
        row
        for row in verified
        if not row["frontier_escalated"] and not row["critical_regression"]
    ]
    paired_costs = [
        row
        for row in rows
        if row["actual_cost"] is not None
        and row["direct_frontier_cost"] is not None
    ]
    replacement_costs = [
        row
        for row in paired_costs
        if row["verified_success"]
        and not row["frontier_escalated"]
        and not row["critical_regression"]
    ]
    spend_avoided = (
        sum(
            row["direct_frontier_cost"] - row["actual_cost"]
            for row in replacement_costs
        )
        if replacement_costs
        else None
    )
    gpu_hours = sum(row["gpu_hours"] for row in rows)
    time_to_green = [
        row["time_to_green_ms"]
        for row in verified
        if row["time_to_green_ms"] is not None
    ]
    pinout_f1 = [
        row["pinout_leaf_f1"]
        for row in rows
        if row["pinout_leaf_f1"] is not None
    ]
    pinout_exact = [
        row["pinout_exact"]
        for row in rows
        if row["pinout_exact"] is not None
    ]
    return {
        "observations": count,
        "verified_success_rate": len(verified) / count if count else 0,
        "verified_successes": len(verified),
        "verified_first_pass_rate": len(first_pass) / count if count else 0,
        "paid_tasks_replaced": len(local_successes),
        "paid_tasks_observed": count,
        "paid_task_replacement_rate": (
            len(local_successes) / count if count else 0
        ),
        "frontier_escalation_rate": (
            sum(bool(row["frontier_escalated"]) for row in rows) / count
            if count
            else 0
        ),
        "mean_time_to_green_ms": mean(time_to_green) if time_to_green else None,
        "mean_repair_cycles": (
            mean(row["repair_cycles"] for row in rows) if rows else None
        ),
        "pinout_leaf_f1": mean(pinout_f1) if pinout_f1 else None,
        "pinout_exact_rate": mean(pinout_exact) if pinout_exact else None,
        "critical_regression_rate": (
            sum(bool(row["critical_regression"]) for row in rows) / count
            if count
            else 0
        ),
        "actual_paid_spend": (
            sum(row["actual_cost"] for row in replacement_costs)
            if replacement_costs
            else None
        ),
        "direct_frontier_baseline_spend": (
            sum(row["direct_frontier_cost"] for row in replacement_costs)
            if replacement_costs
            else None
        ),
        "paid_spend_avoided": spend_avoided,
        "cost_coverage_rate": len(paired_costs) / count if count else 0,
        "gpu_hours": gpu_hours,
        "spend_avoided_per_gpu_hour": (
            spend_avoided / gpu_hours
            if spend_avoided is not None and gpu_hours > 0
            else None
        ),
    }


def write_learning_factory_report(store: Store, destination) -> dict:
    report = learning_factory_metrics(store)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") == payload:
            return report
        raise FileExistsError(f"refusing to overwrite report: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return report


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
