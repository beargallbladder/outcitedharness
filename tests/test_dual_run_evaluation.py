from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from harness.training.promotion import CandidateEvaluation


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "record_dual_run_evaluation.py"
SPEC = importlib.util.spec_from_file_location("record_dual_run_evaluation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dual_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dual_run)


def _previous() -> CandidateEvaluation:
    return CandidateEvaluation(
        task_class="coding",
        frozen_holdout_id="dataset-v1",
        holdout_disjoint=True,
        sample_count=100,
        verified_success_rate=0.8,
        baseline_verified_success_rate=0.7,
        frontier_escalation_rate=0.1,
        baseline_frontier_escalation_rate=0.2,
        cost_per_verified_success=0.1,
        baseline_cost_per_verified_success=0.2,
        p95_latency_ms=100,
        baseline_p95_latency_ms=100,
        first_pass_success_rate=0.7,
        mean_repair_cycles=1,
        critical_regressions=0,
        checkpoint_reproducible=True,
        resume_verified=True,
        candidate_sha256="a" * 64,
        gpu_hours=1,
    )


def _row(index: int) -> dict:
    outcome = {
        "verified_success": True,
        "first_pass": True,
        "frontier_escalated": False,
        "latency_ms": 100 + index,
        "cost": 0.1,
        "repair_cycles": 0,
        "critical_regression": False,
        "gpu_seconds": 1,
        "evidence_sha256": f"{index:064x}",
    }
    return {
        "schema": "harness.dual-run.v1",
        "request_id": f"request-{index}",
        "task_class": "coding",
        "route": "harness-orch",
        "candidate_sha256": "a" * 64,
        "baseline_sha256": "b" * 64,
        "candidate": outcome,
        "baseline": {**outcome, "cost": 0.2, "latency_ms": 120 + index},
    }


def test_dual_run_collector_derives_metrics_from_hash_bound_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dual-run.jsonl"
    path.write_text(
        "".join(json.dumps(_row(index)) + "\n" for index in range(50))
    )

    rows, digest = dual_run.load_dual_runs(path)
    evaluation = dual_run.derive_evaluation(
        rows,
        previous=_previous(),
        log_sha256=digest,
    )

    assert evaluation.sample_count == 50
    assert evaluation.verified_success_rate == 1
    assert evaluation.cost_per_verified_success == pytest.approx(0.1)
    assert evaluation.metadata["dual_run_log_sha256"] == digest
    assert evaluation.metadata["dual_run_derived"] is True


def test_dual_run_collector_rejects_duplicate_requests(tmp_path: Path) -> None:
    path = tmp_path / "dual-run.jsonl"
    row = _row(1)
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="request IDs must be unique"):
        dual_run.load_dual_runs(path)
