from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.economics import (
    ReplacementObservation,
    learning_factory_metrics,
    record_replacement_observation,
    write_learning_factory_report,
)
from harness.storage.db import Store
from harness.training.ledger import (
    ArtifactPayload,
    LearningLedger,
    VerificationPayload,
)
from harness.training.models import LearningEvent, SourceKind


NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)


def _observation(identifier: str, **updates) -> ReplacementObservation:
    values = {
        "observation_id": identifier,
        "task_class": "coding",
        "route": "local-qwen",
        "verified_success": True,
        "first_pass": True,
        "repair_cycles": 0,
        "critical_regression": False,
        "frontier_escalated": False,
        "created_at": NOW,
        "time_to_green_ms": 1000,
        "actual_cost": 0.0,
        "direct_frontier_cost": 2.0,
        "gpu_hours": 1.0,
    }
    values.update(updates)
    return ReplacementObservation(**values)


def _admit_proof(store: Store, root: Path, identifier: str) -> str:
    event_id = f"proof-{identifier}"
    ledger = LearningLedger(store, root / "artifacts")
    event = LearningEvent(
        event_id=event_id,
        event_type="verified_repair",
        source_kind=SourceKind.HARNESS,
        source_uri=f"harness://proof/{identifier}",
        source_revision=hashlib.sha256(identifier.encode()).hexdigest(),
        lineage_id=event_id,
        authorization_scope="test",
        created_at=NOW,
        metadata={"data_use": "training", "disposition": "verified"},
    )
    capture = ledger.capture(
        event,
        [ArtifactPayload(kind="proof", content=f"proof {identifier}")],
        [
            VerificationPayload(
                kind="pytest",
                status="pass",
                verifier="pytest",
                output_kind="proof",
            )
        ],
    )
    ledger.admit_verified_event(
        event_id,
        capture.verifications[0].verification_id,
        policy_version="test-v1",
        reason="mechanical test proof",
    )
    return event_id


def _verified_observation(
    store: Store,
    root: Path,
    identifier: str,
    **updates,
) -> ReplacementObservation:
    return _observation(
        identifier,
        event_id=_admit_proof(store, root, identifier),
        **updates,
    )


def test_learning_economics_reports_paid_replacement_by_class(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    record_replacement_observation(
        store,
        _verified_observation(store, tmp_path, "local"),
    )
    record_replacement_observation(
        store,
        _verified_observation(
            store,
            tmp_path,
            "frontier",
            route="frontier",
            first_pass=False,
            repair_cycles=2,
            frontier_escalated=True,
            actual_cost=1.0,
            direct_frontier_cost=2.0,
        ),
    )

    report = learning_factory_metrics(store)
    overall = report["overall"]

    assert overall["verified_success_rate"] == 1.0
    assert overall["paid_task_replacement_rate"] == 0.5
    assert overall["frontier_escalation_rate"] == 0.5
    assert overall["paid_spend_avoided"] == 2.0
    assert overall["spend_avoided_per_gpu_hour"] == 1.0
    assert report["by_task_class"]["coding"] == overall


def test_observation_is_idempotent_but_immutable(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    observation = _verified_observation(store, tmp_path, "same")
    record_replacement_observation(store, observation)
    record_replacement_observation(store, observation)

    with pytest.raises(ValueError, match="immutable"):
        record_replacement_observation(
            store,
            _observation(
                "same",
                event_id=observation.event_id,
                repair_cycles=1,
            ),
        )


def test_unknown_cost_does_not_inflate_savings(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    record_replacement_observation(
        store,
        _verified_observation(
            store,
            tmp_path,
            "unknown",
            actual_cost=None,
            direct_frontier_cost=None,
        ),
    )

    overall = learning_factory_metrics(store)["overall"]

    assert overall["paid_spend_avoided"] is None
    assert overall["cost_coverage_rate"] == 0


def test_escalated_or_unverified_rows_do_not_claim_avoided_spend(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    record_replacement_observation(
        store,
        _verified_observation(
            store,
            tmp_path,
            "escalated",
            frontier_escalated=True,
            actual_cost=1.0,
            direct_frontier_cost=5.0,
        ),
    )
    record_replacement_observation(
        store,
        _observation(
            "failed",
            verified_success=False,
            first_pass=False,
            actual_cost=0.0,
            direct_frontier_cost=7.0,
        ),
    )

    overall = learning_factory_metrics(store)["overall"]

    assert overall["paid_tasks_replaced"] == 0
    assert overall["paid_spend_avoided"] is None
    assert overall["direct_frontier_baseline_spend"] is None
    assert overall["cost_coverage_rate"] == 1.0


def test_verified_replacement_requires_admitted_mechanical_proof(
    tmp_path: Path,
):
    store = Store(tmp_path / "harness.db")

    with pytest.raises(ValueError, match="admitted learning event"):
        record_replacement_observation(store, _observation("unsupported"))


def test_critical_regression_does_not_count_as_replacement(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    record_replacement_observation(
        store,
        _verified_observation(
            store,
            tmp_path,
            "regression",
            critical_regression=True,
        ),
    )

    overall = learning_factory_metrics(store)["overall"]
    assert overall["paid_tasks_replaced"] == 0
    assert overall["paid_spend_avoided"] is None


def test_replacement_observation_cli_round_trip(tmp_path: Path):
    database = tmp_path / "harness.db"
    store = Store(database)
    input_path = tmp_path / "observation.json"
    payload = asdict(_verified_observation(store, tmp_path, "cli"))
    payload["created_at"] = payload["created_at"].isoformat()
    input_path.write_text(json.dumps(payload))

    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "record_replacement_observation.py"
            ),
            "--database",
            str(database),
            "--input",
            str(input_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cli"
    assert learning_factory_metrics(Store(database))["overall"][
        "paid_tasks_replaced"
    ] == 1


def test_learning_factory_report_is_write_once(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    destination = tmp_path / "report.json"
    write_learning_factory_report(store, destination)
    write_learning_factory_report(store, destination)
    record_replacement_observation(
        store,
        _verified_observation(store, tmp_path, "later"),
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_learning_factory_report(store, destination)
