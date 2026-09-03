from __future__ import annotations

from datetime import datetime, timezone

from harness.shadow.comparison import (
    AdmissionResult,
    ComparisonReport,
    admit_comparison,
    compare_task,
)
from harness.shadow.replay import replay_task
from harness.shadow.spool import ShadowSpool
from harness.training.ledger import LearningLedger


def process_task(
    spool: ShadowSpool,
    ledger: LearningLedger,
    task_id: str,
) -> AdmissionResult | ComparisonReport:
    kinds = {
        str(row.get("candidate_kind"))
        for row in spool.replays(task_id)
    }
    for candidate_kind in ("local", "frontier"):
        if candidate_kind not in kinds:
            replay_task(
                spool,
                task_id,
                candidate_kind=candidate_kind,
            )
    stored = spool.comparison(task_id)
    comparison = (
        ComparisonReport.model_validate(stored)
        if stored is not None
        else compare_task(spool, task_id)
    )
    if comparison.eligible:
        result = admit_comparison(spool, task_id, ledger)
        spool.record_processing(
            task_id=task_id,
            comparison_id=comparison.comparison_id,
            outcome="admitted",
            learning_event_id=result.capture["event_id"],
            created_at=datetime.now(timezone.utc),
        )
        return result
    spool.record_processing(
        task_id=task_id,
        comparison_id=comparison.comparison_id,
        outcome="rejected",
        learning_event_id=None,
        created_at=datetime.now(timezone.utc),
    )
    return comparison
