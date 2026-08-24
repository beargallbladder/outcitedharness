from __future__ import annotations

from harness.cases.schema import Case
from harness.evaluation.base import EvalResult


def evaluate_human(case: Case, answer: str) -> EvalResult:
    preview = answer.strip().replace("\n", " ")
    return EvalResult(
        verdict="PENDING",
        evaluator="human",
        reason="pending human review",
        detail={"preview": preview[:300], "case_id": case.id},
    )
