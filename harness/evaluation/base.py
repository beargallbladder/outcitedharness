from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from harness.cases.schema import Case


Verdict = Literal["PASS", "FAIL", "PARTIAL", "ERROR", "PENDING"]


@dataclass
class EvalResult:
    verdict: Verdict
    evaluator: str
    detail: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    format_ok: bool | None = None
    correctness_ok: bool | None = None


def evaluate(case: Case, answer: str, error: str | None = None) -> EvalResult:
    if error:
        return EvalResult(
            verdict="ERROR",
            evaluator=case.evaluation.type,
            reason=error,
        )

    from harness.evaluation.command import evaluate_command
    from harness.evaluation.exact import (
        evaluate_exact_text,
        evaluate_keyword_rubric,
        evaluate_regex,
    )
    from harness.evaluation.human import evaluate_human
    from harness.evaluation.json_eval import (
        evaluate_exact_json,
        evaluate_json_fields,
        evaluate_numeric_fields,
        evaluate_required_fields,
    )

    dispatch = {
        "exact_text": evaluate_exact_text,
        "exact_json": evaluate_exact_json,
        "numeric_fields": evaluate_numeric_fields,
        "required_fields": evaluate_required_fields,
        "json_fields": evaluate_json_fields,
        "keyword_rubric": evaluate_keyword_rubric,
        "regex": evaluate_regex,
        "command": evaluate_command,
        "human": evaluate_human,
    }
    fn = dispatch[case.evaluation.type]
    return fn(case, answer)
