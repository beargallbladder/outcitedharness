from __future__ import annotations

from dataclasses import dataclass

from harness.cases.schema import Case
from harness.config import AppConfig
from harness.runner import ModelAttempt, invoke_model, make_run_id, summarize_attempts
from harness.storage.db import Store


@dataclass
class EscalationOutcome:
    run_id: str
    cases: list[Case]
    attempts: dict[str, list[ModelAttempt]]


async def run_escalation(
    cfg: AppConfig,
    cases: list[Case],
    store: Store | None = None,
    run_id: str | None = None,
) -> EscalationOutcome:
    store = store or Store(cfg.settings.db_path)
    run_id = run_id or make_run_id("escalate")
    models = cfg.enabled_models()
    if not models:
        raise RuntimeError("No enabled models for escalation")

    store.create_run(run_id, "escalate")
    attempts_by_case: dict[str, list[ModelAttempt]] = {}

    for case in cases:
        attempts: list[ModelAttempt] = []
        for model in models:
            attempt = await invoke_model(cfg, store, run_id, case, model)
            attempts.append(attempt)
            if attempt.evaluation.verdict == "PASS":
                break
        summary = summarize_attempts(case, "escalate", attempts)
        summary["run_id"] = run_id
        store.insert_case_run(summary)
        attempts_by_case[case.id] = attempts

    store.finish_run(run_id, len(cases))
    return EscalationOutcome(run_id=run_id, cases=cases, attempts=attempts_by_case)
