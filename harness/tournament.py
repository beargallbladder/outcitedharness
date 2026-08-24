from __future__ import annotations

import asyncio
from dataclasses import dataclass

from harness.cases.schema import Case
from harness.config import AppConfig
from harness.runner import ModelAttempt, invoke_model, make_run_id, summarize_attempts
from harness.storage.db import Store, utcnow


@dataclass
class TournamentOutcome:
    run_id: str
    cases: list[Case]
    attempts: dict[str, list[ModelAttempt]]


async def run_tournament(
    cfg: AppConfig,
    cases: list[Case],
    store: Store | None = None,
    run_id: str | None = None,
    mode: str = "tournament",
    seeds: list[int] | None = None,
    only: list[str] | None = None,
) -> TournamentOutcome:
    store = store or Store(cfg.settings.db_path)
    run_id = run_id or make_run_id(mode)
    models = cfg.models_for_mode(mode if mode != "benchmark" else "tournament", only=only)
    if not models:
        raise RuntimeError("No enabled models for this mode")

    seed_list = seeds if seeds else [None]
    store.create_run(run_id, mode, notes=f"seeds={seed_list}")
    attempts_by_case: dict[str, list[ModelAttempt]] = {}

    for case in cases:
        case_attempts: list[ModelAttempt] = []
        first_seed_attempts: list[ModelAttempt] = []
        for i, seed in enumerate(seed_list):
            if cfg.settings.tournament_parallel and len(models) > 1:
                raw = await asyncio.gather(
                    *[
                        invoke_model(cfg, store, run_id, case, model, seed=seed)
                        for model in models
                    ]
                )
                attempts = sorted(raw, key=lambda a: (a.model.tier, a.model.key))
            else:
                attempts = []
                for model in models:
                    attempts.append(
                        await invoke_model(cfg, store, run_id, case, model, seed=seed)
                    )
            case_attempts.extend(attempts)
            if i == 0:
                first_seed_attempts = attempts
        # Min-tier uses the first seed so multi-seed variance stays visible.
        summary = summarize_attempts(case, mode, first_seed_attempts)
        summary["run_id"] = run_id
        if not summary["started_at"]:
            summary["started_at"] = utcnow()
        store.insert_case_run(summary)
        attempts_by_case[case.id] = case_attempts

    store.finish_run(run_id, len(cases))
    return TournamentOutcome(run_id=run_id, cases=cases, attempts=attempts_by_case)
