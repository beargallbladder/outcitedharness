from __future__ import annotations

from harness.cases.schema import Case
from harness.config import AppConfig
from harness.storage.db import Store
from harness.tournament import TournamentOutcome, run_tournament


async def run_baseline(
    cfg: AppConfig,
    cases: list[Case],
    store: Store | None = None,
) -> TournamentOutcome:
    models = cfg.models_for_mode("baseline")
    if not models:
        raise RuntimeError(
            "No frontier-tier model is enabled. Configure and enable a tier-4 model in config/models.yaml."
        )
    return await run_tournament(cfg, cases, store=store, mode="baseline")
