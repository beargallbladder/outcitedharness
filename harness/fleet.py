from __future__ import annotations

import asyncio
from dataclasses import dataclass

from harness.config import AppConfig
from harness.providers.factory import build_provider
from harness.workers.registry import load_registry


@dataclass
class FleetCheck:
    worker_id: str
    role: str
    model_key: str
    ok: bool
    detail: str


async def validate_fleet(cfg: AppConfig) -> list[FleetCheck]:
    """Validate configured references and live endpoints before gateway restart."""
    registry = load_registry(cfg.root)
    checks: list[FleetCheck] = []
    probes = []

    for worker in registry.workers.values():
        if not worker.enabled:
            continue
        key = worker.model_key or ""
        model = cfg.models.get(key)
        if not key:
            checks.append(FleetCheck(worker.id, worker.role, "", False, "missing model_key"))
            continue
        if model is None:
            checks.append(FleetCheck(worker.id, worker.role, key, False, "unknown model_key"))
            continue
        if not model.enabled:
            checks.append(FleetCheck(worker.id, worker.role, key, False, "model disabled"))
            continue
        if worker.endpoint and worker.endpoint.rstrip("/") != model.base_url.rstrip("/"):
            checks.append(
                FleetCheck(
                    worker.id,
                    worker.role,
                    key,
                    False,
                    "worker endpoint does not match models.yaml",
                )
            )
            continue
        probes.append((worker, model))

    async def probe(worker, model) -> FleetCheck:
        ok, detail = await build_provider(model).health(cfg.settings.health_timeout_s)
        return FleetCheck(worker.id, worker.role, model.key, ok, "ok" if ok else detail)

    checks.extend(await asyncio.gather(*(probe(worker, model) for worker, model in probes)))
    checks.sort(key=lambda row: (row.role, row.worker_id))
    return checks
