from __future__ import annotations

from typing import TYPE_CHECKING

from harness.config import AppConfig, ModelConfig
from harness.workers.registry import WorkerRegistry

if TYPE_CHECKING:
    from harness.gateway.spec import ClineSpec


def should_failover(status: int, error: str | None, has_next: bool) -> bool:
    """Dead-box only: connect / 5xx / empty. Bad answers do not move the worker."""
    if not has_next:
        return False
    return status >= 500 or bool(error)


def route_models(
    spec: ClineSpec,
    cfg: AppConfig,
    requested: str,
    registry: WorkerRegistry | None = None,
) -> list[ModelConfig]:
    """Resolve Cline model id → workers. auto uses the registry failover chain."""
    from harness.gateway.spec import ladder_for

    return ladder_for(spec, cfg, requested, registry)
