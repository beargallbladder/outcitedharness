from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from harness.config import find_project_root


@dataclass(frozen=True)
class Worker:
    id: str
    enabled: bool
    model_key: str | None
    endpoint: str | None
    capabilities: tuple[str, ...]
    failover_order: int | None
    notes: str = ""

    def status(self) -> str:
        if not self.enabled:
            return "unavailable"
        if not self.model_key:
            return "unconfigured"
        return "healthy"


class WorkerRegistry:
    def __init__(self, workers: dict[str, Worker]):
        self.workers = workers

    def get(self, worker_id: str) -> Worker | None:
        return self.workers.get(worker_id)

    def failover_keys(self) -> list[str]:
        """Enabled workers with a live model_key, in failover_order.

        This is the old auto_ladder: primary_coder → fallback_reasoner → frontier.
        Disabled future nodes are skipped, not errors.
        """
        rows = [
            w
            for w in self.workers.values()
            if w.enabled and w.model_key and w.failover_order is not None
        ]
        rows.sort(key=lambda w: w.failover_order or 0)
        return [w.model_key for w in rows if w.model_key]

    def summary(self) -> list[dict[str, Any]]:
        rows = sorted(
            self.workers.values(),
            key=lambda w: (w.failover_order is None, w.failover_order or 0, w.id),
        )
        return [
            {
                "id": w.id,
                "enabled": w.enabled,
                "status": w.status(),
                "model_key": w.model_key,
                "endpoint": w.endpoint,
                "capabilities": list(w.capabilities),
                "failover_order": w.failover_order,
                "notes": w.notes,
                "detail": None if w.enabled else f"{w.id} unavailable",
            }
            for w in rows
        ]


def load_registry(root: Path | None = None) -> WorkerRegistry:
    if root is not None:
        path = Path(root) / "config" / "workers.yaml"
        if not path.exists():
            return WorkerRegistry({})
        return _parse(path)
    root = find_project_root()
    path = root / "config" / "workers.yaml"
    if not path.exists():
        return WorkerRegistry({})
    return _parse(path)


def _parse(path: Path) -> WorkerRegistry:
    raw = yaml.safe_load(path.read_text()) or {}
    workers: dict[str, Worker] = {}
    for key, spec in (raw.get("workers") or {}).items():
        spec = spec or {}
        caps = spec.get("capabilities") or []
        workers[str(key)] = Worker(
            id=str(key),
            enabled=bool(spec.get("enabled", False)),
            model_key=spec.get("model_key"),
            endpoint=spec.get("endpoint"),
            capabilities=tuple(str(c) for c in caps),
            failover_order=spec.get("failover_order"),
            notes=str(spec.get("notes") or ""),
        )
    return WorkerRegistry(workers)
