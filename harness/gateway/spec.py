from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from harness.config import AppConfig, ModelConfig, find_project_root


@dataclass
class ClineSpec:
    listen_host: str
    listen_port: int
    api_key: str
    aliases: dict[str, str]
    auto_ladder: list[str]
    context_window: int
    max_output_tokens: int


def load_cline_spec(root: Path | None = None) -> ClineSpec:
    root = find_project_root(root)
    raw = yaml.safe_load((root / "config" / "cline.yaml").read_text()) or {}
    aliases = {str(k): str(v) for k, v in (raw.get("aliases") or {}).items()}
    return ClineSpec(
        listen_host=str(raw.get("listen_host", "127.0.0.1")),
        listen_port=int(raw.get("listen_port", 8787)),
        api_key=str(raw.get("api_key") or "harness-local"),
        aliases=aliases,
        auto_ladder=[str(x) for x in (raw.get("auto_ladder") or [])],
        context_window=int(raw.get("context_window", 131072)),
        max_output_tokens=int(raw.get("max_output_tokens", 8192)),
    )


def resolve_alias(spec: ClineSpec, requested: str) -> str:
    if requested in spec.aliases:
        return spec.aliases[requested]
    if requested in spec.aliases.values():
        return requested
    return requested


def ladder_for(
    spec: ClineSpec,
    cfg: AppConfig,
    requested: str,
    registry: Any | None = None,
) -> list[ModelConfig]:
    target = resolve_alias(spec, requested)
    if target == "auto":
        if registry is None:
            from harness.workers.registry import load_registry

            registry = load_registry(cfg.root)
        keys = registry.failover_keys() or spec.auto_ladder
    else:
        keys = [target]
    out: list[ModelConfig] = []
    for key in keys:
        model = cfg.models.get(key)
        if model is None:
            raise KeyError(f"Unknown harness model '{key}' (from Cline model '{requested}')")
        if not model.enabled:
            continue
        out.append(model)
    if not out:
        raise KeyError(f"No enabled models for Cline id '{requested}'")
    return out


def listed_models(spec: ClineSpec) -> list[dict[str, Any]]:
    rows = []
    for alias, target in spec.aliases.items():
        rows.append(
            {
                "id": alias,
                "object": "model",
                "owned_by": "harness",
                "permission": [],
                "root": target,
                "context_length": spec.context_window,
            }
        )
    return rows
