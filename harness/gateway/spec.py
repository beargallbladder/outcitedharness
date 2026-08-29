from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from harness.config import AppConfig, ModelConfig, find_project_root


@dataclass
class GatewaySpec:
    listen_host: str
    listen_port: int
    api_key: str
    aliases: dict[str, str]
    auto_ladder: list[str]
    context_window: int
    max_output_tokens: int


def load_gateway_spec(root: Path | None = None) -> GatewaySpec:
    root = find_project_root(root)
    raw = yaml.safe_load((root / "config" / "gateway.yaml").read_text()) or {}
    aliases = {str(k): str(v) for k, v in (raw.get("aliases") or {}).items()}
    return GatewaySpec(
        listen_host=str(raw.get("listen_host", "127.0.0.1")),
        listen_port=int(raw.get("listen_port", 8787)),
        api_key=str(raw.get("api_key") or "harness-local"),
        aliases=aliases,
        auto_ladder=[str(x) for x in (raw.get("auto_ladder") or [])],
        context_window=int(raw.get("context_window", 131072)),
        max_output_tokens=int(raw.get("max_output_tokens", 8192)),
    )


def resolve_alias(spec: GatewaySpec, requested: str) -> str:
    if requested in spec.aliases:
        return spec.aliases[requested]
    if requested in spec.aliases.values():
        return requested
    return requested


def is_orch_alias(spec: GatewaySpec, requested: str) -> bool:
    return resolve_alias(spec, requested) == "orch"


def ladder_for(
    spec: GatewaySpec,
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
            raise KeyError(
                f"Unknown harness model '{key}' (from gateway model '{requested}')"
            )
        if not model.enabled:
            continue
        out.append(model)
    if not out:
        raise KeyError(f"No enabled models for gateway id '{requested}'")
    return out


def listed_models(spec: GatewaySpec) -> list[dict[str, Any]]:
    rows = []
    for alias, target in spec.aliases.items():
        row = {
            "id": alias,
            "object": "model",
            "owned_by": "harness",
            "permission": [],
            "root": target,
            "context_length": spec.context_window,
        }
        if target == "orch":
            # Orch answers plain chat in text, but when the client offers
            # tools it may reply with native OpenAI-compatible tool calls.
            row["tool_protocol"] = "harness-v0"
        rows.append(row)
    return rows
