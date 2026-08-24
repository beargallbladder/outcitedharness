from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from harness.config import AppConfig, ModelConfig
from harness.providers.factory import build_provider


@dataclass
class HealthRow:
    key: str
    display_name: str
    status: str
    endpoint: str
    detail: str


def endpoint_label(model: ModelConfig) -> str:
    parsed = urlparse(model.base_url)
    host = parsed.hostname or model.base_url
    port = f":{parsed.port}" if parsed.port else ""
    return f"{host}{port}"


async def check_model(cfg: AppConfig, model: ModelConfig) -> HealthRow:
    label = endpoint_label(model)
    if not model.enabled:
        return HealthRow(model.key, model.display_name, "DISABLED", label, "")
    if model.placeholder_url:
        return HealthRow(model.key, model.display_name, "UNCONFIGURED", label, "CHANGE_ME")
    if model.missing_key:
        return HealthRow(
            model.key,
            model.display_name,
            "MISSING_KEY",
            label,
            f"${model.api_key_env}",
        )
    try:
        provider = build_provider(model)
    except ValueError as exc:
        return HealthRow(model.key, model.display_name, "ERROR", label, str(exc))

    ok, detail = await provider.health(cfg.settings.health_timeout_s)
    return HealthRow(
        model.key,
        model.display_name,
        "OK" if ok else "FAIL",
        label,
        detail,
    )


async def check_all(cfg: AppConfig) -> list[HealthRow]:
    rows = []
    for model in sorted(cfg.models.values(), key=lambda m: (m.tier, m.key)):
        rows.append(await check_model(cfg, model))
    return rows
