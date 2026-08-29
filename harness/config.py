from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


PROJECT_MARKERS = ("pyproject.toml", "config/models.yaml")


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "config").is_dir():
            return candidate
    env_root = os.environ.get("HARNESS_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return start


class Capabilities(BaseModel):
    vision: bool = False


class ModelConfig(BaseModel):
    key: str
    enabled: bool = True
    tier: int
    display_name: str
    short_name: str
    provider: str
    base_url: str
    model: str
    temperature: float = 0
    timeout_s: float = 180
    api_key_env: str | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    capabilities: Capabilities = Field(default_factory=Capabilities)
    max_tokens: int | None = None
    coder_context_tokens: int | None = Field(default=None, ge=1024, le=200_000)

    @field_validator("base_url")
    @classmethod
    def strip_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env) or None

    @property
    def placeholder_url(self) -> bool:
        return "CHANGE_ME" in (self.base_url or "") or "CHANGE_ME" in (self.model or "")

    @property
    def missing_key(self) -> bool:
        return bool(self.api_key_env) and not self.api_key


class Pricing(BaseModel):
    input_per_million: float | None = None
    output_per_million: float | None = None


class GCIRepositorySource(BaseModel):
    type: Literal["local_git"] = "local_git"
    path: str
    owner: Literal["self"] = "self"
    enabled: bool = True


class GCIRefreshPolicy(BaseModel):
    enabled: bool = False
    active_interval_seconds: int = Field(default=300, ge=60, le=86_400)
    stale_after_days: int = Field(default=30, ge=1, le=3_650)
    stale_interval_seconds: int = Field(default=86_400, ge=300, le=604_800)
    failure_retry_seconds: int = Field(default=900, ge=60, le=86_400)
    jitter_seconds: int = Field(default=0, ge=0, le=3_600)
    state_path: Path = Path("~/.harness/gci-refresh.sqlite")


class Settings(BaseModel):
    results_dir: Path
    db_path: Path
    health_timeout_s: float = 8
    default_timeout_s: float = 180
    tournament_parallel: bool = True
    max_answer_preview_chars: int = 800
    system_prompt: str = ""
    local_revision_attempts: int = Field(default=0, ge=0, le=3)
    auto_frontier_rescue: bool = False
    max_frontier_calls_per_task: int = Field(default=1, ge=0, le=2)
    frontier_model_key: str = "frontier"
    frontier_max_input_chars: int = Field(default=20_000, ge=4_000, le=60_000)
    frontier_max_output_tokens: int = Field(default=2_048, ge=256, le=8_192)
    coder_context_tokens: int = Field(default=6_000, ge=1024, le=200_000)
    checkpoint_max_file_bytes: int = Field(
        default=1_000_000,
        ge=1_024,
        le=100_000_000,
    )
    code_index_path: Path | None = None
    code_index_repos: list[str] = Field(default_factory=list)
    gci_enabled: bool = False
    gci_url: str = "http://100.81.201.24:8810"
    gci_token_env: str = "HARNESS_GCI_TOKEN"
    gci_timeout_s: float = Field(default=8.0, ge=0.5, le=120.0)
    gci_refresh: GCIRefreshPolicy = Field(default_factory=GCIRefreshPolicy)
    greenfield_runs_root: Path = Path("~/.harness/runs")

    @property
    def gci_repository_sources(self) -> tuple[GCIRepositorySource, ...]:
        return tuple(
            GCIRepositorySource(path=path)
            for path in self.code_index_repos
        )


class AppConfig(BaseModel):
    root: Path
    settings: Settings
    models: dict[str, ModelConfig]
    pricing: dict[str, Pricing]

    def enabled_models(self) -> list[ModelConfig]:
        return sorted(
            (m for m in self.models.values() if m.enabled),
            key=lambda m: (m.tier, m.key),
        )

    def models_for_mode(self, mode: str, only: list[str] | None = None) -> list[ModelConfig]:
        models = self.enabled_models()
        if mode == "baseline":
            models = [m for m in models if m.tier >= 4]
        if only:
            wanted = set(only)
            missing = wanted - {m.key for m in self.models.values()}
            if missing:
                raise ValueError(f"Unknown model keys: {sorted(missing)}")
            models = [m for m in self.models.values() if m.key in wanted]
            models = sorted(models, key=lambda m: (m.tier, m.key))
        return models

    def pricing_for(self, model_key: str) -> Pricing:
        return self.pricing.get(model_key, Pricing())


def _optional_path(root: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_config(root: Path | None = None) -> AppConfig:
    root = find_project_root(root)
    load_dotenv(root / ".env", override=False)

    raw_settings = _read_yaml(root / "config" / "settings.yaml")
    results_dir = (root / raw_settings.get("results_dir", "results")).resolve()
    db_path_value = raw_settings.get("db_path", "results/harness.db")
    db_path = Path(db_path_value)
    if not db_path.is_absolute():
        db_path = (root / db_path).resolve()

    refresh_raw = raw_settings.get("gci_refresh")
    if not isinstance(refresh_raw, dict):
        refresh_raw = {}
    refresh_state_path = Path(
        str(refresh_raw.get("state_path") or "~/.harness/gci-refresh.sqlite")
    ).expanduser()
    settings = Settings(
        results_dir=results_dir,
        db_path=db_path,
        health_timeout_s=float(raw_settings.get("health_timeout_s", 8)),
        default_timeout_s=float(raw_settings.get("default_timeout_s", 180)),
        tournament_parallel=bool(raw_settings.get("tournament_parallel", True)),
        max_answer_preview_chars=int(raw_settings.get("max_answer_preview_chars", 800)),
        system_prompt=str(raw_settings.get("system_prompt") or "").strip(),
        local_revision_attempts=int(raw_settings.get("local_revision_attempts", 1)),
        auto_frontier_rescue=bool(raw_settings.get("auto_frontier_rescue", True)),
        max_frontier_calls_per_task=int(raw_settings.get("max_frontier_calls_per_task", 1)),
        frontier_model_key=str(raw_settings.get("frontier_model_key") or "frontier"),
        frontier_max_input_chars=int(raw_settings.get("frontier_max_input_chars", 20_000)),
        frontier_max_output_tokens=int(raw_settings.get("frontier_max_output_tokens", 2_048)),
        coder_context_tokens=int(raw_settings.get("coder_context_tokens", 6_000)),
        checkpoint_max_file_bytes=int(
            raw_settings.get("checkpoint_max_file_bytes", 1_000_000)
        ),
        code_index_path=_optional_path(root, raw_settings.get("code_index_path")),
        code_index_repos=[str(p) for p in (raw_settings.get("code_index_repos") or []) if str(p).strip()],
        gci_enabled=bool(raw_settings.get("gci_enabled", False)),
        gci_url=str(raw_settings.get("gci_url") or "http://100.81.201.24:8810"),
        gci_token_env=str(raw_settings.get("gci_token_env") or "HARNESS_GCI_TOKEN"),
        gci_timeout_s=float(raw_settings.get("gci_timeout_s", 8.0)),
        gci_refresh=GCIRefreshPolicy(
            enabled=bool(refresh_raw.get("enabled", False)),
            active_interval_seconds=int(
                refresh_raw.get("active_interval_seconds", 300)
            ),
            stale_after_days=int(refresh_raw.get("stale_after_days", 30)),
            stale_interval_seconds=int(
                refresh_raw.get("stale_interval_seconds", 86_400)
            ),
            failure_retry_seconds=int(
                refresh_raw.get("failure_retry_seconds", 900)
            ),
            jitter_seconds=int(refresh_raw.get("jitter_seconds", 0)),
            state_path=refresh_state_path,
        ),
        greenfield_runs_root=Path(
            str(raw_settings.get("greenfield_runs_root") or "~/.harness/runs")
        ).expanduser(),
    )

    raw_models = _read_yaml(root / "config" / "models.yaml").get("models") or {}
    models: dict[str, ModelConfig] = {}
    for key, spec in raw_models.items():
        models[key] = ModelConfig(key=key, **spec)

    raw_pricing = _read_yaml(root / "config" / "pricing.yaml").get("pricing") or {}
    pricing = {key: Pricing(**(values or {})) for key, values in raw_pricing.items()}

    return AppConfig(root=root, settings=settings, models=models, pricing=pricing)
