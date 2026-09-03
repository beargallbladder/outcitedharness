from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from harness.training.models import SourceKind, is_excluded_learning_source
from harness.training.security import assert_no_secrets, assert_value_no_secrets


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,191}$"),
]
RelativePath = Annotated[str, Field(min_length=1, max_length=1024)]

DEFAULT_EXCLUDED_PATHS = (
    ".env",
    ".env.*",
    ".git",
    ".harness-shadow",
    ".harness-shadow.json",
    ".cursor/hooks.json",
    ".cursor/hooks/**",
    ".venv",
    "**/.env",
    "**/.env.*",
    "**/.git/**",
    "**/.harness-shadow/**",
    "**/.venv/**",
    "**/credentials.json",
    "**/node_modules/**",
    "**/secrets.*",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(timezone.utc)


def _relative(value: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    if (
        not normalized
        or normalized == "."
        or Path(normalized).is_absolute()
        or ".." in Path(normalized).parts
        or "\x00" in normalized
    ):
        raise ValueError("path must be a canonical relative path")
    return normalized


class ShadowPolicy(StrictModel):
    version: Literal[1] = 1
    enabled: Literal[True] = True
    repository_id: Identifier
    owner: Literal["self"] = "self"
    data_use: Literal["shadow_learning"] = "shadow_learning"
    authorization_scope: Identifier = "owned_repository_cursor_shadow"
    teacher_model: Identifier = "gpt-5.6-sol-max-fast"
    local_model_key: Identifier = "asus2_qwen"
    allowed_paths: tuple[str, ...] = (".",)
    excluded_paths: tuple[str, ...] = DEFAULT_EXCLUDED_PATHS
    max_prompt_chars: int = Field(default=60_000, ge=1_000, le=200_000)
    max_dirty_patch_bytes: int = Field(default=2_000_000, ge=1_024, le=20_000_000)
    max_untracked_file_bytes: int = Field(default=256_000, ge=1_024, le=5_000_000)
    max_untracked_total_bytes: int = Field(default=2_000_000, ge=1_024, le=20_000_000)
    max_agent_turns: int = Field(default=8, ge=1, le=24)
    max_context_chars: int = Field(default=80_000, ge=4_000, le=400_000)
    max_attempts: int = Field(default=3, ge=1, le=10)

    @field_validator("allowed_paths", "excluded_paths")
    @classmethod
    def validate_path_patterns(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("path policy cannot be empty")
        normalized = tuple(str(value).strip() for value in values)
        if any(not value or "\x00" in value for value in normalized):
            raise ValueError("path patterns must be non-empty")
        return normalized

    @model_validator(mode="after")
    def enforce_learning_boundary(self) -> ShadowPolicy:
        assert_no_secrets(self.repository_id, field="shadow repository_id")
        assert_no_secrets(self.authorization_scope, field="shadow authorization_scope")
        if is_excluded_learning_source(
            SourceKind.OTHER,
            f"repository://{self.repository_id}",
        ):
            raise ValueError("repository is excluded from learning")
        return self


class ModelRuntime(StrictModel):
    version: Literal[1] = 1
    base_url: Annotated[str, Field(pattern=r"^https?://[^?#]+/v1/?$")]
    model: Identifier
    api_key_env: Annotated[
        str,
        Field(pattern=r"^[A-Z][A-Z0-9_]{1,127}$"),
    ] | None = None
    api_key_file: Path | None = None
    timeout_seconds: float = Field(default=900, ge=10, le=3600)
    spool_root: Path = Path("~/.harness/shadow")
    work_root: Path = Path("~/.harness/shadow/work")

    @field_validator("base_url")
    @classmethod
    def canonical_base_url(cls, value: str) -> str:
        if "@" in value:
            raise ValueError("runtime URL cannot contain credentials")
        return value.rstrip("/") + "/"


class UntrackedFile(StrictModel):
    path: RelativePath
    sha256: Sha256
    byte_size: int = Field(ge=0)
    object_path: RelativePath

    @field_validator("path", "object_path")
    @classmethod
    def validate_relative_paths(cls, value: str) -> str:
        return _relative(value)


class RepositorySnapshot(StrictModel):
    repository_id: Identifier
    repository_root: Path
    revision: Annotated[str, Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")]
    dirty_patch: str = ""
    dirty_patch_sha256: Sha256
    untracked_files: tuple[UntrackedFile, ...] = ()
    omitted_path_count: int = Field(default=0, ge=0)
    state_sha256: Sha256
    captured_at: datetime

    @field_validator("captured_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        return _aware(value)

    @model_validator(mode="after")
    def scan_patch(self) -> RepositorySnapshot:
        assert_no_secrets(self.dirty_patch, field="shadow dirty patch")
        return self


class ShadowTask(StrictModel):
    task_id: Identifier
    correlation_id: Identifier
    session_id: Identifier
    generation_id: Identifier | None = None
    prompt: Annotated[str, Field(min_length=1)]
    prompt_sha256: Sha256
    policy: ShadowPolicy
    snapshot: RepositorySnapshot
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        return _aware(value)

    @model_validator(mode="after")
    def scan_task(self) -> ShadowTask:
        if len(self.prompt) > self.policy.max_prompt_chars:
            raise ValueError("shadow prompt exceeds policy limit")
        assert_no_secrets(self.prompt, field="shadow prompt")
        return self


class HookRecord(StrictModel):
    event_id: Identifier
    task_id: Identifier | None = None
    correlation_id: Identifier
    event_type: Identifier
    payload: dict[str, Any]
    payload_sha256: Sha256
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        return _aware(value)

    @model_validator(mode="after")
    def scan_payload(self) -> HookRecord:
        assert_value_no_secrets(self.payload, field="shadow hook payload")
        return self


class ShadowAttempt(StrictModel):
    attempt_id: Identifier
    task_id: Identifier
    status: Literal["completed", "failed", "quarantined"]
    model: Identifier
    model_endpoint_sha256: Sha256
    answer: str = ""
    patch: str = ""
    transcript: tuple[dict[str, Any], ...] = ()
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    error: str | None = None
    workspace_state_sha256: Sha256 | None = None
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        return _aware(value)

    @model_validator(mode="after")
    def scan_attempt(self) -> ShadowAttempt:
        assert_no_secrets(self.answer, field="shadow answer")
        assert_no_secrets(self.patch, field="shadow patch")
        if self.error is not None:
            assert_no_secrets(self.error, field="shadow error")
        assert_value_no_secrets(self.transcript, field="shadow transcript")
        if self.status == "completed" and not (self.answer or self.patch):
            raise ValueError("completed shadow attempt requires an answer or patch")
        if self.status == "failed" and not self.error:
            raise ValueError("failed shadow attempt requires an error")
        return self


def canonical_json(value: Any) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    return __import__("json").dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def safe_identifier(value: Any, *, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._:/-]+", "-", str(value or "")).strip("-")
    return text[:191] or fallback
