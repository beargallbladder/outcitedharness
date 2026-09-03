from __future__ import annotations

import hashlib
import json
import os
import posixpath
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from harness.training.ledger import (
    ArtifactPayload,
    CaptureResult,
    LearningLedger,
    VerificationPayload,
)
from harness.training.models import (
    LearningEvent,
    SourceKind,
    is_excluded_learning_source,
)
from harness.training.security import assert_value_no_secrets


class ConnectorKind(str, Enum):
    GITHUB = "github"
    GITLAB = "gitlab"
    RENDER = "render"
    VERCEL = "vercel"
    ANTHROPIC_USAGE = "anthropic_usage"
    OPENAI_USAGE = "openai_usage"


DEFAULT_HOSTS: dict[ConnectorKind, frozenset[str]] = {
    ConnectorKind.GITHUB: frozenset({"api.github.com"}),
    ConnectorKind.GITLAB: frozenset({"gitlab.com"}),
    ConnectorKind.RENDER: frozenset({"api.render.com"}),
    ConnectorKind.VERCEL: frozenset({"api.vercel.com"}),
    ConnectorKind.ANTHROPIC_USAGE: frozenset({"api.anthropic.com"}),
    ConnectorKind.OPENAI_USAGE: frozenset({"api.openai.com"}),
}


class ConnectorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    kind: ConnectorKind
    enabled: bool = False
    read_only: Literal[True] = True
    base_url: str
    token_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]+$")
    allowed_paths: tuple[str, ...]
    allowed_hosts: tuple[str, ...] = ()
    timeout_s: float = Field(default=20, gt=0, le=120)
    maximum_response_bytes: int = Field(
        default=10_000_000,
        ge=1_024,
        le=100_000_000,
    )

    @field_validator("allowed_paths")
    @classmethod
    def paths_are_absolute(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values or any(
            not value.startswith("/")
            or ".." in value.split("/")
            or _excluded(value)
            for value in values
        ):
            raise ValueError("allowed_paths must be safe absolute API prefixes")
        return values

    @model_validator(mode="after")
    def endpoint_is_scoped_https(self) -> ConnectorSpec:
        parsed = urlparse(self.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("connector base_url must be a credential-free HTTPS URL")
        default_hosts = set(DEFAULT_HOSTS[self.kind])
        hosts = set(self.allowed_hosts) or default_hosts
        if not hosts.issubset(default_hosts):
            raise ValueError(
                "connector allowed_hosts cannot expand the provider host boundary"
            )
        if parsed.hostname not in hosts:
            raise ValueError(
                f"connector host {parsed.hostname!r} is not explicitly allowed"
            )
        if _excluded(self.base_url):
            raise ValueError("CategoryRank and Tapes connectors are disabled")
        return self


class ReadOnlyConnector:
    def __init__(
        self,
        spec: ConnectorSpec,
        *,
        client: httpx.Client | None = None,
    ):
        self.spec = spec
        self._client = client

    def __enter__(self) -> ReadOnlyConnector:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        return None

    def get_json(
        self,
        resource: str,
        *,
        params: dict[str, str | int | float] | None = None,
    ) -> Any:
        if not self.spec.enabled:
            raise RuntimeError(f"connector {self.spec.name!r} is disabled")
        normalized = _normalize_resource(resource)
        if not any(
            normalized == prefix.rstrip("/")
            or normalized.startswith(f"{prefix.rstrip('/')}/")
            for prefix in self.spec.allowed_paths
        ):
            raise PermissionError("resource is outside connector allowlist")
        if _excluded(normalized):
            raise PermissionError("CategoryRank and Tapes resources are disabled")
        query = dict(params or {})
        assert_value_no_secrets(query, field="connector query")
        if is_excluded_learning_source(
            SourceKind.OTHER,
            normalized,
            query,
        ):
            raise PermissionError("CategoryRank and Tapes query values are disabled")
        token = os.environ.get(self.spec.token_env)
        if not token:
            raise RuntimeError(f"missing ${self.spec.token_env}")
        url = f"{self.spec.base_url.rstrip('/')}{normalized}"
        headers = _headers(self.spec.kind, token)
        owns_client = self._client is None
        client = self._client or httpx.Client(
            timeout=self.spec.timeout_s,
            follow_redirects=False,
        )
        try:
            with client.stream(
                "GET",
                url,
                headers=headers,
                params=query,
            ) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if (
                    content_length is not None
                    and int(content_length) > self.spec.maximum_response_bytes
                ):
                    raise ValueError(
                        "connector response exceeds configured size limit"
                    )
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > self.spec.maximum_response_bytes:
                        raise ValueError(
                            "connector response exceeds configured size limit"
                        )
            value = json.loads(content)
            if is_excluded_learning_source(
                SourceKind.OTHER,
                url,
                value,
            ):
                raise PermissionError(
                    "CategoryRank and Tapes response content is disabled"
                )
            return value
        finally:
            if owns_client:
                client.close()

    def capture_json(
        self,
        ledger: LearningLedger,
        resource: str,
        *,
        params: dict[str, str | int | float] | None = None,
    ) -> CaptureResult:
        value = self.get_json(resource, params=params)
        normalized = _normalize_resource(resource)
        content = json.dumps(value, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        event = LearningEvent(
            event_id=f"connector-{self.spec.name}-{digest[:24]}",
            event_type="external_read_only_capture",
            source_kind=(
                SourceKind.GIT
                if self.spec.kind in {ConnectorKind.GITHUB, ConnectorKind.GITLAB}
                else SourceKind.OTHER
            ),
            source_uri=f"{self.spec.base_url.rstrip('/')}{normalized}",
            source_revision=digest,
            lineage_id=f"connector:{self.spec.name}:{normalized}",
            authorization_scope=(
                f"read-only:{self.spec.name}:{self.spec.token_env}"
            ),
            created_at=datetime.now(timezone.utc),
            metadata={
                "connector": self.spec.name,
                "kind": self.spec.kind.value,
                "query_keys": sorted((params or {}).keys()),
                "disposition": "quarantine",
            },
        )
        return ledger.capture(
            event,
            [
                ArtifactPayload(
                    kind="connector_response",
                    content=content,
                    media_type="application/json",
                )
            ],
            [
                VerificationPayload(
                    kind="http_read",
                    status="unknown",
                    verifier=f"connector:{self.spec.name}",
                    output_kind="connector_response",
                    metadata={
                        "transport_succeeded": True,
                        "proof_scope": "transport_only",
                    },
                )
            ],
        )


def load_connector_specs(path: Path) -> tuple[ConnectorSpec, ...]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    connectors = raw.get("connectors") if isinstance(raw, dict) else None
    if not isinstance(connectors, list):
        raise ValueError("connector config requires a connectors list")
    return tuple(ConnectorSpec.model_validate(row) for row in connectors)


def _normalize_resource(resource: str) -> str:
    if not resource.startswith("/") or "?" in resource or "#" in resource:
        raise ValueError("resource must be an absolute API path without query text")
    normalized = posixpath.normpath(resource)
    if normalized != resource.rstrip("/") and normalized + "/" != resource:
        raise ValueError("resource path is not canonical")
    return normalized


def _excluded(value: str) -> bool:
    return is_excluded_learning_source(
        SourceKind.OTHER,
        value,
    )


def _headers(kind: ConnectorKind, token: str) -> dict[str, str]:
    if kind is ConnectorKind.ANTHROPIC_USAGE:
        return {
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
        }
    return {"Authorization": f"Bearer {token}"}
