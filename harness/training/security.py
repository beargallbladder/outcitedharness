from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class SecretDetectedError(ValueError):
    """Raised when material containing a credential is offered for training."""


@dataclass(frozen=True)
class SecretFinding:
    kind: str
    start: int
    end: int


_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("openai_token", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("anthropic_token", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    (
        "jwt",
        re.compile(
            r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
        ),
    ),
    (
        "credentialed_url",
        re.compile(r"(?i)\bhttps?://[^/\s:@]+:[^/\s@]+@[^/\s]+"),
    ),
    ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    (
        "assigned_secret",
        re.compile(
            r"""(?ix)
            \b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|
            password|passwd)\b["']?\s*[:=]\s*["']?
            (?!redacted\b|removed\b|unknown\b|none\b|null\b|example\b|test\b)
            [A-Za-z0-9._~+/=-]{8,}
            """
        ),
    ),
)

_SECRET_FIELD_KEYS = {
    "apikey",
    "accesstoken",
    "refreshtoken",
    "authtoken",
    "authorization",
    "clientsecret",
    "password",
    "passwd",
    "privatekey",
}
_SAFE_PLACEHOLDERS = {
    "",
    "example",
    "none",
    "null",
    "redacted",
    "removed",
    "test",
    "unknown",
}

_PII_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "email",
        re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])"),
        "[REDACTED_EMAIL]",
    ),
    (
        "ipv4",
        re.compile(
            r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"
        ),
        "[REDACTED_IP]",
    ),
    (
        "phone",
        re.compile(
            r"(?<!\w)(?!(?:19|20)\d{2}-\d{2}-\d{2}\b)"
            r"(?:\+?\d[\d .()-]{7,}\d)(?!\w)"
        ),
        "[REDACTED_PHONE]",
    ),
)


def find_secrets(text: str) -> tuple[SecretFinding, ...]:
    findings: list[SecretFinding] = []
    for kind, pattern in _SECRET_PATTERNS:
        findings.extend(
            SecretFinding(kind=kind, start=match.start(), end=match.end())
            for match in pattern.finditer(text)
        )
    return tuple(sorted(findings, key=lambda row: (row.start, row.end, row.kind)))


def assert_no_secrets(text: str, *, field: str = "text") -> None:
    findings = find_secrets(text)
    if findings:
        kinds = ", ".join(sorted({finding.kind for finding in findings}))
        raise SecretDetectedError(f"{field} contains secret material ({kinds})")


def assert_value_no_secrets(value: Any, *, field: str = "value") -> None:
    """Recursively reject credentials hidden in metadata or structured rows."""

    if isinstance(value, str):
        assert_no_secrets(value, field=field)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if normalized_key in _SECRET_FIELD_KEYS and (
                not isinstance(item, str)
                or item.casefold().strip() not in _SAFE_PLACEHOLDERS
            ):
                raise SecretDetectedError(
                    f"{field}.{key} is a credential-bearing field"
                )
            assert_value_no_secrets(item, field=f"{field}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            assert_value_no_secrets(item, field=f"{field}[{index}]")


def redact_text(text: str) -> str:
    """Redact common personal identifiers.

    This helper can also make text safer for retrieval or logging, but redaction is
    not a license to train on credentials: exporters call ``assert_no_secrets``
    before applying these PII substitutions.
    """

    redacted = text
    for _kind, pattern, replacement in _PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    for kind, pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(f"[REDACTED_{kind.upper()}]", redacted)
    return redacted
