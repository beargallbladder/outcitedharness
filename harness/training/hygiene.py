from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable, Iterable
from typing import TypeVar

from harness.training.models import TextPair
from harness.training.security import (
    SecretDetectedError,
    SecretFinding,
    assert_no_secrets,
    find_secrets,
    redact_text,
)


T = TypeVar("T")

_SPACE = re.compile(r"\s+")


def normalize_for_dedupe(text: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text).strip()).casefold()


def content_fingerprint(*parts: str) -> str:
    canonical = "\0".join(normalize_for_dedupe(part) for part in parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def deduplicate(
    records: Iterable[T],
    *,
    key: Callable[[T], str],
) -> list[T]:
    """Keep the first record for each normalized, deterministic content key."""

    seen: set[str] = set()
    output: list[T] = []
    for record in records:
        fingerprint = content_fingerprint(key(record))
        if fingerprint not in seen:
            seen.add(fingerprint)
            output.append(record)
    return output


def deduplicate_pairs(records: Iterable[TextPair]) -> list[TextPair]:
    seen: set[str] = set()
    output: list[TextPair] = []
    for record in records:
        fingerprint = content_fingerprint(record.prompt, record.response)
        if fingerprint not in seen:
            seen.add(fingerprint)
            output.append(record)
    return output


__all__ = [
    "SecretDetectedError",
    "SecretFinding",
    "assert_no_secrets",
    "content_fingerprint",
    "deduplicate",
    "deduplicate_pairs",
    "find_secrets",
    "normalize_for_dedupe",
    "redact_text",
]
