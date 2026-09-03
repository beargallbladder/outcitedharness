from __future__ import annotations

import pytest

from harness.training.security import (
    SecretDetectedError,
    assert_no_secrets,
    assert_value_no_secrets,
    redact_text,
)


@pytest.mark.parametrize(
    "secret",
    [
        "sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
        "hf_abcdefghijklmnopqrstuvwxyz123456",
        "AIzaabcdefghijklmnopqrstuvwxyz123456789",
        "https://user:password@example.com/path",
        "eyJabcdefghijk.abcdefghijkl.abcdefghijkl",
        '"api_key":"abcdefghijklmnopqrstuvwxyz"',
    ],
)
def test_common_provider_and_structured_secrets_are_rejected(secret: str):
    with pytest.raises(SecretDetectedError):
        assert_no_secrets(secret)
    assert secret not in redact_text(secret)


def test_structured_secret_key_rejects_short_unpatterned_value():
    with pytest.raises(SecretDetectedError, match="credential-bearing field"):
        assert_value_no_secrets({"client_secret": "short"})


def test_documented_placeholders_remain_allowed():
    assert_value_no_secrets({"api_key": "example", "password": "redacted"})


def test_phone_redaction_does_not_rewrite_iso_dates():
    assert redact_text("released 2026-08-07") == "released 2026-08-07"
    assert redact_text("call +1 (415) 555-0123") == "call [REDACTED_PHONE]"
