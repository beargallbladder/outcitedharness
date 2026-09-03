from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from harness.storage.db import Store
from harness.training.connectors import (
    ConnectorKind,
    ConnectorSpec,
    ReadOnlyConnector,
    load_connector_specs,
)
from harness.training.ledger import LearningLedger


def _spec(**updates) -> ConnectorSpec:
    values = {
        "name": "owned-github",
        "kind": ConnectorKind.GITHUB,
        "enabled": True,
        "base_url": "https://api.github.com",
        "token_env": "TEST_GITHUB_TOKEN",
        "allowed_paths": ("/repos/acme/project/commits",),
    }
    values.update(updates)
    return ConnectorSpec(**values)


def test_connector_allows_only_scoped_https_reads(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"sha": "abc"})

    monkeypatch.setenv("TEST_GITHUB_TOKEN", "test-token")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    connector = ReadOnlyConnector(_spec(), client=client)

    assert connector.get_json("/repos/acme/project/commits") == {"sha": "abc"}
    assert seen == {"method": "GET", "authorization": "Bearer test-token"}
    with pytest.raises(PermissionError):
        connector.get_json("/user")
    with pytest.raises(ValueError):
        connector.get_json("/repos/acme/project/../secrets")


@pytest.mark.parametrize(
    "updates",
    [
        {"base_url": "http://api.github.com"},
        {"base_url": "https://evil.example"},
        {"allowed_hosts": ("api.github.com", "evil.example")},
        {"allowed_paths": ("/categoryrank/export",)},
        {"allowed_paths": ("/category%20rank/export",)},
    ],
)
def test_connector_rejects_untrusted_or_excluded_scope(updates):
    with pytest.raises(ValidationError):
        _spec(**updates)


def test_connector_capture_writes_pointer_only_ledger(tmp_path: Path, monkeypatch):
    payload = {"runs": [{"id": 1, "status": "failed"}]}
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=payload)
        )
    )
    monkeypatch.setenv("TEST_GITHUB_TOKEN", "test-token")
    connector = ReadOnlyConnector(_spec(), client=client)
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")

    capture = connector.capture_json(
        ledger,
        "/repos/acme/project/commits",
        params={"per_page": 1},
    )

    ledger.verify_event(capture.event_id)
    database = (tmp_path / "harness.db").read_bytes().decode(
        "utf-8", errors="ignore"
    )
    assert json.dumps(payload) not in database
    assert "test-token" not in database
    with store.connect() as conn:
        verification = conn.execute(
            """
            SELECT status, metadata_json FROM learning_verifications
            WHERE event_id = ?
            """,
            (capture.event_id,),
        ).fetchone()
    assert verification["status"] == "unknown"
    assert json.loads(verification["metadata_json"])["proof_scope"] == "transport_only"


def test_connector_rejects_excluded_query_and_streams_size_limit(monkeypatch):
    monkeypatch.setenv("TEST_GITHUB_TOKEN", "test-token")
    excluded = ReadOnlyConnector(
        _spec(),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, json={"ok": True})
            )
        ),
    )
    with pytest.raises(PermissionError, match="query values"):
        excluded.get_json(
            "/repos/acme/project/commits",
            params={"scope": "file:///owned/tapes/export"},
        )
    with pytest.raises(PermissionError, match="query values"):
        excluded.get_json(
            "/repos/acme/project/commits",
            params={"scope": "category rank export"},
        )
    excluded_response = ReadOnlyConnector(
        _spec(),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={"source": "Tapes export", "rows": []},
                )
            )
        ),
    )
    with pytest.raises(PermissionError, match="response content"):
        excluded_response.get_json("/repos/acme/project/commits")

    oversized = ReadOnlyConnector(
        _spec(maximum_response_bytes=1024),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    content=b"x" * 1025,
                )
            )
        ),
    )
    with pytest.raises(ValueError, match="size limit"):
        oversized.get_json("/repos/acme/project/commits")


def test_connector_example_is_disabled_and_parseable():
    root = Path(__file__).resolve().parents[1]
    specs = load_connector_specs(root / "config" / "learning-connectors.example.yaml")

    assert len(specs) == 5
    assert all(not spec.enabled and spec.read_only for spec in specs)
