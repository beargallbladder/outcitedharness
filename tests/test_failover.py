"""Lock current DGX → M5 (→ frontier) dead-box failover before wrapping it."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from harness.config import AppConfig, ModelConfig, Settings
from harness.gateway.proxy import ProxyResult
from harness.gateway.server import create_app
from harness.gateway.spec import ClineSpec, ladder_for


def _model(key: str, provider: str = "openai_compatible", url: str = "http://example.invalid/v1") -> ModelConfig:
    return ModelConfig(
        key=key,
        tier={"dgx_qwen": 0, "m5_qwen": 1, "frontier": 4}.get(key, 9),
        display_name=key,
        short_name=key,
        provider=provider,
        base_url=url,
        model=key,
        timeout_s=5,
    )


def _cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        root=tmp_path,
        settings=Settings(results_dir=tmp_path / "results", db_path=tmp_path / "harness.db"),
        models={
            "dgx_qwen": _model("dgx_qwen", url="http://192.168.4.38:8900/v1"),
            "m5_qwen": _model("m5_qwen", url="http://127.0.0.1:8082/v1"),
            "frontier": _model("frontier", provider="anthropic", url="https://api.anthropic.com/v1"),
        },
        pricing={},
    )


def _spec() -> ClineSpec:
    return ClineSpec(
        listen_host="127.0.0.1",
        listen_port=8787,
        api_key="harness-local",
        aliases={
            "harness-auto": "auto",
            "harness-local": "dgx_qwen",
            "harness-m5": "m5_qwen",
            "harness-frontier": "frontier",
        },
        auto_ladder=["dgx_qwen", "m5_qwen", "frontier"],
        context_window=131072,
        max_output_tokens=8192,
    )


def test_auto_ladder_is_dgx_then_m5_then_frontier(tmp_path: Path):
    keys = [m.key for m in ladder_for(_spec(), _cfg(tmp_path), "harness-auto")]
    assert keys == ["dgx_qwen", "m5_qwen", "frontier"]


def test_harness_local_is_dgx_only(tmp_path: Path):
    keys = [m.key for m in ladder_for(_spec(), _cfg(tmp_path), "harness-local")]
    assert keys == ["dgx_qwen"]


def test_harness_m5_does_not_include_dgx(tmp_path: Path):
    keys = [m.key for m in ladder_for(_spec(), _cfg(tmp_path), "harness-m5")]
    assert keys == ["m5_qwen"]


def _ok(model_key: str) -> ProxyResult:
    result = ProxyResult()
    result.model_key = model_key
    result.upstream_model = model_key
    result.status = 200
    result.body = b'{"choices":[{"message":{"content":"pong from ' + model_key.encode() + b'"}}]}'
    return result


def _dead(model_key: str, status: int = 502, error: str = "connect failed") -> ProxyResult:
    result = ProxyResult()
    result.model_key = model_key
    result.upstream_model = model_key
    result.status = status
    result.error = error
    result.body = b'{"error":{"message":"' + error.encode() + b'"}}'
    return result


@pytest.fixture
def routed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    script: dict[str, ProxyResult] = {}

    async def fake_openai(model, body, timeout_s):
        calls.append(model.key)
        return script.get(model.key, _ok(model.key))

    async def fake_anthropic(model, body, timeout_s, requested, max_output):
        calls.append(model.key)
        return script.get(model.key, _ok(model.key))

    monkeypatch.setattr("harness.gateway.server.complete_openai", fake_openai)
    monkeypatch.setattr("harness.gateway.server.complete_anthropic", fake_anthropic)
    app = create_app(_cfg(tmp_path), _spec())
    return TestClient(app), calls, script


def test_auto_uses_primary_when_healthy(routed):
    client, calls, _script = routed
    resp = client.post("/v1/chat/completions", json={"model": "harness-auto", "messages": []})
    assert resp.status_code == 200
    assert calls == ["dgx_qwen"]
    assert b"pong from dgx_qwen" in resp.content


def test_auto_fails_over_to_m5_when_dgx_is_dead(routed):
    client, calls, script = routed
    script["dgx_qwen"] = _dead("dgx_qwen")
    resp = client.post("/v1/chat/completions", json={"model": "harness-auto", "messages": []})
    assert resp.status_code == 200
    assert calls == ["dgx_qwen", "m5_qwen"]
    assert b"pong from m5_qwen" in resp.content


def test_auto_fails_over_on_empty_completion(routed):
    client, calls, script = routed
    script["dgx_qwen"] = _dead("dgx_qwen", status=200, error="empty completion")
    resp = client.post("/v1/chat/completions", json={"model": "harness-auto", "messages": []})
    assert resp.status_code == 200
    assert calls == ["dgx_qwen", "m5_qwen"]


def test_local_does_not_fail_over_to_m5(routed):
    client, calls, script = routed
    script["dgx_qwen"] = _dead("dgx_qwen")
    resp = client.post("/v1/chat/completions", json={"model": "harness-local", "messages": []})
    assert resp.status_code == 502
    assert calls == ["dgx_qwen"]
