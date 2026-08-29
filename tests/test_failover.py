"""Lock the internal worker ladder retained by Harness orchestration."""

from __future__ import annotations

from pathlib import Path

from harness.config import AppConfig, ModelConfig, Settings
from harness.gateway.spec import GatewaySpec, ladder_for


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


def _spec() -> GatewaySpec:
    return GatewaySpec(
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
