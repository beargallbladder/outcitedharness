from pathlib import Path

import yaml

from harness.config import find_project_root
from harness.workers.registry import load_registry
from harness.workers.router import should_failover


def test_repo_registry_preserves_auto_ladder():
    root = find_project_root()
    registry = load_registry(root)
    cline = yaml.safe_load((root / "config" / "cline.yaml").read_text())
    assert registry.failover_keys() == list(cline["auto_ladder"])
    assert registry.failover_keys() == ["dgx_qwen", "m5_qwen", "frontier"]


def test_primary_and_fallback_are_the_live_boxes():
    registry = load_registry(find_project_root())
    primary = registry.get("primary_coder")
    fallback = registry.get("fallback_reasoner")
    assert primary is not None and primary.enabled and primary.model_key == "dgx_qwen"
    assert fallback is not None and fallback.enabled and fallback.model_key == "m5_qwen"
    assert "coding" in primary.capabilities
    assert "tool_calling" in primary.capabilities
    assert "long_context" in primary.capabilities


def test_future_workers_are_unavailable_not_missing():
    registry = load_registry(find_project_root())
    by_id = {row["id"]: row for row in registry.summary()}
    for name in ("secondary", "fast", "monster"):
        assert name in by_id
        assert by_id[name]["enabled"] is False
        assert by_id[name]["status"] == "unavailable"
        assert by_id[name]["detail"] == f"{name} unavailable"


def test_empty_root_does_not_walk_into_the_repo(tmp_path: Path):
    registry = load_registry(tmp_path)
    assert registry.failover_keys() == []
    assert registry.summary() == []


def test_should_failover_only_on_dead_box():
    assert should_failover(502, "connect failed", has_next=True) is True
    assert should_failover(200, "empty completion", has_next=True) is True
    assert should_failover(200, None, has_next=True) is False
    assert should_failover(502, "connect failed", has_next=False) is False


def test_healthz_exposes_registry(tmp_path: Path):
    dest = tmp_path / "config"
    dest.mkdir()
    dest.joinpath("workers.yaml").write_text(
        (find_project_root() / "config" / "workers.yaml").read_text()
    )
    from starlette.testclient import TestClient

    from harness.config import AppConfig, ModelConfig, Settings
    from harness.gateway.server import create_app
    from harness.gateway.spec import ClineSpec

    def model(key: str, provider: str, url: str) -> ModelConfig:
        return ModelConfig(
            key=key,
            tier=0,
            display_name=key,
            short_name=key,
            provider=provider,
            base_url=url,
            model=key,
        )

    cfg = AppConfig(
        root=tmp_path,
        settings=Settings(results_dir=tmp_path / "results", db_path=tmp_path / "h.db"),
        models={
            "dgx_qwen": model("dgx_qwen", "openai_compatible", "http://192.168.4.38:8900/v1"),
            "m5_qwen": model("m5_qwen", "openai_compatible", "http://127.0.0.1:8082/v1"),
            "frontier": model("frontier", "anthropic", "https://api.anthropic.com/v1"),
        },
        pricing={},
    )
    spec = ClineSpec(
        listen_host="127.0.0.1",
        listen_port=8787,
        api_key="harness-local",
        aliases={"harness-auto": "auto"},
        auto_ladder=["dgx_qwen", "m5_qwen", "frontier"],
        context_window=131072,
        max_output_tokens=8192,
    )
    body = TestClient(create_app(cfg, spec)).get("/healthz").json()
    assert body["auto_ladder"] == ["dgx_qwen", "m5_qwen", "frontier"]
    by_id = {w["id"]: w for w in body["workers"]}
    assert by_id["primary_coder"]["status"] == "healthy"
    assert by_id["secondary"]["detail"] == "secondary unavailable"
