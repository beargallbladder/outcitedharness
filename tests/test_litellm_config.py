from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load((ROOT / "config" / "litellm.yaml").read_text())


def test_litellm_routes_are_explicit_and_local_by_default():
    config = _config()
    rows = config["model_list"]
    by_name = {row["model_name"]: row for row in rows}

    assert set(by_name) == {
        "local-coder",
        "local-qwen38",
        "local-critic",
        "harness-orch",
        "frontier-claude",
    }
    assert by_name["local-coder"]["litellm_params"]["api_base"].endswith(":8900/v1")
    assert by_name["local-qwen38"]["litellm_params"]["api_base"].endswith(
        ":8888/v1"
    )
    assert by_name["local-critic"]["litellm_params"]["api_base"].endswith(
        ":8900/v1"
    )
    assert by_name["harness-orch"]["litellm_params"]["api_base"] == (
        "http://127.0.0.1:8787/v1"
    )


def test_paid_route_is_manual_and_secrets_are_environment_backed():
    config = _config()
    rows = config["model_list"]
    frontier = next(row for row in rows if row["model_name"] == "frontier-claude")
    settings = config.get("litellm_settings") or {}

    assert frontier["litellm_params"]["api_key"] == "os.environ/ANTHROPIC_API_KEY"
    assert config["general_settings"]["master_key"] == (
        "os.environ/LITELLM_MASTER_KEY"
    )
    assert "fallbacks" not in settings
    assert "default_fallbacks" not in settings
    assert "context_window_fallbacks" not in settings
