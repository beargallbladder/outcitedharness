from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "configure_cline.py"
SPEC = importlib.util.spec_from_file_location("configure_cline", SCRIPT)
assert SPEC and SPEC.loader
configure_cline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(configure_cline)


def test_cline_profiles_pin_travel_gateway_and_direct_fallback() -> None:
    assert configure_cline.TRAVEL_BASE_URL == (
        "https://m5max-ai.tail61e9a0.ts.net/v1"
    )
    assert configure_cline.DIRECT_BASE_URL == "http://100.68.133.1:8888/v1"
    assert configure_cline.DEFAULT_MODEL == "local-qwen38"
    assert configure_cline.DIRECT_MODEL == (
        "qwen38-flash-next-nvfp4-sglang"
    )
    assert configure_cline.EXTENSION == "saoudrizwan.claude-dev@4.1.16"

    info = configure_cline._model_info(configure_cline.DEFAULT_MODEL)
    assert info["contextWindow"] == 262144
    assert info["maxTokens"] == 8192
    assert info["supportsTools"] is True
    assert info["supportsImages"] is False
    assert info["supportsPromptCache"] is False


def test_configure_writes_active_provider_and_model_catalog(
    tmp_path: Path, monkeypatch
) -> None:
    state = tmp_path / "globalState.json"
    secrets = tmp_path / "secrets.json"
    providers = tmp_path / "settings" / "providers.json"
    models = tmp_path / "settings" / "models.json"
    monkeypatch.setattr(configure_cline, "GLOBAL_STATE", state)
    monkeypatch.setattr(configure_cline, "SECRETS", secrets)
    monkeypatch.setattr(configure_cline, "PROVIDERS", providers)
    monkeypatch.setattr(configure_cline, "MODELS", models)

    configure_cline.configure(api_key="sk-m4-test")

    state_data = json.loads(state.read_text())
    secrets_data = json.loads(secrets.read_text())
    provider_data = json.loads(providers.read_text())
    model_data = json.loads(models.read_text())
    assert state_data["openAiBaseUrl"] == configure_cline.TRAVEL_BASE_URL
    assert state_data["openAiModelId"] == configure_cline.DEFAULT_MODEL
    assert secrets_data["openAiApiKey"] == "sk-m4-test"
    settings = provider_data["providers"]["openai"]["settings"]
    assert settings["baseUrl"] == configure_cline.TRAVEL_BASE_URL
    assert settings["model"] == configure_cline.DEFAULT_MODEL
    assert settings["apiKey"] == "sk-m4-test"
    assert provider_data["lastUsedProvider"] == "openai"
    legacy_settings = provider_data["providers"]["openai-compatible"]["settings"]
    assert legacy_settings["apiKey"] == "sk-m4-test"
    catalog = model_data["providers"]["openai"]["models"]
    assert set(catalog) == {
        "local-qwen38",
        "local-coder",
        "harness-orch",
    }
    assert set(model_data["providers"]["openai-compatible"]["models"]) == set(
        catalog
    )
    assert catalog["local-qwen38"]["contextWindow"] == 262144


def test_direct_profile_is_manual_and_contains_only_qwen(
    tmp_path: Path, monkeypatch
) -> None:
    state = tmp_path / "globalState.json"
    secrets = tmp_path / "secrets.json"
    providers = tmp_path / "settings" / "providers.json"
    models = tmp_path / "settings" / "models.json"
    monkeypatch.setattr(configure_cline, "GLOBAL_STATE", state)
    monkeypatch.setattr(configure_cline, "SECRETS", secrets)
    monkeypatch.setattr(configure_cline, "PROVIDERS", providers)
    monkeypatch.setattr(configure_cline, "MODELS", models)

    configure_cline.configure(mode="direct", api_key="sk-qwen38-test-secret")

    state_data = json.loads(state.read_text())
    secrets_data = json.loads(secrets.read_text())
    provider_data = json.loads(providers.read_text())
    model_data = json.loads(models.read_text())
    assert state_data["openAiBaseUrl"] == configure_cline.DIRECT_BASE_URL
    assert state_data["openAiModelId"] == configure_cline.DIRECT_MODEL
    assert secrets_data["openAiApiKey"] == "sk-qwen38-test-secret"
    settings = provider_data["providers"]["openai"]["settings"]
    assert settings["apiKey"] == "sk-qwen38-test-secret"
    catalog = model_data["providers"]["openai"]["models"]
    assert set(catalog) == {configure_cline.DIRECT_MODEL}


def test_travel_profile_can_select_harness_orchestration(
    tmp_path: Path, monkeypatch
) -> None:
    state = tmp_path / "globalState.json"
    secrets = tmp_path / "secrets.json"
    providers = tmp_path / "settings" / "providers.json"
    models = tmp_path / "settings" / "models.json"
    monkeypatch.setattr(configure_cline, "GLOBAL_STATE", state)
    monkeypatch.setattr(configure_cline, "SECRETS", secrets)
    monkeypatch.setattr(configure_cline, "PROVIDERS", providers)
    monkeypatch.setattr(configure_cline, "MODELS", models)

    configure_cline.configure(
        selected_model="harness-orch",
        api_key="sk-m4-test",
    )

    state_data = json.loads(state.read_text())
    assert state_data["openAiModelId"] == "harness-orch"
