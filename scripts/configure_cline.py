#!/usr/bin/env python3
"""Configure Cline for M5-routed travel use or manual direct-Qwen recovery."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CLINE_ROOT = Path.home() / ".cline" / "data"
GLOBAL_STATE = CLINE_ROOT / "globalState.json"
SECRETS = CLINE_ROOT / "secrets.json"
PROVIDERS = CLINE_ROOT / "settings" / "providers.json"
MODELS = CLINE_ROOT / "settings" / "models.json"
TRAVEL_BASE_URL = "https://m5max-ai.tail61e9a0.ts.net/v1"
DIRECT_BASE_URL = "http://100.68.133.1:8888/v1"
DEFAULT_MODEL = "local-qwen38"
DIRECT_MODEL = "qwen38-flash-next-nvfp4-sglang"
OPENAI_PROVIDER_ID = "openai"
LEGACY_OPENAI_PROVIDER_ID = "openai-compatible"
TRAVEL_MODELS = {
    "local-qwen38": {
        "label": "Qwen3.8 TP2 via M5",
        "maxTokens": 8192,
        "contextWindow": 262144,
    },
    "local-coder": {
        "label": "Qwen3.8 TP2 coding alias via M5",
        "maxTokens": 8192,
        "contextWindow": 262144,
    },
    "harness-orch": {
        "label": "Harness multi-box orchestration via M5",
        "maxTokens": 8192,
        "contextWindow": 131072,
    },
}
EXTENSION_ID = "saoudrizwan.claude-dev"
EXTENSION_VERSION = "4.1.16"
EXTENSION = f"{EXTENSION_ID}@{EXTENSION_VERSION}"
BACKUP_TAG = "pre-cline-profile"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, 0o600)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _backup(path: Path, stamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.bak-{BACKUP_TAG}-{stamp}")
    shutil.copy2(path, backup)
    os.chmod(backup, 0o600)
    return backup


def _model_info(model: str = DIRECT_MODEL) -> dict[str, Any]:
    if model in TRAVEL_MODELS:
        configured = TRAVEL_MODELS[model]
        label = str(configured["label"])
        max_tokens = int(configured["maxTokens"])
        context_window = int(configured["contextWindow"])
    else:
        label = "Direct dual-node Qwen3.8 TP2 SGLang"
        max_tokens = 8192
        context_window = 262144
    return {
        "id": model,
        "label": label,
        "maxTokens": max_tokens,
        "contextWindow": context_window,
        "supportsImages": False,
        "supportsPromptCache": False,
        "supportsTools": True,
        "inputPrice": 0,
        "outputPrice": 0,
    }


def configure(
    *,
    mode: str = "travel",
    selected_model: str | None = None,
    api_key: str | None = None,
    dry_run: bool = False,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    if mode not in {"travel", "direct"}:
        raise SystemExit("mode must be travel or direct")
    if mode == "travel":
        if not api_key:
            raise SystemExit(
                "Travel mode requires --api-key-stdin or M4_CLINE_API_KEY"
            )
        base_url = TRAVEL_BASE_URL
        model = selected_model or DEFAULT_MODEL
        if model not in TRAVEL_MODELS:
            raise SystemExit(f"Unknown travel model: {model}")
        catalog = TRAVEL_MODELS
    else:
        if not api_key:
            raise SystemExit(
                "Direct mode requires --api-key-stdin or QWEN38_API_KEY"
            )
        base_url = DIRECT_BASE_URL
        model = DIRECT_MODEL
        catalog = {
            DIRECT_MODEL: {
                "label": "Direct dual-node Qwen3.8 TP2 SGLang",
                "maxTokens": 8192,
                "contextWindow": 262144,
            }
        }

    state = _read_json(GLOBAL_STATE)
    info = _model_info(model)
    state.update(
        {
            "apiProvider": "openai",
            "openAiBaseUrl": base_url,
            "openAiApiKey": api_key,
            "openAiModelId": model,
            "openAiModelInfo": info,
            "planModeApiProvider": "openai",
            "actModeApiProvider": "openai",
            "planModeOpenAiModelId": model,
            "actModeOpenAiModelId": model,
            "planModeOpenAiModelInfo": info,
            "actModeOpenAiModelInfo": info,
        }
    )

    secrets = _read_json(SECRETS)
    secrets["openAiApiKey"] = api_key

    providers = _read_json(PROVIDERS)
    provider_map = providers.setdefault("providers", {})
    if not isinstance(provider_map, dict):
        raise SystemExit(f"Invalid provider map: {PROVIDERS}")
    for provider_id in (OPENAI_PROVIDER_ID, LEGACY_OPENAI_PROVIDER_ID):
        entry = provider_map.setdefault(provider_id, {})
        if not isinstance(entry, dict):
            raise SystemExit(
                f"Invalid OpenAI-compatible provider: {PROVIDERS}"
            )
        settings = entry.setdefault("settings", {})
        if not isinstance(settings, dict):
            raise SystemExit(
                f"Invalid OpenAI-compatible settings: {PROVIDERS}"
            )
        settings.update(
            {
                "provider": provider_id,
                "apiKey": api_key,
                "model": model,
                "baseUrl": base_url,
            }
        )
    providers["lastUsedProvider"] = OPENAI_PROVIDER_ID

    models = _read_json(MODELS)
    models["version"] = 1
    model_providers = models.setdefault("providers", {})
    if not isinstance(model_providers, dict):
        raise SystemExit(f"Invalid model provider map: {MODELS}")
    for provider_id in (OPENAI_PROVIDER_ID, LEGACY_OPENAI_PROVIDER_ID):
        openai_models = model_providers.setdefault(provider_id, {})
        if not isinstance(openai_models, dict):
            raise SystemExit(
                f"Invalid OpenAI-compatible model map: {MODELS}"
            )
        model_map = openai_models.setdefault("models", {})
        if not isinstance(model_map, dict):
            raise SystemExit(f"Invalid models map: {MODELS}")
        model_map.clear()
        for model_id in catalog:
            model_info = _model_info(model_id)
            model_map[model_id] = {
                key: value
                for key, value in model_info.items()
                if key not in {"id", "label"}
            }

    if dry_run:
        return None, None, None, None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    state_backup = _backup(GLOBAL_STATE, stamp)
    secrets_backup = _backup(SECRETS, stamp)
    providers_backup = _backup(PROVIDERS, stamp)
    models_backup = _backup(MODELS, stamp)
    _atomic_json(GLOBAL_STATE, state)
    _atomic_json(SECRETS, secrets)
    _atomic_json(PROVIDERS, providers)
    _atomic_json(MODELS, models)
    return state_backup, secrets_backup, providers_backup, models_backup


def rollback_latest() -> None:
    restored = 0
    for path in (GLOBAL_STATE, SECRETS, PROVIDERS, MODELS):
        backups = list(path.parent.glob(f"{path.name}.bak-pre-*"))
        if not backups:
            continue
        latest = max(backups, key=lambda candidate: candidate.stat().st_mtime)
        shutil.copy2(latest, path)
        os.chmod(path, 0o600)
        restored += 1
    if restored != 4:
        raise SystemExit("Could not find all four Cline backups")


def install_extension() -> None:
    executables = list(
        dict.fromkeys(
            executable
            for executable in (shutil.which("cursor"), shutil.which("code"))
            if executable
        )
    )
    if not executables:
        raise SystemExit("Neither cursor nor code CLI is available")
    for executable in executables:
        subprocess.run(
            [executable, "--install-extension", EXTENSION, "--force"],
            check=True,
        )


def _editor_extension_host_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", r"Code Helper \(Plugin\)|Cursor Helper \(Plugin\)"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rollback-latest", action="store_true")
    parser.add_argument("--install-extension", action="store_true")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Use the emergency direct-Qwen profile instead of the M5 gateway.",
    )
    parser.add_argument(
        "--api-key-stdin",
        action="store_true",
        help="Read the selected profile's API key from standard input.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(TRAVEL_MODELS),
        help="Select an approved travel model (default: local-qwen38).",
    )
    args = parser.parse_args()
    if args.rollback_latest:
        rollback_latest()
        print("Restored the latest pre-cutover Cline provider backups.")
        return
    if args.install_extension:
        install_extension()
    mode = "direct" if args.direct else "travel"
    api_key = os.environ.get(
        "QWEN38_API_KEY" if mode == "direct" else "M4_CLINE_API_KEY"
    )
    if args.api_key_stdin:
        api_key = sys.stdin.readline().strip()
    if not args.dry_run and _editor_extension_host_running():
        raise SystemExit(
            "Quit VS Code and Cursor completely before changing Cline profiles."
        )
    if args.direct and args.model:
        raise SystemExit("--model cannot be used with --direct")
    backups = configure(
        mode=mode,
        selected_model=args.model,
        api_key=api_key,
        dry_run=args.dry_run,
    )
    selected_base = DIRECT_BASE_URL if mode == "direct" else TRAVEL_BASE_URL
    selected_model = (
        DIRECT_MODEL if mode == "direct" else (args.model or DEFAULT_MODEL)
    )
    if args.dry_run:
        print(
            f"Cline {mode} configuration is valid for "
            f"{selected_base} model={selected_model}."
        )
    else:
        print(f"Cline now uses {selected_base} model={selected_model}.")
        print("Backups: " + ", ".join(str(path) for path in backups if path))


if __name__ == "__main__":
    main()
