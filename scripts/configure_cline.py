#!/usr/bin/env python3
"""Point Cline at the loopback LiteLLM boundary without exposing its key."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CLINE_ROOT = Path.home() / ".cline" / "data"
GLOBAL_STATE = CLINE_ROOT / "globalState.json"
PROVIDERS = CLINE_ROOT / "settings" / "providers.json"
BASE_URL = "http://127.0.0.1:7410/v1"
MODEL = "harness-orch"
EXTENSION = "saoudrizwan.claude-dev"


def _env_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("\"'")
    return ""


def _master_key() -> str:
    value = os.environ.get("LITELLM_MASTER_KEY") or _env_value(
        ROOT / ".env", "LITELLM_MASTER_KEY"
    )
    if not value:
        raise SystemExit("LITELLM_MASTER_KEY is missing from the environment or .env")
    return value


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
    backup = path.with_name(f"{path.name}.bak-pre7410-{stamp}")
    shutil.copy2(path, backup)
    os.chmod(backup, 0o600)
    return backup


def _model_info() -> dict[str, Any]:
    return {
        "id": MODEL,
        "label": "Harness orchestration via LiteLLM",
        "maxTokens": 8192,
        "contextWindow": 131072,
        "supportsImages": False,
        "supportsPromptCache": False,
        "supportsTools": True,
        "inputPrice": 0,
        "outputPrice": 0,
    }


def configure(*, dry_run: bool = False) -> tuple[Path | None, Path | None]:
    key = _master_key()
    state = _read_json(GLOBAL_STATE)
    info = _model_info()
    state.update(
        {
            "apiProvider": "openai",
            "openAiBaseUrl": BASE_URL,
            "openAiApiKey": key,
            "openAiModelId": MODEL,
            "openAiModelInfo": info,
            "planModeApiProvider": "openai",
            "actModeApiProvider": "openai",
            "planModeOpenAiModelId": MODEL,
            "actModeOpenAiModelId": MODEL,
            "planModeOpenAiModelInfo": info,
            "actModeOpenAiModelInfo": info,
        }
    )

    providers = _read_json(PROVIDERS)
    provider_map = providers.setdefault("providers", {})
    if not isinstance(provider_map, dict):
        raise SystemExit(f"Invalid provider map: {PROVIDERS}")
    entry = provider_map.setdefault("openai-compatible", {})
    if not isinstance(entry, dict):
        raise SystemExit(f"Invalid openai-compatible provider: {PROVIDERS}")
    settings = entry.setdefault("settings", {})
    if not isinstance(settings, dict):
        raise SystemExit(f"Invalid openai-compatible settings: {PROVIDERS}")
    settings.update(
        {
            "provider": "openai-compatible",
            "apiKey": key,
            "model": MODEL,
            "baseUrl": BASE_URL,
        }
    )
    providers["lastUsedProvider"] = "openai-compatible"

    if dry_run:
        return None, None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    state_backup = _backup(GLOBAL_STATE, stamp)
    providers_backup = _backup(PROVIDERS, stamp)
    _atomic_json(GLOBAL_STATE, state)
    _atomic_json(PROVIDERS, providers)
    return state_backup, providers_backup


def rollback_latest() -> None:
    restored = 0
    for path in (GLOBAL_STATE, PROVIDERS):
        backups = sorted(path.parent.glob(f"{path.name}.bak-pre7410-*"))
        if not backups:
            continue
        shutil.copy2(backups[-1], path)
        os.chmod(path, 0o600)
        restored += 1
    if restored != 2:
        raise SystemExit("Could not find both Cline backups")


def install_extension() -> None:
    executable = shutil.which("cursor") or shutil.which("code")
    if not executable:
        raise SystemExit("Neither cursor nor code CLI is available")
    subprocess.run(
        [executable, "--install-extension", EXTENSION, "--force"],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rollback-latest", action="store_true")
    parser.add_argument("--install-extension", action="store_true")
    args = parser.parse_args()
    if args.rollback_latest:
        rollback_latest()
        print("Restored the latest pre-7410 Cline provider backups.")
        return
    if args.install_extension:
        install_extension()
    backups = configure(dry_run=args.dry_run)
    if args.dry_run:
        print(f"Cline configuration is valid for {BASE_URL} model={MODEL}.")
    else:
        print(f"Cline now uses {BASE_URL} model={MODEL}.")
        print("Backups: " + ", ".join(str(path) for path in backups if path))


if __name__ == "__main__":
    main()
