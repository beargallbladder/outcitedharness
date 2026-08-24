#!/usr/bin/env python3
"""Install Cline in VS Code and write the harness OpenAI-compatible provider."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = shutil.which("code") or "/Users/samkim/bin/code"
EXT_ID = "saoudrizwan.claude-dev"
GATEWAY = "http://127.0.0.1:8787/v1"
KEY = "sk-harness-local"
MODEL = "harness-local"


def run(args: list[str]) -> None:
    subprocess.run(args, check=True)


def install_cline() -> None:
    run([CODE, "--install-extension", EXT_ID, "--force"])


def provider_payload() -> dict:
    info = {
        "id": MODEL,
        "label": "Harness local (Spark Coder-Next)",
        "maxTokens": 8192,
        "contextWindow": 131072,
        "supportsImages": False,
        "supportsPromptCache": False,
        "supportsTools": True,
        "inputPrice": 0,
        "outputPrice": 0,
    }
    return {
        "apiProvider": "openai",
        "openAiBaseUrl": GATEWAY,
        "openAiApiKey": KEY,
        "openAiModelId": MODEL,
        "openAiModelInfo": info,
        "planModeApiProvider": "openai",
        "actModeApiProvider": "openai",
        "planModeOpenAiModelId": MODEL,
        "actModeOpenAiModelId": MODEL,
        "planModeOpenAiModelInfo": info,
        "actModeOpenAiModelInfo": info,
    }


def write_hint_files() -> None:
    dest = Path.home() / "Library/Application Support/Code/User/globalStorage" / EXT_ID
    dest.mkdir(parents=True, exist_ok=True)
    settings_dir = dest / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    payload = provider_payload()
    (settings_dir / "harness-provider.json").write_text(json.dumps(payload, indent=2) + "\n")
    (settings_dir / "cline_cli_config.json").write_text(json.dumps(payload, indent=2) + "\n")
    docs = Path.home() / "Documents" / "Cline"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "harness-provider.json").write_text(json.dumps(payload, indent=2) + "\n")


def write_user_settings() -> None:
    user = Path.home() / "Library/Application Support/Code/User/settings.json"
    user.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if user.exists() and user.read_text().strip():
        data = json.loads(user.read_text())
    data.update(
        {
            "cline.preferredLanguage": "English",
        }
    )
    user.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    install_cline()
    write_hint_files()
    write_user_settings()
    print(f"Cline installed. Point it at {GATEWAY} model={MODEL} key={KEY}")


if __name__ == "__main__":
    main()
