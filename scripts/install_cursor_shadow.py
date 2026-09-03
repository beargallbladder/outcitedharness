#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import plistlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from harness.shadow.models import DEFAULT_EXCLUDED_PATHS, ModelRuntime, ShadowPolicy
from harness.shadow.policy import POLICY_NAME, load_policy
from harness.shadow.repository import discover_repository


EVENTS = (
    "beforeSubmitPrompt",
    "beforeReadFile",
    "afterFileEdit",
    "afterShellExecution",
    "afterAgentResponse",
    "stop",
)
HOOK_RELATIVE_PATH = Path(".cursor/hooks/shadow-capture")
HOOK_COMMAND = ".cursor/hooks/shadow-capture"
LAUNCH_LABEL = "com.harness.cursor-shadow"
PROCESSOR_LAUNCH_LABEL = "com.harness.cursor-shadow-processor"


def _atomic_write(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), mode)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        temporary.unlink(missing_ok=True)


def _run(argv: list[str], operation: str) -> None:
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:1000]
        raise RuntimeError(f"{operation} failed: {detail}")


def _install_venv(source: Path, venv: Path) -> Path:
    uv = shutil.which("uv")
    if uv:
        if not venv.exists():
            _run([uv, "venv", str(venv)], "shadow virtual environment creation")
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        _run(
            [
                uv,
                "pip",
                "install",
                "--reinstall",
                "--python",
                str(python),
                str(source),
            ],
            "shadow runtime installation",
        )
    else:
        if not venv.exists():
            _run(
                [os.fspath(Path(os.sys.executable)), "-m", "venv", str(venv)],
                "shadow virtual environment creation",
            )
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        _run(
            [str(python), "-m", "pip", "install", "--force-reinstall", str(source)],
            "shadow runtime installation",
        )
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError("shadow virtual-environment Python is not executable")
    return python


def _policy(repository_id: str) -> ShadowPolicy:
    return ShadowPolicy(
        repository_id=repository_id,
        allowed_paths=(".",),
        excluded_paths=DEFAULT_EXCLUDED_PATHS,
    )


def _install_policy(repository: Path, repository_id: str) -> ShadowPolicy:
    current = load_policy(repository)
    if current is not None:
        if current.repository_id != repository_id:
            raise ValueError(
                "existing shadow policy belongs to a different repository identity"
            )
        return current
    policy = _policy(repository_id)
    _atomic_write(
        repository / POLICY_NAME,
        json.dumps(
            policy.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n",
        0o644,
    )
    return policy


def _install_hook(repository: Path) -> None:
    script = """#!/bin/sh
set -eu

PYTHON="${HARNESS_SHADOW_PYTHON:-$HOME/.harness/shadow/venv/bin/python}"
HOOK_DIR=$(CDPATH= cd "$(dirname "$0")" && /bin/pwd -P)
REPOSITORY_ROOT=$(CDPATH= cd "$HOOK_DIR/../.." && /bin/pwd -P)
exec "$PYTHON" -m harness.shadow hook "$1" --repository-root "$REPOSITORY_ROOT"
"""
    _atomic_write(repository / HOOK_RELATIVE_PATH, script.encode(), 0o755)
    hooks_path = repository / ".cursor/hooks.json"
    if hooks_path.exists():
        value = json.loads(hooks_path.read_text())
        if not isinstance(value, dict) or not isinstance(value.get("hooks"), dict):
            raise ValueError("existing Cursor hooks file has an unsupported shape")
        if value.get("version") != 1:
            raise ValueError("existing Cursor hooks file is not schema version 1")
    else:
        value = {"version": 1, "hooks": {}}
    hooks = value["hooks"]
    for event in EVENTS:
        rows = hooks.setdefault(event, [])
        if not isinstance(rows, list):
            raise ValueError(f"existing Cursor hook {event} is not a list")
        command = f"{HOOK_COMMAND} {event}"
        if not any(
            isinstance(row, dict) and row.get("command") == command for row in rows
        ):
            rows.append(
                {
                    "command": command,
                    "timeout": 15 if event == "beforeSubmitPrompt" else 5,
                    "failClosed": False,
                }
            )
    _atomic_write(
        hooks_path,
        json.dumps(value, indent=2, sort_keys=True).encode() + b"\n",
        0o644,
    )


def _install_runtime(
    *,
    shadow_root: Path,
    base_url: str,
    model: str,
    api_key_env: str | None,
) -> Path:
    api_key_file: Path | None = None
    if api_key_env:
        candidate = shadow_root / "secrets" / "model-api-key"
        key = os.environ.get(api_key_env)
        if key:
            if any(character in key for character in "\r\n\x00"):
                raise ValueError("model API key contains an invalid control character")
            candidate.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(candidate.parent, 0o700)
            _atomic_write(candidate, (key + "\n").encode(), 0o600)
        if candidate.is_file() and not candidate.is_symlink():
            api_key_file = candidate
    runtime = ModelRuntime(
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        api_key_file=api_key_file,
        spool_root=shadow_root,
        work_root=shadow_root / "work",
    )
    path = shadow_root / "runtime.json"
    _atomic_write(
        path,
        json.dumps(
            runtime.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n",
        0o600,
    )
    return path


def _install_launchd(
    *,
    python: Path,
    runtime: Path,
    shadow_root: Path,
    activate: bool,
) -> Path:
    path = Path.home() / "Library/LaunchAgents" / f"{LAUNCH_LABEL}.plist"
    payload: dict[str, Any] = {
        "Label": LAUNCH_LABEL,
        "ProgramArguments": [
            str(python),
            "-m",
            "harness.shadow",
            "worker",
            "--runtime",
            str(runtime),
        ],
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "ThrottleInterval": 10,
        "StandardOutPath": str(shadow_root / "worker.out.log"),
        "StandardErrorPath": str(shadow_root / "worker.err.log"),
    }
    _atomic_write(path, plistlib.dumps(payload, sort_keys=True), 0o600)
    if activate:
        domain = f"gui/{os.getuid()}"
        subprocess.run(
            ["/bin/launchctl", "bootout", domain, str(path)],
            text=True,
            capture_output=True,
            check=False,
        )
        _run(
            ["/bin/launchctl", "bootstrap", domain, str(path)],
            "launchd shadow worker activation",
        )
        _run(
            ["/bin/launchctl", "kickstart", "-k", f"{domain}/{LAUNCH_LABEL}"],
            "launchd shadow worker start",
        )
    return path


def _install_processor_launchd(
    *,
    python: Path,
    shadow_root: Path,
    activate: bool,
) -> Path:
    path = (
        Path.home()
        / "Library/LaunchAgents"
        / f"{PROCESSOR_LAUNCH_LABEL}.plist"
    )
    payload: dict[str, Any] = {
        "Label": PROCESSOR_LAUNCH_LABEL,
        "ProgramArguments": [
            str(python),
            "-m",
            "harness.shadow",
            "process",
            "--spool",
            str(shadow_root),
        ],
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "ThrottleInterval": 10,
        "StandardOutPath": str(shadow_root / "processor.out.log"),
        "StandardErrorPath": str(shadow_root / "processor.err.log"),
    }
    _atomic_write(path, plistlib.dumps(payload, sort_keys=True), 0o600)
    if activate:
        domain = f"gui/{os.getuid()}"
        subprocess.run(
            ["/bin/launchctl", "bootout", domain, str(path)],
            text=True,
            capture_output=True,
            check=False,
        )
        _run(
            ["/bin/launchctl", "bootstrap", domain, str(path)],
            "launchd shadow processor activation",
        )
        _run(
            [
                "/bin/launchctl",
                "kickstart",
                "-k",
                f"{domain}/{PROCESSOR_LAUNCH_LABEL}",
            ],
            "launchd shadow processor start",
        )
    return path


def _install_systemd(
    *,
    python: Path,
    runtime: Path,
    shadow_root: Path,
    activate: bool,
) -> Path:
    path = Path.home() / ".config/systemd/user/harness-cursor-shadow.service"
    body = f"""[Unit]
Description=Harness local Qwen shadow worker

[Service]
Type=simple
ExecStart={python} -m harness.shadow worker --runtime {runtime}
Restart=always
RestartSec=10
WorkingDirectory={shadow_root}
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=default.target
"""
    _atomic_write(path, body.encode(), 0o600)
    if activate:
        _run(["systemctl", "--user", "daemon-reload"], "systemd reload")
        _run(
            ["systemctl", "--user", "enable", "--now", path.name],
            "systemd shadow worker activation",
        )
    return path


def _install_processor_systemd(
    *,
    python: Path,
    shadow_root: Path,
    activate: bool,
) -> Path:
    path = (
        Path.home()
        / ".config/systemd/user/harness-cursor-shadow-processor.service"
    )
    body = f"""[Unit]
Description=Harness shadow replay and learning processor

[Service]
Type=simple
ExecStart={python} -m harness.shadow process --spool {shadow_root}
Restart=always
RestartSec=10
WorkingDirectory={shadow_root}
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=default.target
"""
    _atomic_write(path, body.encode(), 0o600)
    if activate:
        _run(["systemctl", "--user", "daemon-reload"], "systemd reload")
        _run(
            ["systemctl", "--user", "enable", "--now", path.name],
            "systemd shadow processor activation",
        )
    return path


def install(arguments: argparse.Namespace) -> dict[str, Any]:
    repository = discover_repository(Path(arguments.repository))
    shadow_root = Path(arguments.shadow_root).expanduser().resolve()
    shadow_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(shadow_root, 0o700)
    policy = _install_policy(repository, arguments.repository_id)
    _install_hook(repository)
    if arguments.skip_runtime_install:
        python = shadow_root / "venv" / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        if not python.is_file() or not os.access(python, os.X_OK):
            raise RuntimeError("preinstalled shadow runtime Python is unavailable")
    else:
        source = discover_repository(Path(arguments.harness_source))
        python = _install_venv(source, shadow_root / "venv")
    runtime = _install_runtime(
        shadow_root=shadow_root,
        base_url=arguments.base_url,
        model=arguments.model,
        api_key_env=arguments.api_key_env,
    )
    system = platform.system()
    service: Path | None = None
    processor_service: Path | None = None
    if system == "Darwin":
        service = _install_launchd(
            python=python,
            runtime=runtime,
            shadow_root=shadow_root,
            activate=arguments.activate,
        )
        processor_service = _install_processor_launchd(
            python=python,
            shadow_root=shadow_root,
            activate=arguments.activate,
        )
    elif system == "Linux":
        service = _install_systemd(
            python=python,
            runtime=runtime,
            shadow_root=shadow_root,
            activate=arguments.activate,
        )
        processor_service = _install_processor_systemd(
            python=python,
            shadow_root=shadow_root,
            activate=arguments.activate,
        )
    return {
        "repository": str(repository),
        "repository_id": policy.repository_id,
        "hook": str(repository / HOOK_RELATIVE_PATH),
        "python": str(python),
        "runtime": str(runtime),
        "service": str(service) if service else None,
        "processor_service": (
            str(processor_service) if processor_service else None
        ),
        "activated": bool(arguments.activate),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install an opt-in Cursor-to-Qwen shadow capture client."
    )
    parser.add_argument("--repository", default=".")
    parser.add_argument("--repository-id", required=True)
    parser.add_argument(
        "--harness-source",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--shadow-root",
        default="~/.harness/shadow",
    )
    parser.add_argument(
        "--base-url",
        default="http://100.68.133.1:8888/v1",
    )
    parser.add_argument(
        "--model",
        default="qwen38-flash-next-nvfp4-sglang",
    )
    parser.add_argument("--api-key-env")
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Load and start the per-user background service.",
    )
    parser.add_argument(
        "--skip-runtime-install",
        action="store_true",
        help="Use an already provisioned shadow virtual environment.",
    )
    return parser.parse_args()


def main() -> int:
    result = install(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
