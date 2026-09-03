from __future__ import annotations

import json
import plistlib
import stat
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from install_cursor_shadow import (  # noqa: E402
    EVENTS,
    _install_hook,
    _install_launchd,
    _install_policy,
    _install_processor_launchd,
    _install_runtime,
)


def test_installer_preserves_unrelated_cursor_hooks(tmp_path: Path) -> None:
    cursor = tmp_path / ".cursor"
    cursor.mkdir()
    existing = {
        "version": 1,
        "hooks": {
            "afterFileEdit": [
                {
                    "command": ".cursor/hooks/existing",
                    "failClosed": True,
                }
            ]
        },
    }
    (cursor / "hooks.json").write_text(json.dumps(existing))

    _install_hook(tmp_path)
    _install_hook(tmp_path)

    value = json.loads((cursor / "hooks.json").read_text())
    assert value["hooks"]["afterFileEdit"][0] == existing["hooks"]["afterFileEdit"][0]
    for event in EVENTS:
        rows = value["hooks"][event]
        shadow = [
            row
            for row in rows
            if row.get("command") == f".cursor/hooks/shadow-capture {event}"
        ]
        assert len(shadow) == 1
        assert shadow[0]["failClosed"] is False
    hook = cursor / "hooks/shadow-capture"
    assert hook.stat().st_mode & stat.S_IXUSR
    assert "ruby" not in hook.read_text().casefold()
    assert 'REPOSITORY_ROOT=$(CDPATH= cd "$HOOK_DIR/../.."' in hook.read_text()


def test_installer_writes_opt_in_policy_and_private_runtime(tmp_path: Path) -> None:
    policy = _install_policy(tmp_path, "owner/repository")
    assert policy.enabled
    assert _install_policy(tmp_path, "owner/repository") == policy

    shadow = tmp_path / "shadow"
    runtime = _install_runtime(
        shadow_root=shadow,
        base_url="http://model.internal:8888/v1",
        model="qwen-local",
        api_key_env=None,
    )
    value = json.loads(runtime.read_text())
    assert value["model"] == "qwen-local"
    assert value["api_key_env"] is None
    assert stat.S_IMODE(runtime.stat().st_mode) == 0o600


def test_runtime_uses_private_key_file_for_background_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("QWEN38_API_KEY", "test-model-credential")
    runtime = _install_runtime(
        shadow_root=tmp_path / "shadow",
        base_url="http://model.internal:8888/v1",
        model="qwen-local",
        api_key_env="QWEN38_API_KEY",
    )

    value = json.loads(runtime.read_text())
    key_file = Path(value["api_key_file"])
    assert value["api_key_env"] == "QWEN38_API_KEY"
    assert "test-model-credential" not in runtime.read_text()
    assert key_file.read_text().strip() == "test-model-credential"
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600


def test_launch_agent_uses_absolute_python_and_no_embedded_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")

    path = _install_launchd(
        python=python,
        runtime=runtime,
        shadow_root=tmp_path,
        activate=False,
    )
    payload = plistlib.loads(path.read_bytes())
    assert payload["ProgramArguments"][0] == str(python)
    assert payload["ProgramArguments"][-1] == str(runtime)
    assert "EnvironmentVariables" not in payload
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    processor_path = _install_processor_launchd(
        python=python,
        shadow_root=tmp_path,
        activate=False,
    )
    processor = plistlib.loads(processor_path.read_bytes())
    assert processor["ProgramArguments"] == [
        str(python),
        "-m",
        "harness.shadow",
        "process",
        "--spool",
        str(tmp_path),
    ]
    assert "EnvironmentVariables" not in processor
