from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from harness.cli import app
from harness.greenfield.controller import GreenfieldController
from harness.repo_contract import build_repo_contract


class FakeAdapter:
    def bootstrap(self, root: Path, _manifest):
        (root / "src").mkdir()
        (root / "src" / "app.py").write_text("VALUE = 1\n")
        (root / "pyproject.toml").write_text(
            """[project]
name = "gateway-sample"
version = "0.1.0"
dependencies = []

[tool.pytest.ini_options]
testpaths = ["tests"]
"""
        )
        (root / ".harness.toml").write_text(
            """[verification]
required = ["unit"]

[verification.commands.unit]
argv = ["python", "-m", "pytest", "-q"]
"""
        )
        return build_repo_contract(root)

    @staticmethod
    def contract(root):
        return build_repo_contract(root)

    @staticmethod
    def verify(_contract):
        return []


def _cfg(tmp_path: Path):
    return SimpleNamespace(
        root=tmp_path,
        settings=SimpleNamespace(
            db_path=tmp_path / "harness.sqlite",
            results_dir=tmp_path / "results",
            checkpoint_max_file_bytes=1_000_000,
            greenfield_runs_root=tmp_path / "runs",
            gci_enabled=False,
            gci_token_env="NO_TOKEN",
            gci_url="http://spark.test:8810",
            gci_timeout_s=1,
        ),
    )


def test_build_cli_help_lists_lifecycle_commands():
    result = CliRunner().invoke(app, ["build", "--help"])
    assert result.exit_code == 0
    for command in (
        "new",
        "approve",
        "status",
        "resume",
        "run",
        "retry",
        "cancel",
        "rollback",
        "publish",
    ):
        assert command in result.stdout


@pytest.mark.asyncio
async def test_gateway_refuses_greenfield_in_wrong_workspace(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    monkeypatch.setattr("harness.greenfield.controller.adapter_for", lambda _stack: FakeAdapter())
    controller = GreenfieldController(cfg)
    planned = controller.plan(
        intent="Build a Python service with a mechanically tested health endpoint",
        name="gateway-sample",
        stack="python",
        destination=tmp_path / "published",
        discovery_search=lambda *_args, **_kwargs: [],
    )
    active = controller.approve_and_provision(planned.run_id)
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    result = await run_orch(
        cfg,
        f"Continue greenfield {active.run_id}",
        messages=[],
        tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        extra={"workspace_root": str(wrong)},
    )
    assert result.error == "greenfield workspace mismatch"
    assert str(active.workspace_root) in result.text
    assert not result.tool_calls


@pytest.mark.asyncio
async def test_gateway_binds_active_milestone_task_to_isolated_workspace(
    tmp_path: Path,
    monkeypatch,
):
    from harness.gateway.orch import OrchResult, _run_greenfield_orch

    cfg = _cfg(tmp_path)
    monkeypatch.setattr("harness.greenfield.controller.adapter_for", lambda _stack: FakeAdapter())
    controller = GreenfieldController(cfg)
    planned = controller.plan(
        intent="Build a Python service with a mechanically tested health endpoint",
        name="gateway-sample",
        stack="python",
        destination=tmp_path / "published",
        discovery_search=lambda *_args, **_kwargs: [],
    )
    active = controller.approve_and_provision(planned.run_id)
    expected_task = active.milestones[1].task_id
    calls = []

    async def inner(_cfg, intent, **kwargs):
        calls.append((intent, kwargs))
        return OrchResult(text="gather", loop_phase="gather")

    monkeypatch.setattr("harness.gateway.orch.run_orch", inner)
    result = await _run_greenfield_orch(
        cfg,
        active.run_id,
        thread="",
        messages=[],
        tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        extra={"workspace_root": active.workspace_root},
    )
    assert result.loop_phase == "gather"
    assert calls[0][1]["_greenfield_internal"] is True
    assert calls[0][1]["_task_id_override"] == expected_task
    assert calls[0][1]["_workspace_override"] == Path(active.workspace_root)
