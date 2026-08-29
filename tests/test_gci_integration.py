from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from harness.cli import app
from harness.dispatch import default_gather_calls, merge_tool_catalog
from harness.gci.integration import GLOBAL_CONTEXT_MARKER, global_discovery_context
from harness.gci.models import GCIHit


def _settings():
    return SimpleNamespace(
        gci_enabled=True,
        gci_url="http://spark.test:8810",
        gci_token_env="TEST_GCI_TOKEN",
        gci_timeout_s=1.0,
    )


def test_global_context_is_namespaced_and_explicitly_nonactionable(monkeypatch):
    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def search(self, *_args, **_kwargs):
            return [
                GCIHit(
                    repo_id="foreign",
                    source_host="other-host",
                    repo_root="/foreign/repo",
                    revision="abc",
                    state_hash="state-hash",
                    path="src/foreign.py",
                    symbol="foreign",
                    symbol_type="function",
                    start_line=1,
                    end_line=4,
                    score=0.9,
                    match_type="semantic",
                    text="def foreign(): return True",
                )
            ]

    monkeypatch.setenv("TEST_GCI_TOKEN", "secret")
    monkeypatch.setattr("harness.gci.integration.GCIClient", FakeClient)
    context = global_discovery_context(_settings(), "find implementation")
    assert GLOBAL_CONTEXT_MARKER in context
    assert "do not authorize reading or editing" in context
    assert "gci://other-host/foreign/src/foreign.py" in context


def test_default_gather_only_uses_explicitly_bound_gci_paths(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "harness.gci.integration.workspace_paths",
        lambda *_args, **_kwargs: ["src/local.py"],
    )
    calls = default_gather_calls(
        merge_tool_catalog({"read_file": ("path",)}),
        "fix local scoring",
        workspace=tmp_path,
        gci_settings=_settings(),
    )
    arguments = [call["function"]["arguments"] for call in calls]
    assert any("src/local.py" in value for value in arguments)
    assert all("/foreign/" not in value for value in arguments)


def test_greenfield_discovery_paths_never_become_execution_reads(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "harness.task.code_index.gather_paths_for_intent",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "harness.gci.integration.workspace_paths",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("GCI must not run during Greenfield execution")
        ),
    )
    calls = default_gather_calls(
        merge_tool_catalog({"read_file": ("path",)}),
        (
            "GREENFIELD RUN gf_test\n"
            "Implement the approved API milestone.\n\n"
            "GREENFIELD DISCOVERY (ADVISORY ONLY)\n"
            "gci://other-host/foreign/src/foreign.py#1-4\n"
            "/foreign/repo/services/api/main.py"
        ),
        workspace=tmp_path,
        gci_settings=_settings(),
    )
    arguments = [call["function"]["arguments"] for call in calls]
    assert all("foreign.py" not in value for value in arguments)
    assert all("services/api/main.py" not in value for value in arguments)


def test_gci_cli_surface():
    runner = CliRunner()
    result = runner.invoke(app, ["gci", "--help"])
    assert result.exit_code == 0
    for command in (
        "scan",
        "refresh",
        "auto-run",
        "search",
        "status",
        "pause",
        "resume",
        "serve",
    ):
        assert command in result.stdout
