from __future__ import annotations

from pathlib import Path

import pytest

from harness.greenfield.models import GreenfieldManifest
from harness.greenfield.stacks.base import StackAdapter, StackAdapterError
from harness.greenfield.stacks.node_typescript import NodeTypeScriptAdapter
from harness.greenfield.stacks.python import PythonAdapter


def _manifest(tmp_path: Path, stack: str) -> GreenfieldManifest:
    return GreenfieldManifest(
        run_id="gf-stack",
        project_name="sample-service",
        stack=stack,
        runtime="python>=3.11" if stack == "python" else "node>=20",
        package_manager="uv" if stack == "python" else "npm",
        approved_dependencies=(),
        destination=str(tmp_path / "published"),
        destination_fingerprint="dest",
        spec_hash="spec",
        plan_hash="plan",
        discovery_hash="discovery",
    )


def test_python_adapter_generates_repo_contract(tmp_path: Path, monkeypatch):
    root = tmp_path / "python"
    root.mkdir()
    monkeypatch.setattr(
        "harness.greenfield.stacks.python.execute_package_action",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(StackAdapter, "verify", lambda *_args, **_kwargs: [])
    contract = PythonAdapter().bootstrap(root, _manifest(tmp_path, "python"))
    assert [command.name for command in contract.commands] == ["unit", "lint"]
    assert (root / "src" / "sample_service" / "app.py").is_file()
    assert (root / "tests" / "test_smoke.py").is_file()


def test_node_adapter_generates_repo_contract(tmp_path: Path, monkeypatch):
    root = tmp_path / "node"
    root.mkdir()
    monkeypatch.setattr(
        "harness.greenfield.stacks.node_typescript.execute_package_action",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(StackAdapter, "verify", lambda *_args, **_kwargs: [])
    contract = NodeTypeScriptAdapter().bootstrap(
        root,
        _manifest(tmp_path, "node-typescript"),
    )
    assert [command.name for command in contract.commands] == [
        "test",
        "lint",
        "typecheck",
    ]
    assert (root / "src" / "index.ts").is_file()
    assert (root / "tests" / "smoke.test.ts").is_file()


def test_bootstrap_resume_refuses_divergent_generated_file(tmp_path: Path, monkeypatch):
    root = tmp_path / "python"
    root.mkdir()
    (root / "README.md").write_text("user changed this\n")
    monkeypatch.setattr(
        "harness.greenfield.stacks.python.execute_package_action",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(StackAdapterError, match="divergent"):
        PythonAdapter().bootstrap(root, _manifest(tmp_path, "python"))
