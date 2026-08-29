from __future__ import annotations

import json
import re
from pathlib import Path

from harness.greenfield.models import GreenfieldManifest
from harness.greenfield.package_policy import PackageAction, execute_package_action
from harness.greenfield.stacks.base import StackAdapter
from harness.repo_contract import RepoContract


class PythonAdapter(StackAdapter):
    stack = "python"

    def bootstrap(self, root: Path, manifest: GreenfieldManifest) -> RepoContract:
        module = re.sub(r"[^a-z0-9_]", "_", manifest.project_name.lower())
        dependencies = ",\n    ".join(
            json.dumps(item) for item in manifest.approved_dependencies
        )
        dependency_block = f"\n    {dependencies},\n" if dependencies else ""
        self.write_file(
            root,
            "pyproject.toml",
            f"""[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = {json.dumps(manifest.project_name)}
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [{dependency_block}]

[project.optional-dependencies]
dev = ["pytest", "ruff"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
""",
        )
        self.write_file(
            root,
            f"src/{module}/__init__.py",
            f'"""Public API for {manifest.project_name}."""\n\n'
            "from .app import health\n\n"
            '__all__ = ["health"]\n',
        )
        self.write_file(
            root,
            f"src/{module}/app.py",
            '"""Initial mechanically verified application surface."""\n\n'
            "def health() -> dict[str, str]:\n"
            '    return {"status": "ok"}\n',
        )
        self.write_file(
            root,
            "tests/test_smoke.py",
            f"from {module} import health\n\n\n"
            "def test_health_smoke():\n"
            '    assert health() == {"status": "ok"}\n',
        )
        self.write_file(
            root,
            ".harness.toml",
            """[verification]
required = ["unit", "lint"]

[verification.commands.unit]
argv = [".venv/bin/pytest", "-q"]
timeout = 60

[verification.commands.lint]
argv = [".venv/bin/ruff", "check", "."]
timeout = 60
""",
        )
        self.write_file(
            root,
            ".gitignore",
            ".venv/\n__pycache__/\n.pytest_cache/\n.ruff_cache/\n*.pyc\n.env\n",
        )
        self.write_file(
            root,
            ".env.example",
            "# Add documented non-secret configuration only when required.\n",
        )
        self.write_file(
            root,
            "README.md",
            f"# {manifest.project_name}\n\n"
            "Autonomously bootstrapped and mechanically verified by Harness.\n",
        )
        execute_package_action(
            PackageAction("uv", ("uv", "sync", "--extra", "dev"), root),
            repo_root=root,
            approved_dependencies=manifest.approved_dependencies,
        )
        contract = self.contract(root)
        self.verify(contract)
        return contract
