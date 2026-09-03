from __future__ import annotations

import json
from pathlib import Path

from harness.repo_contract import build_repo_contract


def test_python_contract_prefers_repo_venv(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "demo"
dependencies = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]
""".strip()
    )
    pytest = tmp_path / ".venv" / "bin" / "pytest"
    pytest.parent.mkdir(parents=True)
    pytest.write_text("#!/bin/sh\n")

    contract = build_repo_contract(tmp_path)

    assert contract is not None
    assert contract.languages == ["python"]
    assert [item.command for item in contract.commands] == [".venv/bin/pytest -q"]
    assert contract.known_test_roots == ["tests/"]
    assert "pyproject.toml" in contract.configs
    assert len(contract.fingerprint) == 64


def test_package_contract_selects_all_safe_verifier_scripts(tmp_path: Path):
    (tmp_path / "package.json").write_text(
        json.dumps(
            {
                "scripts": {
                    "test": "vitest",
                    "test:engine": "cd engine && pytest",
                    "lint": "eslint .",
                    "typecheck": "tsc --noEmit",
                    "deploy": "rm -rf production",
                }
            }
        )
    )
    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n")

    contract = build_repo_contract(tmp_path)

    assert contract is not None
    assert [item.command for item in contract.commands] == [
        "pnpm run test",
        "pnpm run test:engine",
        "pnpm run lint",
        "pnpm run typecheck",
    ]
    assert all("deploy" not in item.command for item in contract.commands)


def test_harness_override_replaces_inference_and_sets_timeout(tmp_path: Path):
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest", "lint": "eslint ."}})
    )
    (tmp_path / ".harness.toml").write_text(
        """
[verification]
required = ["unit", "lint", "script"]

[verification.commands.unit]
argv = [".venv/bin/pytest", "-q"]
timeout = 90

[verification.commands.lint]
argv = ["python", "-m", "ruff", "check", "."]
timeout = 30

[verification.commands.script]
argv = ["python3", "tests/shadow/contract.py"]

[verification.commands.unsafe]
argv = ["bash", "-c", "rm -rf /"]
""".strip()
    )

    contract = build_repo_contract(tmp_path)

    assert contract is not None
    assert [(item.name, item.command, item.timeout_s) for item in contract.commands] == [
        ("unit", ".venv/bin/pytest -q", 90),
        ("lint", "python -m ruff check .", 30),
        ("script", "python3 tests/shadow/contract.py", 60),
    ]
    assert all(item.source == ".harness.toml" for item in contract.commands)


def test_single_line_ci_verifier_is_deterministic(tmp_path: Path):
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text(
        """
steps:
  - run: python -m pytest -q
  - run: npm run lint
  - run: echo unsafe | sh
  - run: |
      pytest
""".strip()
    )

    contract = build_repo_contract(tmp_path)

    assert contract is not None
    assert [item.command for item in contract.commands] == [
        "python -m pytest -q",
        "npm run lint",
    ]
