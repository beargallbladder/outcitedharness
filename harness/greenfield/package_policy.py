from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path


_UNSAFE = re.compile(r"""[;&|`$<>]|\$\(""")
PYTHON_TOOLING = {"pytest", "ruff"}
NODE_TOOLING = {"typescript", "eslint", "@eslint/js", "@types/node", "typescript-eslint"}


class PackagePolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class PackageAction:
    manager: str
    argv: tuple[str, ...]
    cwd: Path
    timeout_s: int = 300


def package_name(spec: str) -> str:
    value = spec.strip()
    if value.startswith("@"):
        slash = value.find("/")
        if slash < 2:
            return value
        match = re.match(r"^(@[^/]+/[A-Za-z0-9._-]+)", value)
    else:
        match = re.match(r"^([A-Za-z0-9._-]+)", value)
    return match.group(1).lower() if match else value.lower()


def _assert_python_manifest(root: Path, approved: set[str]) -> None:
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PackagePolicyError(f"invalid Python package manifest: {exc}") from exc
    project = data.get("project") if isinstance(data.get("project"), dict) else {}
    actual = {package_name(str(item)) for item in project.get("dependencies", [])}
    optional = project.get("optional-dependencies")
    dev = (
        {package_name(str(item)) for item in optional.get("dev", [])}
        if isinstance(optional, dict) and isinstance(optional.get("dev"), list)
        else set()
    )
    if not actual.issubset(approved):
        raise PackagePolicyError(f"unapproved Python dependencies: {sorted(actual - approved)}")
    if not dev.issubset(PYTHON_TOOLING):
        raise PackagePolicyError(f"unapproved Python tooling: {sorted(dev - PYTHON_TOOLING)}")


def _assert_node_manifest(root: Path, approved: set[str]) -> None:
    try:
        data = json.loads((root / "package.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PackagePolicyError(f"invalid Node package manifest: {exc}") from exc
    dependencies = data.get("dependencies") if isinstance(data.get("dependencies"), dict) else {}
    dev = data.get("devDependencies") if isinstance(data.get("devDependencies"), dict) else {}
    actual = {str(name).lower() for name in dependencies}
    tooling = {str(name).lower() for name in dev}
    if not actual.issubset(approved):
        raise PackagePolicyError(f"unapproved Node dependencies: {sorted(actual - approved)}")
    if not tooling.issubset(NODE_TOOLING):
        raise PackagePolicyError(f"unapproved Node tooling: {sorted(tooling - NODE_TOOLING)}")


def validate_package_action(
    action: PackageAction,
    *,
    repo_root: Path,
    approved_dependencies: tuple[str, ...],
) -> None:
    root = repo_root.resolve()
    if action.cwd.resolve() != root:
        raise PackagePolicyError("package action cwd must be the leased repository")
    if not 1 <= action.timeout_s <= 900:
        raise PackagePolicyError("package action timeout is outside policy")
    if not action.argv or any(_UNSAFE.search(part) for part in action.argv):
        raise PackagePolicyError("unsafe package-manager argv")
    approved = {package_name(item) for item in approved_dependencies}
    if action.manager == "uv":
        if action.argv != ("uv", "sync", "--extra", "dev"):
            raise PackagePolicyError("only adapter-owned uv sync is allowed")
        _assert_python_manifest(root, approved)
    elif action.manager == "npm":
        if action.argv != ("npm", "install", "--ignore-scripts"):
            raise PackagePolicyError("only adapter-owned npm install is allowed")
        _assert_node_manifest(root, approved)
    else:
        raise PackagePolicyError(f"unsupported package manager: {action.manager}")
    if shutil.which(action.argv[0]) is None:
        raise PackagePolicyError(f"required package manager is unavailable: {action.argv[0]}")


def execute_package_action(
    action: PackageAction,
    *,
    repo_root: Path,
    approved_dependencies: tuple[str, ...],
) -> subprocess.CompletedProcess[str]:
    validate_package_action(
        action,
        repo_root=repo_root,
        approved_dependencies=approved_dependencies,
    )
    allowed_env = {
        name: value
        for name, value in os.environ.items()
        if name in {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SSL_CERT_FILE"}
    }
    if action.manager == "uv":
        allowed_env["UV_DEFAULT_INDEX"] = "https://pypi.org/simple"
    elif action.manager == "npm":
        allowed_env["NPM_CONFIG_REGISTRY"] = "https://registry.npmjs.org"
        allowed_env["NPM_CONFIG_USERCONFIG"] = os.devnull
    proc = subprocess.run(
        list(action.argv),
        cwd=action.cwd,
        env=allowed_env,
        capture_output=True,
        text=True,
        timeout=action.timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        raise PackagePolicyError(
            f"{action.manager} setup failed ({proc.returncode}): "
            f"{(proc.stderr or proc.stdout)[-1200:]}"
        )
    return proc
