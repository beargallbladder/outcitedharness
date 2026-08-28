"""Deterministic, repository-scoped verification policy.

The contract is compiled from checked-in configuration. Models never author
commands, and every selected command is still checked by orch_loop's allowlist.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT_S = 60
CONFIG_NAMES = (
    ".harness.toml",
    "pyproject.toml",
    "pytest.ini",
    "package.json",
    "pnpm-workspace.yaml",
    "tsconfig.json",
    "ruff.toml",
    ".ruff.toml",
)
_SCRIPT_NAMES = re.compile(r"^(?:test(?::[A-Za-z0-9_-]+)?|lint|typecheck|check)$")
_UNSAFE = re.compile(r"""[;&|`$<>]|\$\(""")


@dataclass(frozen=True)
class VerificationCommand:
    name: str
    argv: tuple[str, ...]
    timeout_s: int = DEFAULT_TIMEOUT_S
    source: str = ""

    @property
    def command(self) -> str:
        return shlex.join(self.argv)


@dataclass
class RepoContract:
    repo_root: str
    languages: list[str] = field(default_factory=list)
    known_test_roots: list[str] = field(default_factory=list)
    commands: list[VerificationCommand] = field(default_factory=list)
    configs: list[str] = field(default_factory=list)
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _safe_timeout(raw: Any) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_S
    return max(1, min(value, 600))


def _safe_argv(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    argv = tuple(str(part).strip() for part in raw if str(part).strip())
    if not argv or any(_UNSAFE.search(part) for part in argv):
        return ()
    head = Path(argv[0]).name.lower()
    if head in {"pytest", "ruff", "eslint", "tsc"}:
        return argv
    if head in {"python", "python3"}:
        return argv if len(argv) >= 3 and argv[1] == "-m" and argv[2] in {"pytest", "ruff"} else ()
    if head == "npx":
        return argv if len(argv) >= 2 and Path(argv[1]).name.lower() == "tsc" else ()
    if head in {"npm", "pnpm"}:
        if len(argv) >= 3 and argv[1] in {"run", "run-script"}:
            return argv if _SCRIPT_NAMES.match(argv[2]) else ()
        if head == "pnpm" and len(argv) >= 2:
            return argv if _SCRIPT_NAMES.match(argv[1]) else ()
    return ()


def _override_commands(path: Path) -> list[VerificationCommand]:
    data = _read_toml(path)
    verification = data.get("verification")
    section = verification if isinstance(verification, dict) else {}
    commands = section.get("commands")
    if not isinstance(commands, dict):
        commands = data.get("commands")
    if not isinstance(commands, dict):
        return []
    required = section.get("required")
    order = (
        [str(name) for name in required if str(name).strip()]
        if isinstance(required, list)
        else list(commands)
    )
    out: list[VerificationCommand] = []
    for name in order:
        spec = commands.get(name)
        if not isinstance(spec, dict):
            continue
        argv = _safe_argv(spec.get("argv"))
        if not argv:
            continue
        out.append(
            VerificationCommand(
                name=name,
                argv=argv,
                timeout_s=_safe_timeout(spec.get("timeout", spec.get("timeout_s"))),
                source=".harness.toml",
            )
        )
    return out


def _python_commands(root: Path, pyproject: dict[str, Any]) -> list[VerificationCommand]:
    pytest_configured = (root / "pytest.ini").is_file()
    tool = pyproject.get("tool") if isinstance(pyproject.get("tool"), dict) else {}
    pytest_configured = pytest_configured or isinstance(tool.get("pytest"), dict)
    project = pyproject.get("project") if isinstance(pyproject.get("project"), dict) else {}
    dependencies = list(project.get("dependencies") or [])
    optional = project.get("optional-dependencies")
    if isinstance(optional, dict):
        for values in optional.values():
            if isinstance(values, list):
                dependencies.extend(values)
    pytest_configured = pytest_configured or any(
        re.match(r"(?i)\s*pytest(?:\W|$)", str(dep)) for dep in dependencies
    )
    out: list[VerificationCommand] = []
    if pytest_configured:
        local = root / ".venv" / "bin" / "pytest"
        argv = (".venv/bin/pytest", "-q") if local.is_file() else ("python", "-m", "pytest", "-q")
        out.append(VerificationCommand("unit", argv, source="pyproject/pytest"))
    ruff_configured = isinstance(tool.get("ruff"), dict) or (root / "ruff.toml").is_file() or (
        root / ".ruff.toml"
    ).is_file()
    if ruff_configured:
        local = root / ".venv" / "bin" / "ruff"
        argv = (".venv/bin/ruff", "check", ".") if local.is_file() else (
            "python",
            "-m",
            "ruff",
            "check",
            ".",
        )
        out.append(VerificationCommand("lint", argv, source="ruff config"))
    return out


def _package_commands(root: Path, package: dict[str, Any]) -> list[VerificationCommand]:
    scripts = package.get("scripts") if isinstance(package.get("scripts"), dict) else {}
    if not scripts:
        return []
    if (root / "pnpm-lock.yaml").is_file() or (root / "pnpm-workspace.yaml").is_file():
        manager = "pnpm"
    elif (root / "package-lock.json").is_file():
        manager = "npm"
    else:
        manager = "npm"
    names = [
        name
        for name in scripts
        if isinstance(name, str) and _SCRIPT_NAMES.match(name)
    ]
    priority = {"test": 0, "lint": 1, "typecheck": 2, "check": 3}
    names.sort(key=lambda name: (priority.get(name, 0 if name.startswith("test:") else 9), name))
    return [
        VerificationCommand(
            name=name,
            argv=(manager, "run", name),
            source="package.json",
        )
        for name in names
    ]


def _workflow_commands(root: Path) -> list[VerificationCommand]:
    workflow_dir = root / ".github" / "workflows"
    if not workflow_dir.is_dir():
        return []
    out: list[VerificationCommand] = []
    for path in sorted((*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml"))):
        try:
            text = path.read_text()
        except OSError:
            continue
        for index, match in enumerate(
            re.finditer(r"(?m)^\s*(?:-\s*)?run:\s*([^\n#]+?)\s*$", text),
            1,
        ):
            raw = match.group(1).strip().strip("\"'")
            try:
                argv = shlex.split(raw)
            except ValueError:
                continue
            safe = _safe_argv(argv)
            if not safe:
                continue
            out.append(
                VerificationCommand(
                    name=f"ci-{path.stem}-{index}",
                    argv=safe,
                    source=str(path.relative_to(root)),
                )
            )
    return out


def _dedupe(commands: list[VerificationCommand]) -> list[VerificationCommand]:
    out: list[VerificationCommand] = []
    seen: set[str] = set()
    for command in commands:
        rendered = command.command
        if rendered in seen:
            continue
        seen.add(rendered)
        out.append(command)
    return out


def build_repo_contract(root: Path | str | None) -> RepoContract | None:
    if root is None:
        return None
    repo = Path(root).expanduser().resolve()
    if not repo.is_dir():
        return None
    configs = [
        name for name in CONFIG_NAMES if (repo / name).is_file()
    ]
    workflow_dir = repo / ".github" / "workflows"
    if workflow_dir.is_dir():
        configs.extend(
            str(path.relative_to(repo))
            for path in sorted((*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")))
        )

    override_path = repo / ".harness.toml"
    override = _override_commands(override_path) if override_path.is_file() else []
    pyproject = _read_toml(repo / "pyproject.toml") if (repo / "pyproject.toml").is_file() else {}
    package = _read_json(repo / "package.json") if (repo / "package.json").is_file() else {}
    languages: list[str] = []
    test_roots: list[str] = []
    if pyproject or (repo / "pytest.ini").is_file():
        languages.append("python")
        test_roots.append("tests/")
    if package:
        languages.append("javascript")

    inferred = [
        *_python_commands(repo, pyproject),
        *_package_commands(repo, package),
        *_workflow_commands(repo),
    ]
    commands = _dedupe(override or inferred)
    digest = hashlib.sha256()
    for rel in configs:
        path = repo / rel
        digest.update(rel.encode())
        try:
            digest.update(path.read_bytes())
        except OSError:
            pass
    return RepoContract(
        repo_root=str(repo),
        languages=languages,
        known_test_roots=test_roots,
        commands=commands,
        configs=configs,
        fingerprint=digest.hexdigest(),
    )
