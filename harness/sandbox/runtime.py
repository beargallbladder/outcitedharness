from __future__ import annotations

import hashlib
import shutil
import socket
import time
import urllib.request
from pathlib import Path

from .backend import DockerCLIBackend, SubprocessCommandRunner
from .events import SandboxEventStore
from .policy import SandboxPolicy
from .preview import TailscalePreviewPublisher
from .registry import JsonSandboxRegistry
from .service import SandboxService


DEFAULT_ROOT = Path("/Volumes/M5_4TB/harness-sandboxes")
DEFAULT_DOCKER_CONTEXT = "colima-harness-sandbox"
_IGNORED_CONTEXT_NAMES = frozenset(
    {
        ".DS_Store",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)


def create_service(
    root: Path = DEFAULT_ROOT,
    *,
    docker_context: str = DEFAULT_DOCKER_CONTEXT,
    max_active_sandboxes: int = 8,
) -> SandboxService:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    runner = SubprocessCommandRunner(
        environment={"DOCKER_CONTEXT": docker_context}
    )
    backend = DockerCLIBackend(
        SandboxPolicy(allowed_mount_roots=(root,)),
        runner,
    )
    return SandboxService(
        backend,
        JsonSandboxRegistry(root / "state" / "sandboxes.json"),
        preview_publisher=TailscalePreviewPublisher(timeout=60),
        max_active_sandboxes=max_active_sandboxes,
        event_store=SandboxEventStore(root / "state" / "sandbox-events.sqlite3"),
    )


def stage_context(source: Path, root: Path, sandbox_id: str) -> Path:
    source = source.expanduser().resolve(strict=True)
    if not source.is_dir():
        raise ValueError("sandbox source context must be a directory")
    if not (source / "Dockerfile").is_file():
        raise ValueError("sandbox source context must contain a Dockerfile")
    root = root.expanduser().resolve()
    workspace_root = root / "workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)
    source_hash = context_hash(source)
    destination = workspace_root / f"{sandbox_id}-{source_hash[:12]}"
    if destination.exists():
        if context_hash(destination) != source_hash:
            raise ValueError("staged sandbox context exists with different content")
        return destination
    temporary = workspace_root / f".{destination.name}.staging"
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        shutil.copytree(
            source,
            temporary,
            symlinks=True,
            ignore=_context_ignore,
        )
        staged_hash = context_hash(temporary)
        if staged_hash != source_hash:
            raise ValueError("staged sandbox context hash mismatch")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def context_hash(context: Path) -> str:
    context = context.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    for path in sorted(context.rglob("*")):
        relative = path.relative_to(context)
        if any(part in _IGNORED_CONTEXT_NAMES for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"symlinked build input is forbidden: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"non-regular build input is forbidden: {relative}")
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def allocate_port(service: SandboxService) -> int:
    reserved = {
        proxy.host_port
        for record in service.registry.list()
        if record.manifest is not None
        for proxy in record.manifest.proxies
    }
    for port in range(20_000, 45_001):
        if port in reserved:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError("no sandbox preview ports are available")


def wait_http(
    url: str,
    *,
    timeout: float = 60,
    expected: str | None = None,
) -> str:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                body = response.read(1024 * 1024).decode(
                    "utf-8", errors="replace"
                )
            if expected is None or expected in body:
                return body
            last_error = RuntimeError("health response lacks expected marker")
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"sandbox health check timed out: {last_error}")


def _context_ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _IGNORED_CONTEXT_NAMES}
