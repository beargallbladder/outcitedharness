#!/usr/bin/env python3
"""Operate policy-constrained M5 preview sandboxes through Docker CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from harness.sandbox import (
    BuildSpec,
    DockerCLIBackend,
    EgressPolicy,
    JsonSandboxRegistry,
    PortBinding,
    ResourceLimits,
    SandboxPolicy,
    SandboxService,
    SandboxSpec,
)


DEFAULT_ROOT = Path("/Volumes/M5_4TB/harness-sandboxes")
DEFAULT_CONTEXT = "colima-harness-sandbox"


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _port_available(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _allocate_port() -> int:
    for port in range(20_000, 45_001):
        if _port_available(port):
            return port
    raise RuntimeError("no preview port is available")


def _service(root: Path) -> SandboxService:
    root = root.expanduser().resolve()
    workspaces = root / "workspaces"
    workspaces.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DOCKER_CONTEXT", DEFAULT_CONTEXT)
    policy = SandboxPolicy(allowed_mount_roots=(workspaces,))
    backend = DockerCLIBackend(policy, timeout=60)
    return SandboxService(
        backend,
        JsonSandboxRegistry(root / "state" / "sandboxes.json"),
    )


def _json_status(status: object) -> None:
    values = vars(status).copy()
    for key, value in values.items():
        if hasattr(value, "value"):
            values[key] = value.value
        elif hasattr(value, "isoformat"):
            values[key] = value.isoformat()
    print(json.dumps(values, sort_keys=True))


def _write_canary(context: Path) -> None:
    context.mkdir(parents=True, exist_ok=False)
    (context / "Dockerfile").write_text(
        "FROM python:3.12-alpine@sha256:"
        "d09d15e60962ca365d1cd544a48773bac9d33f2fb1b00f2aa0deec78ade7dc31\n"
        "COPY index.html /srv/index.html\n"
        "WORKDIR /srv\n"
        "USER 65532:65532\n"
        'CMD ["python", "-m", "http.server", "8080", "--bind", "0.0.0.0"]\n'
    )
    (context / "index.html").write_text("HARNESS_SANDBOX_CANARY_OK\n")


def _canary(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve()
    sandbox_id = args.sandbox_id
    context = root / "workspaces" / sandbox_id
    if context.exists():
        raise RuntimeError(f"canary workspace already exists: {context}")
    _write_canary(context)
    state_hash = _tree_hash(context)
    image = f"harness-preview/{sandbox_id}:{state_hash[:12]}"
    service = _service(root)
    backend = service.backend
    try:
        backend.build(
            BuildSpec(
                image=image,
                context=context,
                state_hash=state_hash,
            )
        )
        port = args.port or _allocate_port()
        status = service.create(
            SandboxSpec(
                sandbox_id=sandbox_id,
                image=image,
                state_hash=state_hash,
                ports=(PortBinding(container_port=8080, host_port=port),),
                limits=ResourceLimits(
                    cpus=args.cpus,
                    memory_bytes=args.memory_mib * 1024 * 1024,
                    pids=args.pids,
                ),
                egress=EgressPolicy(),
                ttl_seconds=args.ttl,
            )
        )
        response = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--retry",
                "20",
                "--retry-all-errors",
                "--retry-delay",
                "1",
                f"http://127.0.0.1:{port}/",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        if response.returncode != 0 or "HARNESS_SANDBOX_CANARY_OK" not in response.stdout:
            record = service.registry.require(sandbox_id)
            proxy_logs = ""
            if record.manifest and record.manifest.proxies:
                logs = subprocess.run(
                    ["docker", "logs", record.manifest.proxies[0].container_id],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=10,
                )
                proxy_logs = (logs.stderr or logs.stdout).strip()[-1200:]
            service.destroy(sandbox_id)
            raise RuntimeError(
                "sandbox health check failed: "
                f"{response.stderr.strip()} proxy_logs={proxy_logs}"
            )
        payload = vars(status).copy()
        payload.update(
            {
                "state": status.state.value,
                "created_at": status.created_at.isoformat(),
                "expires_at": status.expires_at.isoformat(),
                "updated_at": status.updated_at.isoformat(),
                "local_url": f"http://127.0.0.1:{port}/",
            }
        )
        print(json.dumps(payload, sort_keys=True))
        return 0
    except Exception:
        if service.registry.get(sandbox_id) is not None:
            try:
                service.destroy(sandbox_id)
            except Exception:
                pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    canary = subparsers.add_parser("canary")
    canary.add_argument("--sandbox-id", default="canary")
    canary.add_argument("--port", type=int)
    canary.add_argument("--cpus", type=float, default=0.5)
    canary.add_argument("--memory-mib", type=int, default=128)
    canary.add_argument("--pids", type=int, default=64)
    canary.add_argument("--ttl", type=int, default=900)

    for name in ("status", "destroy"):
        action = subparsers.add_parser(name)
        action.add_argument("sandbox_id")
    subparsers.add_parser("list")
    subparsers.add_parser("recover")
    subparsers.add_parser("reap")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "canary":
            return _canary(args)
        service = _service(args.root)
        if args.command == "status":
            _json_status(service.status(args.sandbox_id, refresh=True))
        elif args.command == "destroy":
            _json_status(service.destroy(args.sandbox_id))
        elif args.command == "list":
            for status in service.list():
                _json_status(status)
        elif args.command == "recover":
            for status in service.recover():
                _json_status(status)
        elif args.command == "reap":
            for status in service.reap_expired():
                _json_status(status)
        return 0
    except Exception as error:
        print(f"sandboxctl: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
