from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence
from uuid import uuid4

from .models import (
    BuildSpec,
    EgressMode,
    PortBinding,
    SandboxManifest,
    SandboxProxyManifest,
    SandboxSpec,
)
from .policy import SandboxPolicy


MANAGED_LABEL = "io.harness.sandbox.managed"
BUILD_LABEL = "io.harness.sandbox.build"
MANIFEST_LABEL = "io.harness.sandbox.manifest"
SANDBOX_LABEL = "io.harness.sandbox.id"
STATE_HASH_LABEL = "io.harness.sandbox.state-hash"
SPEC_HASH_LABEL = "io.harness.sandbox.spec-hash"
DENY_NETWORK = "harness-sandbox-internal"
PROXY_LABEL = "io.harness.sandbox.proxy"
PROXY_IMAGE = (
    "caddy@sha256:4c6e91c6ed0e2fa03efd5b44747b625f"
    "ec79bc9cd06ac5235a779726618e530d"
)
_CONTAINER_ID = re.compile(r"^[a-fA-F0-9]{12,64}$")


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    def run(self, argv: Sequence[str], *, timeout: float) -> CommandResult:
        """Execute a literal argv without involving a shell."""


class SubprocessCommandRunner:
    def __init__(self, *, environment: dict[str, str] | None = None) -> None:
        self.environment = dict(environment or {})

    def run(self, argv: Sequence[str], *, timeout: float) -> CommandResult:
        environment = None
        if self.environment:
            import os

            environment = {**os.environ, **self.environment}
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            check=False,
            env=environment,
            shell=False,
            text=True,
            timeout=timeout,
        )
        return CommandResult(proc.returncode, proc.stdout, proc.stderr)


class BackendError(RuntimeError):
    pass


class OwnershipError(BackendError):
    pass


@dataclass(frozen=True)
class BackendContainer:
    container_id: str
    name: str
    image: str
    sandbox_id: str
    manifest_id: str
    state_hash: str
    spec_hash: str
    running: bool
    status: str
    is_proxy: bool = False


class DockerCLIBackend:
    """Small Docker CLI adapter that emits only policy-owned literal argv."""

    def __init__(
        self,
        policy: SandboxPolicy,
        runner: CommandRunner | None = None,
        *,
        executable: str = "docker",
        timeout: float = 30.0,
    ) -> None:
        if not executable or "/" in executable or "\x00" in executable:
            raise ValueError("Docker executable must be a bare command name")
        if not 1 <= timeout <= 300:
            raise ValueError("backend timeout must be between 1 and 300 seconds")
        self.policy = policy
        self.runner = runner or SubprocessCommandRunner()
        self.executable = executable
        self.timeout = timeout

    def build(self, spec: BuildSpec) -> str:
        self.policy.validate_build(spec)
        context = spec.context.resolve()
        dockerfile = (context / spec.dockerfile).resolve()
        argv = [
            self.executable,
            "build",
            "--platform",
            spec.platform,
            "--network",
            "default" if spec.allow_network else "none",
            "--pull=false",
            "--label",
            f"{BUILD_LABEL}=true",
            "--label",
            f"{STATE_HASH_LABEL}={spec.state_hash.lower()}",
            "--file",
            str(dockerfile),
            "--tag",
            spec.image,
        ]
        for key, value in sorted(spec.build_args.items()):
            argv.extend(("--build-arg", f"{key}={value}"))
        argv.append(str(context))
        self._execute(argv, timeout=max(self.timeout, 300.0))
        return spec.image

    def create(self, spec: SandboxSpec) -> SandboxManifest:
        self.policy.validate(spec)
        if spec.egress.mode is EgressMode.DENY:
            self._ensure_deny_network()
        manifest_id = uuid4().hex
        name = f"harness-{spec.sandbox_id}-{manifest_id[:12]}"
        labels = {
            MANAGED_LABEL: "true",
            MANIFEST_LABEL: manifest_id,
            SANDBOX_LABEL: spec.sandbox_id,
            STATE_HASH_LABEL: spec.state_hash.lower(),
            SPEC_HASH_LABEL: spec.spec_hash,
        }
        argv = [
            self.executable,
            "create",
            "--name",
            name,
            "--platform",
            spec.platform,
        ]
        for key, value in labels.items():
            argv.extend(("--label", f"{key}={value}"))
        argv.extend(
            (
                "--user",
                spec.user,
                "--cpus",
                f"{spec.limits.cpus:g}",
                "--memory",
                str(spec.limits.memory_bytes),
                "--pids-limit",
                str(spec.limits.pids),
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges:true",
                "--read-only",
                "--init",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=64m",
                "--tmpfs",
                "/run:rw,noexec,nosuid,nodev,size=16m",
                "--network",
                self._network_for(spec),
            )
        )
        for mount in spec.mounts:
            option = f"type=bind,src={mount.source.resolve()},dst={mount.target}"
            if mount.read_only:
                option += ",readonly"
            argv.extend(("--mount", option))
        if spec.egress.mode is not EgressMode.DENY:
            for port in spec.ports:
                argv.extend(
                    (
                        "--publish",
                        f"{port.host_ip}:{port.host_port}:"
                        f"{port.container_port}/{port.protocol}",
                    )
                )
        for key, value in sorted(spec.environment.items()):
            argv.extend(("--env", f"{key}={value}"))
        argv.append(spec.image)
        argv.extend(spec.command)

        result = self._execute(argv)
        container_id = result.stdout.strip()
        if not _CONTAINER_ID.fullmatch(container_id):
            raise BackendError("Docker create returned an invalid container id")
        proxies: list[SandboxProxyManifest] = []
        if spec.egress.mode is EgressMode.DENY:
            try:
                for port in spec.ports:
                    proxies.append(
                        self._create_proxy(spec, manifest_id, name, labels, port)
                    )
            except Exception:
                for proxy in proxies:
                    self.runner.run(
                        (self.executable, "rm", "--force", proxy.container_id),
                        timeout=self.timeout,
                    )
                self.runner.run(
                    (self.executable, "rm", "--force", container_id),
                    timeout=self.timeout,
                )
                raise
        return SandboxManifest(
            manifest_id=manifest_id,
            sandbox_id=spec.sandbox_id,
            container_id=container_id.lower(),
            container_name=name,
            state_hash=spec.state_hash.lower(),
            spec_hash=spec.spec_hash,
            image=spec.image,
            proxies=tuple(proxies),
        )

    def start(self, manifest: SandboxManifest) -> BackendContainer:
        self.inspect_owned(manifest)
        self._execute((self.executable, "start", manifest.container_id))
        for proxy in manifest.proxies:
            self._inspect_proxy(manifest, proxy)
            self._execute((self.executable, "start", proxy.container_id))
        return self.inspect_owned(manifest)

    def stop(
        self, manifest: SandboxManifest, *, grace_seconds: int = 10
    ) -> BackendContainer:
        if not 0 <= grace_seconds <= 60:
            raise ValueError("stop grace must be between 0 and 60 seconds")
        self.inspect_owned(manifest)
        for proxy in manifest.proxies:
            self._inspect_proxy(manifest, proxy)
            self._execute(
                (
                    self.executable,
                    "stop",
                    "--time",
                    str(grace_seconds),
                    proxy.container_id,
                )
            )
        self._execute(
            (
                self.executable,
                "stop",
                "--time",
                str(grace_seconds),
                manifest.container_id,
            )
        )
        return self.inspect_owned(manifest)

    def remove_owned(self, manifest: SandboxManifest) -> None:
        self.inspect_owned(manifest)
        for proxy in manifest.proxies:
            self._inspect_proxy(manifest, proxy)
        for proxy in manifest.proxies:
            self._execute((self.executable, "rm", "--force", proxy.container_id))
        self._execute((self.executable, "rm", "--force", manifest.container_id))

    def inspect_owned(self, manifest: SandboxManifest) -> BackendContainer:
        container = self.inspect(manifest.container_id)
        expected = (
            container.manifest_id == manifest.manifest_id
            and container.sandbox_id == manifest.sandbox_id
            and container.state_hash == manifest.state_hash.lower()
            and container.spec_hash == manifest.spec_hash
            and container.container_id.startswith(manifest.container_id)
        )
        if not expected:
            raise OwnershipError(
                f"container {manifest.container_id} is not owned by its manifest"
            )
        return container

    def logs_owned(self, manifest: SandboxManifest, *, tail: int = 200) -> str:
        if not 1 <= tail <= 10_000:
            raise ValueError("log tail must be between 1 and 10000")
        self.inspect_owned(manifest)
        result = self._execute(
            (
                self.executable,
                "logs",
                "--tail",
                str(tail),
                manifest.container_id,
            )
        )
        return result.stdout + result.stderr

    def inspect(self, container_id: str) -> BackendContainer:
        if not _CONTAINER_ID.fullmatch(container_id):
            raise BackendError("refusing to inspect an invalid container id")
        result = self._execute(
            (self.executable, "inspect", "--type", "container", container_id)
        )
        try:
            rows = json.loads(result.stdout)
            if not isinstance(rows, list) or len(rows) != 1:
                raise ValueError("expected exactly one container")
            return self._parse_container(rows[0])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise BackendError("Docker inspect returned malformed data") from exc

    def discover_managed(self) -> tuple[BackendContainer, ...]:
        result = self._execute(
            (
                self.executable,
                "ps",
                "--all",
                "--quiet",
                "--filter",
                f"label={MANAGED_LABEL}=true",
            )
        )
        ids = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
        if not ids:
            return ()
        if any(not _CONTAINER_ID.fullmatch(container_id) for container_id in ids):
            raise BackendError("Docker listed an invalid managed container id")
        inspect_result = self._execute(
            (self.executable, "inspect", "--type", "container", *ids)
        )
        try:
            rows = json.loads(inspect_result.stdout)
            if not isinstance(rows, list):
                raise ValueError("inspect result is not a list")
            return tuple(
                container
                for row in rows
                if not (container := self._parse_container(row)).is_proxy
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise BackendError("Docker inspect returned malformed managed data") from exc

    def _network_for(self, spec: SandboxSpec) -> str:
        if spec.egress.mode is EgressMode.DENY:
            return DENY_NETWORK
        if spec.egress.mode is EgressMode.ALLOWLIST:
            assert spec.egress.policy_network is not None
            return spec.egress.policy_network
        return "bridge"

    def _create_proxy(
        self,
        spec: SandboxSpec,
        manifest_id: str,
        app_name: str,
        labels: dict[str, str],
        port: PortBinding,
    ) -> SandboxProxyManifest:
        if port.protocol != "tcp":
            raise BackendError("deny-egress preview proxies currently require TCP")
        proxy_name = f"{app_name}-p{port.container_port}"
        proxy_labels = {**labels, PROXY_LABEL: "true"}
        argv = [
            self.executable,
            "create",
            "--name",
            proxy_name,
            "--network",
            "bridge",
            "--user",
            "65532:65532",
            "--cpus",
            "0.25",
            "--memory",
            str(64 * 1024 * 1024),
            "--pids-limit",
            "64",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "NET_BIND_SERVICE",
            "--security-opt",
            "no-new-privileges:true",
            "--read-only",
            "--tmpfs",
            "/data:rw,noexec,nosuid,nodev,size=8m,uid=65532,gid=65532",
            "--tmpfs",
            "/config:rw,noexec,nosuid,nodev,size=8m,uid=65532,gid=65532",
            "--publish",
            f"{port.host_ip}:{port.host_port}:{port.container_port}/tcp",
        ]
        for key, value in proxy_labels.items():
            argv.extend(("--label", f"{key}={value}"))
        argv.extend(
            (
                PROXY_IMAGE,
                "caddy",
                "reverse-proxy",
                "--from",
                f"http://:{port.container_port}",
                "--to",
                f"http://{app_name}:{port.container_port}",
            )
        )
        result = self._execute(argv)
        container_id = result.stdout.strip().lower()
        if not _CONTAINER_ID.fullmatch(container_id):
            raise BackendError("Docker proxy create returned an invalid container id")
        try:
            self._execute(
                (self.executable, "network", "connect", DENY_NETWORK, container_id)
            )
        except Exception:
            self.runner.run(
                (self.executable, "rm", "--force", container_id),
                timeout=self.timeout,
            )
            raise
        return SandboxProxyManifest(
            container_id=container_id,
            container_name=proxy_name,
            host_port=port.host_port,
            container_port=port.container_port,
            protocol=port.protocol,
        )

    def _inspect_proxy(
        self,
        manifest: SandboxManifest,
        proxy: SandboxProxyManifest,
    ) -> BackendContainer:
        container = self.inspect(proxy.container_id)
        if (
            container.manifest_id != manifest.manifest_id
            or container.sandbox_id != manifest.sandbox_id
            or container.state_hash != manifest.state_hash
            or container.spec_hash != manifest.spec_hash
            or not container.is_proxy
        ):
            raise OwnershipError(
                f"proxy {proxy.container_id} is not owned by its manifest"
            )
        return container

    def _ensure_deny_network(self) -> None:
        inspect_argv = (
            self.executable,
            "network",
            "inspect",
            DENY_NETWORK,
        )
        result = self.runner.run(inspect_argv, timeout=self.timeout)
        if result.returncode != 0:
            created = self.runner.run(
                (
                    self.executable,
                    "network",
                    "create",
                    "--driver",
                    "bridge",
                    "--internal",
                    "--label",
                    f"{MANAGED_LABEL}=true",
                    DENY_NETWORK,
                ),
                timeout=self.timeout,
            )
            if created.returncode != 0:
                # A concurrent creator may have won; ownership is still verified.
                retry = self.runner.run(inspect_argv, timeout=self.timeout)
                if retry.returncode != 0:
                    raise BackendError("could not create the internal sandbox network")
                result = retry
            else:
                result = self.runner.run(inspect_argv, timeout=self.timeout)
        try:
            rows = json.loads(result.stdout)
            network = rows[0]
            labels = network.get("Labels") or {}
            if (
                not network.get("Internal")
                or network.get("Driver") != "bridge"
                or labels.get(MANAGED_LABEL) != "true"
            ):
                raise ValueError("network ownership or isolation mismatch")
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OwnershipError(
                f"network {DENY_NETWORK} is not an owned internal bridge"
            ) from exc

    def _execute(
        self, argv: Sequence[str], *, timeout: float | None = None
    ) -> CommandResult:
        if not argv or any(not isinstance(part, str) or "\x00" in part for part in argv):
            raise BackendError("invalid command argv")
        result = self.runner.run(tuple(argv), timeout=timeout or self.timeout)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-1200:]
            raise BackendError(
                f"Docker command {argv[1] if len(argv) > 1 else 'unknown'} "
                f"failed ({result.returncode}): {detail}"
            )
        return result

    @staticmethod
    def _parse_container(raw: object) -> BackendContainer:
        if not isinstance(raw, dict):
            raise ValueError("container is not an object")
        config = raw["Config"]
        state = raw["State"]
        if not isinstance(config, dict) or not isinstance(state, dict):
            raise ValueError("container fields are malformed")
        labels = config.get("Labels") or {}
        if not isinstance(labels, dict) or labels.get(MANAGED_LABEL) != "true":
            raise OwnershipError("container is not managed by the sandbox service")
        container_id = str(raw["Id"]).lower()
        if not _CONTAINER_ID.fullmatch(container_id):
            raise ValueError("invalid container id")
        return BackendContainer(
            container_id=container_id,
            name=str(raw.get("Name") or "").removeprefix("/"),
            image=str(config.get("Image") or ""),
            sandbox_id=str(labels[SANDBOX_LABEL]),
            manifest_id=str(labels[MANIFEST_LABEL]),
            state_hash=str(labels[STATE_HASH_LABEL]).lower(),
            spec_hash=str(labels[SPEC_HASH_LABEL]),
            running=bool(state.get("Running", False)),
            status=str(state.get("Status") or "unknown"),
            is_proxy=labels.get(PROXY_LABEL) == "true",
        )
