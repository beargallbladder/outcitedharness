from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .models import BuildSpec, EgressMode, SandboxSpec


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENSITIVE_ENV = re.compile(
    r"(?:^|_)(?:SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIALS?|PRIVATE_KEY|API_KEY)(?:_|$)",
    re.IGNORECASE,
)
_SENSITIVE_BUILD_ARG = re.compile(
    r"(?:^|_)(?:SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIALS?|PRIVATE_KEY|API_KEY|AUTH)(?:_|$)",
    re.IGNORECASE,
)
_FORBIDDEN_TARGETS = (
    "/dev",
    "/etc",
    "/proc",
    "/run",
    "/sys",
    "/var/run",
)
_SENSITIVE_CONTEXT_NAME = re.compile(
    r"^(?:\.env(?:\..*)?|id_(?:rsa|dsa|ecdsa|ed25519)|credentials\.json|"
    r"\.npmrc|\.pypirc)$",
    re.IGNORECASE,
)
_REMOTE_ADD = re.compile(
    r"(?im)^\s*ADD\b[^\n]*(?:(?:https?|git|ssh)://|git@)"
)
_SECRET_MOUNT = re.compile(
    r"(?im)^\s*RUN\s+--mount=type=(?:secret|ssh)(?:,|\s)"
)
_FOREIGN_PLATFORM = re.compile(
    r"(?im)^\s*FROM\s+--platform=(?!linux/arm64(?:\s|$))\S+"
)


class PolicyViolation(ValueError):
    """Raised before an unsafe sandbox operation can reach the backend."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class SandboxPolicy:
    allowed_mount_roots: tuple[Path, ...]
    minimum_host_port: int = 20_000
    maximum_host_port: int = 45_000
    maximum_ports: int = 8
    egress_networks: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    allow_build_network: bool = False
    allow_runtime_unrestricted_egress: bool = False
    maximum_build_context_bytes: int = 2 * 1024**3

    def __post_init__(self) -> None:
        roots = tuple(Path(root).resolve() for root in self.allowed_mount_roots)
        object.__setattr__(self, "allowed_mount_roots", roots)
        networks = {
            str(name): tuple(str(host) for host in hosts)
            for name, hosts in self.egress_networks.items()
        }
        object.__setattr__(self, "egress_networks", MappingProxyType(networks))
        if not roots:
            raise ValueError("at least one allowed mount root is required")
        if not 1024 <= self.minimum_host_port <= self.maximum_host_port <= 65535:
            raise ValueError("host port range is invalid")
        if not 0 <= self.maximum_ports <= 64:
            raise ValueError("maximum_ports must be between 0 and 64")
        if not 1024 <= self.maximum_build_context_bytes <= 16 * 1024**3:
            raise ValueError("maximum build context size is invalid")
        for name, hosts in networks.items():
            if not name or not hosts:
                raise ValueError("egress network mappings may not be empty")

    def validate(self, spec: SandboxSpec) -> None:
        if spec.platform != "linux/arm64":
            raise PolicyViolation("only linux/arm64 sandboxes are permitted")
        if spec.user in {"", "0", "0:0", "root"}:
            raise PolicyViolation("root containers are forbidden")

        self._validate_environment(spec)
        self._validate_mounts(spec)
        self._validate_ports(spec)
        self._validate_egress(spec)

    def validate_build(self, spec: BuildSpec) -> None:
        context = spec.context.resolve()
        if not context.is_dir():
            raise PolicyViolation("build context must be an existing directory")
        if not any(_is_within(context, root) for root in self.allowed_mount_roots):
            raise PolicyViolation("build context is outside allowed roots")
        dockerfile = (context / spec.dockerfile).resolve()
        if not dockerfile.is_file() or not _is_within(dockerfile, context):
            raise PolicyViolation("dockerfile must be a file inside the build context")
        self._validate_build_context(context)
        try:
            dockerfile_text = dockerfile.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise PolicyViolation("Dockerfile must be readable UTF-8 text") from exc
        if _REMOTE_ADD.search(dockerfile_text):
            raise PolicyViolation("remote ADD sources are forbidden")
        if _SECRET_MOUNT.search(dockerfile_text):
            raise PolicyViolation("Dockerfile secret and SSH mounts are forbidden")
        if _FOREIGN_PLATFORM.search(dockerfile_text):
            raise PolicyViolation("Dockerfile may not override the ARM64 platform")
        if spec.allow_network and not self.allow_build_network:
            raise PolicyViolation("networked builds are disabled by policy")
        for key, value in spec.build_args.items():
            if (
                not _ENV_NAME.fullmatch(key)
                or _SENSITIVE_BUILD_ARG.search(key)
                or "\x00" in value
            ):
                raise PolicyViolation("unsafe or secret-bearing build argument")

    def _validate_build_context(self, context: Path) -> None:
        total = 0
        for directory, dirnames, filenames in os.walk(context, followlinks=False):
            base = Path(directory)
            for name in tuple(dirnames):
                child = base / name
                if child.is_symlink():
                    raise PolicyViolation(
                        f"symlinked build directory is forbidden: "
                        f"{child.relative_to(context)}"
                    )
            for name in filenames:
                child = base / name
                relative = child.relative_to(context)
                mode = child.lstat().st_mode
                if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                    raise PolicyViolation(
                        f"non-regular build input is forbidden: {relative}"
                    )
                if _SENSITIVE_CONTEXT_NAME.fullmatch(name):
                    raise PolicyViolation(
                        f"secret-bearing build input is forbidden: {relative}"
                    )
                total += child.stat().st_size
                if total > self.maximum_build_context_bytes:
                    raise PolicyViolation("build context exceeds the configured size cap")

    def _validate_environment(self, spec: SandboxSpec) -> None:
        for key in spec.environment:
            if not _ENV_NAME.fullmatch(key):
                raise PolicyViolation(f"invalid environment name: {key!r}")
            if _SENSITIVE_ENV.search(key):
                raise PolicyViolation(f"secret-bearing environment is forbidden: {key}")

    def _validate_mounts(self, spec: SandboxSpec) -> None:
        targets: set[str] = set()
        for mount in spec.mounts:
            try:
                source = mount.source.resolve(strict=True)
            except OSError as exc:
                raise PolicyViolation(f"mount source is unavailable: {mount.source}") from exc
            if "," in str(source) or "," in mount.target:
                raise PolicyViolation("mount paths may not contain commas")
            if not any(_is_within(source, root) for root in self.allowed_mount_roots):
                raise PolicyViolation(f"mount source is outside allowed roots: {source}")
            lowered = os.fspath(source).lower()
            if lowered.endswith("/docker.sock") or lowered.endswith("/podman.sock"):
                raise PolicyViolation("container-engine sockets may not be mounted")
            target = mount.target.rstrip("/") or "/"
            if any(
                target == forbidden or target.startswith(f"{forbidden}/")
                for forbidden in _FORBIDDEN_TARGETS
            ):
                raise PolicyViolation(f"sensitive mount target is forbidden: {target}")
            if target in targets:
                raise PolicyViolation(f"duplicate mount target: {target}")
            targets.add(target)

    def _validate_ports(self, spec: SandboxSpec) -> None:
        if len(spec.ports) > self.maximum_ports:
            raise PolicyViolation("too many published ports")
        seen: set[tuple[str, int, str]] = set()
        for port in spec.ports:
            if not self.minimum_host_port <= port.host_port <= self.maximum_host_port:
                raise PolicyViolation(
                    f"host port {port.host_port} is outside the allowed range"
                )
            binding = (port.host_ip, port.host_port, port.protocol)
            if binding in seen:
                raise PolicyViolation("duplicate host port binding")
            seen.add(binding)

    def _validate_egress(self, spec: SandboxSpec) -> None:
        if (
            spec.egress.mode is EgressMode.ALLOW_ALL
            and not self.allow_runtime_unrestricted_egress
        ):
            raise PolicyViolation("unrestricted runtime egress is disabled")
        if spec.egress.mode is EgressMode.ALLOWLIST and not spec.egress.policy_network:
            raise PolicyViolation("allowlisted egress requires a policy-enforcing network")
        if spec.egress.mode is EgressMode.ALLOWLIST:
            configured = self.egress_networks.get(spec.egress.policy_network or "")
            if configured is None:
                raise PolicyViolation("egress policy network is not configured")
            if set(configured) != set(spec.egress.allowed_hosts):
                raise PolicyViolation(
                    "requested hosts do not match the configured egress network"
                )
