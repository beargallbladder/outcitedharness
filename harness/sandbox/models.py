from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}$")
_IMAGE = re.compile(
    r"^(?:[a-zA-Z0-9.-]+(?::[0-9]+)?/)?"
    r"[a-z0-9]+(?:[._/-][a-z0-9]+)*"
    r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}|@sha256:[a-fA-F0-9]{64})?$"
)
_STATE_HASH = re.compile(r"^[a-fA-F0-9]{64}$")
_DNS_HOST = re.compile(
    r"^(?=.{1,253}\.?$)"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.?$"
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_time(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("persisted timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


class SandboxState(StrEnum):
    CREATING = "creating"
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    EXPIRED = "expired"
    MISSING = "missing"
    ERROR = "error"
    REMOVED = "removed"


class EgressMode(StrEnum):
    DENY = "deny"
    ALLOWLIST = "allowlist"
    ALLOW_ALL = "allow_all"


@dataclass(frozen=True)
class ResourceLimits:
    cpus: float = 1.0
    memory_bytes: int = 512 * 1024 * 1024
    pids: int = 128

    def __post_init__(self) -> None:
        if not 0.1 <= self.cpus <= 8.0:
            raise ValueError("cpus must be between 0.1 and 8")
        if not 64 * 1024 * 1024 <= self.memory_bytes <= 16 * 1024**3:
            raise ValueError("memory must be between 64 MiB and 16 GiB")
        if not 16 <= self.pids <= 4096:
            raise ValueError("pids must be between 16 and 4096")


@dataclass(frozen=True)
class PortBinding:
    container_port: int
    host_port: int
    protocol: str = "tcp"
    host_ip: str = "127.0.0.1"

    def __post_init__(self) -> None:
        if not 1 <= self.container_port <= 65535:
            raise ValueError("container port is invalid")
        if not 1 <= self.host_port <= 65535:
            raise ValueError("host port is invalid")
        if self.protocol not in {"tcp", "udp"}:
            raise ValueError("port protocol must be tcp or udp")
        if self.host_ip not in {"127.0.0.1", "::1"}:
            raise ValueError("ports may only bind to loopback")


@dataclass(frozen=True)
class Mount:
    source: Path
    target: str = "/workspace"
    read_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source))
        if not self.source.is_absolute():
            raise ValueError("mount source must be absolute")
        if not self.target.startswith("/") or ".." in Path(self.target).parts:
            raise ValueError("mount target must be an absolute normalized path")


@dataclass(frozen=True)
class EgressPolicy:
    mode: EgressMode = EgressMode.DENY
    allowed_hosts: tuple[str, ...] = ()
    policy_network: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", EgressMode(self.mode))
        object.__setattr__(
            self, "allowed_hosts", tuple(str(host) for host in self.allowed_hosts)
        )
        if self.mode is EgressMode.DENY:
            if self.allowed_hosts or self.policy_network:
                raise ValueError("deny egress cannot specify hosts or a network")
        elif self.mode is EgressMode.ALLOWLIST:
            if not self.allowed_hosts:
                raise ValueError("allowlist egress requires at least one host")
            if not self.policy_network or not _IDENTIFIER.fullmatch(self.policy_network):
                raise ValueError("allowlist egress requires a valid policy network")
            for host in self.allowed_hosts:
                try:
                    ipaddress.ip_address(host)
                    valid = True
                except ValueError:
                    valid = bool(_DNS_HOST.fullmatch(host))
                if not valid:
                    raise ValueError(f"invalid egress host: {host!r}")
        elif self.allowed_hosts or self.policy_network:
            raise ValueError("allow-all egress cannot specify an allowlist")


@dataclass(frozen=True)
class SandboxSpec:
    sandbox_id: str
    image: str
    state_hash: str
    command: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    mounts: tuple[Mount, ...] = ()
    ports: tuple[PortBinding, ...] = ()
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    egress: EgressPolicy = field(default_factory=EgressPolicy)
    ttl_seconds: int = 3600
    platform: str = "linux/arm64"
    user: str = "65532:65532"

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.sandbox_id):
            raise ValueError("sandbox_id must be a safe identifier")
        if not _IMAGE.fullmatch(self.image):
            raise ValueError("image reference is invalid")
        if not _STATE_HASH.fullmatch(self.state_hash):
            raise ValueError("state_hash must be a SHA-256 hex digest")
        if self.platform != "linux/arm64":
            raise ValueError("sandbox platform must be linux/arm64")
        if self.user in {"", "0", "0:0", "root"}:
            raise ValueError("sandbox user must be non-root")
        if not 60 <= self.ttl_seconds <= 7 * 24 * 3600:
            raise ValueError("ttl must be between 60 seconds and 7 days")
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )
        object.__setattr__(self, "mounts", tuple(self.mounts))
        object.__setattr__(self, "ports", tuple(self.ports))
        if any(not isinstance(item, str) or "\x00" in item for item in self.command):
            raise ValueError("command arguments must be NUL-free strings")
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or "\x00" in key
            or "\x00" in value
            or "=" in key
            for key, value in self.environment.items()
        ):
            raise ValueError("environment entries must be valid strings")

    @property
    def spec_hash(self) -> str:
        payload = {
            "sandbox_id": self.sandbox_id,
            "image": self.image,
            "state_hash": self.state_hash.lower(),
            "command": self.command,
            "environment": sorted(self.environment.items()),
            "mounts": [
                (str(m.source), m.target, m.read_only) for m in self.mounts
            ],
            "ports": [asdict(port) for port in self.ports],
            "limits": asdict(self.limits),
            "egress": {
                "mode": self.egress.mode.value,
                "allowed_hosts": self.egress.allowed_hosts,
                "policy_network": self.egress.policy_network,
            },
            "ttl_seconds": self.ttl_seconds,
            "platform": self.platform,
            "user": self.user,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BuildSpec:
    image: str
    context: Path
    state_hash: str
    dockerfile: Path = Path("Dockerfile")
    platform: str = "linux/arm64"
    build_args: Mapping[str, str] = field(default_factory=dict)
    allow_network: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", Path(self.context))
        object.__setattr__(self, "dockerfile", Path(self.dockerfile))
        args = dict(self.build_args)
        object.__setattr__(self, "build_args", MappingProxyType(args))
        if not self.context.is_absolute():
            raise ValueError("build context must be absolute")
        if self.dockerfile.is_absolute() or ".." in self.dockerfile.parts:
            raise ValueError("dockerfile must be relative to the build context")
        if not _IMAGE.fullmatch(self.image):
            raise ValueError("image reference is invalid")
        if not _STATE_HASH.fullmatch(self.state_hash):
            raise ValueError("state_hash must be a SHA-256 hex digest")
        if self.platform != "linux/arm64":
            raise ValueError("build platform must be linux/arm64")
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or "\x00" in key
            or "\x00" in value
            for key, value in args.items()
        ):
            raise ValueError("build arguments must be NUL-free strings")


@dataclass(frozen=True)
class SandboxProxyManifest:
    container_id: str
    container_name: str
    host_port: int
    container_port: int
    protocol: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SandboxProxyManifest:
        return cls(
            container_id=str(raw["container_id"]),
            container_name=str(raw["container_name"]),
            host_port=int(raw["host_port"]),
            container_port=int(raw["container_port"]),
            protocol=str(raw["protocol"]),
        )


@dataclass(frozen=True)
class SandboxManifest:
    manifest_id: str
    sandbox_id: str
    container_id: str
    container_name: str
    state_hash: str
    spec_hash: str
    image: str
    proxies: tuple[SandboxProxyManifest, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["proxies"] = [proxy.to_dict() for proxy in self.proxies]
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SandboxManifest:
        return cls(
            manifest_id=str(raw["manifest_id"]),
            sandbox_id=str(raw["sandbox_id"]),
            container_id=str(raw["container_id"]),
            container_name=str(raw["container_name"]),
            state_hash=str(raw["state_hash"]),
            spec_hash=str(raw["spec_hash"]),
            image=str(raw["image"]),
            proxies=tuple(
                SandboxProxyManifest.from_dict(item)
                for item in raw.get("proxies", ())
            ),
        )


@dataclass(frozen=True)
class SandboxStatus:
    sandbox_id: str
    state: SandboxState
    state_hash: str
    created_at: datetime
    expires_at: datetime
    updated_at: datetime
    container_id: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class SandboxRecord:
    sandbox_id: str
    state: SandboxState
    state_hash: str
    spec_hash: str
    created_at: datetime
    expires_at: datetime
    updated_at: datetime
    manifest: SandboxManifest | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sandbox_id": self.sandbox_id,
            "state": self.state.value,
            "state_hash": self.state_hash,
            "spec_hash": self.spec_hash,
            "created_at": format_time(self.created_at),
            "expires_at": format_time(self.expires_at),
            "updated_at": format_time(self.updated_at),
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SandboxRecord:
        manifest = raw.get("manifest")
        created_at = parse_time(str(raw["created_at"]))
        expires_at = parse_time(str(raw["expires_at"]))
        updated_at = parse_time(str(raw["updated_at"]))
        if created_at is None or expires_at is None or updated_at is None:
            raise ValueError("sandbox record timestamps may not be null")
        return cls(
            sandbox_id=str(raw["sandbox_id"]),
            state=SandboxState(str(raw["state"])),
            state_hash=str(raw["state_hash"]),
            spec_hash=str(raw["spec_hash"]),
            created_at=created_at,
            expires_at=expires_at,
            updated_at=updated_at,
            manifest=SandboxManifest.from_dict(manifest) if manifest else None,
            detail=str(raw["detail"]) if raw.get("detail") is not None else None,
        )

    def status(self) -> SandboxStatus:
        return SandboxStatus(
            sandbox_id=self.sandbox_id,
            state=self.state,
            state_hash=self.state_hash,
            created_at=self.created_at,
            expires_at=self.expires_at,
            updated_at=self.updated_at,
            container_id=self.manifest.container_id if self.manifest else None,
            detail=self.detail,
        )
