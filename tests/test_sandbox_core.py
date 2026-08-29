from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import pytest

from harness.sandbox import (
    BuildSpec,
    CommandResult,
    DockerCLIBackend,
    EgressMode,
    EgressPolicy,
    JsonSandboxRegistry,
    Mount,
    PolicyViolation,
    PortBinding,
    PreviewRoute,
    RecordExistsError,
    ResourceLimits,
    SandboxPolicy,
    SandboxService,
    SandboxSpec,
    SandboxState,
)
from harness.sandbox.backend import (
    DENY_NETWORK,
    MANAGED_LABEL,
    PROXY_IMAGE,
    STATE_HASH_LABEL,
)


STATE_HASH = "a" * 64
CONTAINER_ID = "b" * 64
PROXY_ID = "e" * 64
NETWORK_JSON = json.dumps(
    [
        {
            "Name": DENY_NETWORK,
            "Driver": "bridge",
            "Internal": True,
            "Labels": {MANAGED_LABEL: "true"},
        }
    ]
)


class RecordingRunner:
    def __init__(self, results: list[CommandResult] | None = None) -> None:
        self.results = list(results or [])
        self.calls: list[tuple[tuple[str, ...], float]] = []

    def run(self, argv: Sequence[str], *, timeout: float) -> CommandResult:
        self.calls.append((tuple(argv), timeout))
        return self.results.pop(0) if self.results else CommandResult(0)


class FakeDockerRunner:
    """In-memory Docker CLI surface; no process or container is started."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.containers: dict[str, dict[str, object]] = {}

    def run(self, argv: Sequence[str], *, timeout: float) -> CommandResult:
        del timeout
        args = tuple(argv)
        self.calls.append(args)
        operation = args[1]
        if operation == "network":
            if args[2] == "inspect":
                return CommandResult(0, NETWORK_JSON)
            if args[2] == "create":
                return CommandResult(0, f"{'d' * 64}\n")
            if args[2] == "connect":
                return CommandResult(0)
            raise AssertionError(f"unexpected fake network operation: {args[2]}")
        if operation == "create":
            labels = {
                args[index + 1].split("=", 1)[0]: args[index + 1].split("=", 1)[1]
                for index, value in enumerate(args)
                if value == "--label"
            }
            name = args[args.index("--name") + 1]
            is_proxy = labels.get("io.harness.sandbox.proxy") == "true"
            container_id = PROXY_ID if is_proxy else CONTAINER_ID
            image = (
                args[args.index(PROXY_IMAGE)]
                if is_proxy
                else args[self._image_index(args)]
            )
            self.containers[container_id] = {
                "Id": container_id,
                "Name": f"/{name}",
                "Config": {"Image": image, "Labels": labels},
                "State": {"Running": False, "Status": "created"},
            }
            return CommandResult(0, f"{container_id}\n")
        if operation == "inspect":
            ids = args[4:]
            rows = [self.containers[item] for item in ids if item in self.containers]
            if len(rows) != len(ids):
                return CommandResult(1, stderr="No such container")
            return CommandResult(0, json.dumps(rows))
        if operation == "start":
            container = self.containers[args[2]]
            container["State"] = {"Running": True, "Status": "running"}
            return CommandResult(0, f"{args[2]}\n")
        if operation == "stop":
            container = self.containers[args[-1]]
            container["State"] = {"Running": False, "Status": "exited"}
            return CommandResult(0, f"{args[-1]}\n")
        if operation == "rm":
            self.containers.pop(args[-1])
            return CommandResult(0, f"{args[-1]}\n")
        if operation == "logs":
            return CommandResult(0, "application ready\n")
        if operation == "ps":
            return CommandResult(0, "\n".join(self.containers))
        raise AssertionError(f"unexpected fake Docker operation: {operation}")

    @staticmethod
    def _image_index(args: tuple[str, ...]) -> int:
        # Test specs have no command. The image is therefore the final argument.
        return len(args) - 1


class FakePreviewPublisher:
    def __init__(self) -> None:
        self.published: list[tuple[int, int | None]] = []
        self.removed: list[PreviewRoute] = []

    def publish(
        self, host_port: int, *, https_port: int | None = None
    ) -> PreviewRoute:
        selected = https_port or host_port
        self.published.append((host_port, https_port))
        return PreviewRoute(
            host_port=host_port,
            https_port=selected,
            url=f"https://m5.tailnet.ts.net:{selected}/",
        )

    def remove(self, route: PreviewRoute) -> None:
        self.removed.append(route)


def policy(root: Path, **overrides: object) -> SandboxPolicy:
    return SandboxPolicy(allowed_mount_roots=(root,), **overrides)


def spec(root: Path, **overrides: object) -> SandboxSpec:
    values: dict[str, object] = {
        "sandbox_id": "job-1",
        "image": "python:3.12-slim",
        "state_hash": STATE_HASH,
        "mounts": (Mount(root, read_only=False),),
        "limits": ResourceLimits(cpus=1.5, memory_bytes=256 * 1024 * 1024, pids=64),
        "ports": (PortBinding(container_port=8000, host_port=20_001),),
    }
    values.update(overrides)
    return SandboxSpec(**values)


def test_docker_create_is_arm64_non_root_and_locked_down(tmp_path: Path) -> None:
    runner = RecordingRunner(
        [
            CommandResult(0, NETWORK_JSON),
            CommandResult(0, f"{CONTAINER_ID}\n"),
            CommandResult(0, f"{PROXY_ID}\n"),
            CommandResult(0),
        ]
    )
    backend = DockerCLIBackend(policy(tmp_path), runner)

    manifest = backend.create(spec(tmp_path))

    argv, _ = runner.calls[1]
    assert argv[:2] == ("docker", "create")
    assert ("--platform", "linux/arm64") == _pair(argv, "--platform")
    assert ("--user", "65532:65532") == _pair(argv, "--user")
    assert ("--cpus", "1.5") == _pair(argv, "--cpus")
    assert ("--memory", str(256 * 1024 * 1024)) == _pair(argv, "--memory")
    assert ("--pids-limit", "64") == _pair(argv, "--pids-limit")
    assert ("--cap-drop", "ALL") == _pair(argv, "--cap-drop")
    assert ("--network", DENY_NETWORK) == _pair(argv, "--network")
    assert "no-new-privileges:true" in argv
    assert "--read-only" in argv
    assert "--privileged" not in argv
    assert "--network=host" not in argv
    assert "docker.sock" not in " ".join(argv)
    assert f"{STATE_HASH_LABEL}={STATE_HASH}" in argv
    assert manifest.container_id == CONTAINER_ID
    assert manifest.state_hash == STATE_HASH
    assert manifest.proxies[0].container_id == PROXY_ID


def test_build_defaults_to_offline_arm64_and_rejects_secret_args(
    tmp_path: Path,
) -> None:
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    runner = RecordingRunner([CommandResult(0)])
    backend = DockerCLIBackend(policy(tmp_path), runner)

    backend.build(
        BuildSpec(image="local/sandbox:test", context=tmp_path, state_hash=STATE_HASH)
    )
    argv, timeout = runner.calls[0]
    assert argv[:2] == ("docker", "build")
    assert _pair(argv, "--platform") == ("--platform", "linux/arm64")
    assert _pair(argv, "--network") == ("--network", "none")
    assert "--secret" not in argv
    assert timeout == 300.0

    with pytest.raises(PolicyViolation, match="secret-bearing"):
        backend.build(
            BuildSpec(
                image="local/sandbox:test",
                context=tmp_path,
                state_hash=STATE_HASH,
                build_args={"API_TOKEN": "not-allowed"},
            )
        )


def test_build_rejects_remote_add_and_secret_context_files(
    tmp_path: Path,
) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM --platform=linux/arm64 alpine\n"
        "ADD https://example.test/app /app\n",
        encoding="utf-8",
    )
    build = BuildSpec(
        image="local/sandbox:test",
        context=tmp_path,
        state_hash=STATE_HASH,
    )
    sandbox_policy = policy(tmp_path)

    with pytest.raises(PolicyViolation, match="remote ADD"):
        sandbox_policy.validate_build(build)

    dockerfile.write_text(
        "FROM --platform=linux/arm64 alpine\nCOPY . /app\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("TOKEN=do-not-copy\n", encoding="utf-8")
    with pytest.raises(PolicyViolation, match="secret-bearing build input"):
        sandbox_policy.validate_build(build)


def test_build_rejects_symlinks_and_foreign_platforms(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM --platform=linux/amd64 alpine\n", encoding="utf-8"
    )
    build = BuildSpec(
        image="local/sandbox:test",
        context=tmp_path,
        state_hash=STATE_HASH,
    )
    sandbox_policy = policy(tmp_path)

    with pytest.raises(PolicyViolation, match="ARM64"):
        sandbox_policy.validate_build(build)

    dockerfile.write_text(
        "FROM --platform=linux/arm64 alpine\n", encoding="utf-8"
    )
    (tmp_path / "outside-link").symlink_to(tmp_path.parent / "outside")
    with pytest.raises(PolicyViolation, match="non-regular build input"):
        sandbox_policy.validate_build(build)


def test_policy_rejects_secrets_socket_host_network_and_unbounded_ports(
    tmp_path: Path,
) -> None:
    socket = tmp_path / "docker.sock"
    socket.touch()
    sandbox_policy = policy(tmp_path)

    with pytest.raises(PolicyViolation, match="secret-bearing"):
        sandbox_policy.validate(spec(tmp_path, environment={"API_TOKEN": "secret"}))
    with pytest.raises(PolicyViolation, match="sockets"):
        sandbox_policy.validate(spec(tmp_path, mounts=(Mount(socket),)))
    with pytest.raises(PolicyViolation, match="outside the allowed range"):
        sandbox_policy.validate(
            spec(
                tmp_path,
                ports=(PortBinding(container_port=80, host_port=19_999),),
            )
        )
    with pytest.raises(PolicyViolation, match="unrestricted"):
        sandbox_policy.validate(
            spec(
                tmp_path,
                egress=EgressPolicy(mode=EgressMode.ALLOW_ALL),
            )
        )


def test_allowlist_requires_policy_enforcing_network(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="policy network"):
        EgressPolicy(mode=EgressMode.ALLOWLIST, allowed_hosts=("pypi.org",))

    runner = RecordingRunner([CommandResult(0, f"{CONTAINER_ID}\n")])
    backend = DockerCLIBackend(
        policy(tmp_path, egress_networks={"harness-egress-pypi": ("pypi.org",)}),
        runner,
    )
    backend.create(
        spec(
            tmp_path,
            egress=EgressPolicy(
                mode=EgressMode.ALLOWLIST,
                allowed_hosts=("pypi.org",),
                policy_network="harness-egress-pypi",
            ),
        )
    )
    assert _pair(runner.calls[0][0], "--network") == (
        "--network",
        "harness-egress-pypi",
    )


def test_registry_survives_restart_and_service_reaps_by_manifest(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 29, tzinfo=timezone.utc)
    current = [now]
    runner = FakeDockerRunner()
    backend = DockerCLIBackend(policy(tmp_path), runner)
    registry_path = tmp_path / ".state" / "sandboxes.json"
    service = SandboxService(
        backend,
        JsonSandboxRegistry(registry_path),
        clock=lambda: current[0],
    )

    created = service.create(spec(tmp_path, ttl_seconds=60))
    assert created.state is SandboxState.RUNNING

    restarted = SandboxService(
        DockerCLIBackend(policy(tmp_path), runner),
        JsonSandboxRegistry(registry_path),
        clock=lambda: current[0],
    )
    recovered = restarted.recover()
    assert recovered[0].state is SandboxState.RUNNING
    assert recovered[0].container_id == CONTAINER_ID

    current[0] = now + timedelta(seconds=61)
    reaped = restarted.reap_expired()
    assert reaped[0].state is SandboxState.REMOVED
    assert not runner.containers
    persisted = JsonSandboxRegistry(registry_path).require("job-1")
    assert persisted.state is SandboxState.REMOVED
    inspect_calls = [call for call in runner.calls if call[1] == "inspect"]
    assert inspect_calls
    assert all(call[-1] in {CONTAINER_ID, PROXY_ID} for call in inspect_calls)


def test_cleanup_refuses_a_state_hash_label_mismatch(tmp_path: Path) -> None:
    runner = FakeDockerRunner()
    backend = DockerCLIBackend(policy(tmp_path), runner)
    service = SandboxService(
        backend,
        JsonSandboxRegistry(tmp_path / "registry.json"),
    )
    service.create(spec(tmp_path), start=False)
    labels = runner.containers[CONTAINER_ID]["Config"]["Labels"]  # type: ignore[index]
    labels[STATE_HASH_LABEL] = "c" * 64  # type: ignore[index]

    with pytest.raises(Exception, match="not owned"):
        service.destroy("job-1")
    assert CONTAINER_ID in runner.containers


def test_preview_route_is_persisted_and_removed_before_sandbox(
    tmp_path: Path,
) -> None:
    runner = FakeDockerRunner()
    publisher = FakePreviewPublisher()
    registry = JsonSandboxRegistry(tmp_path / "registry.json")
    service = SandboxService(
        DockerCLIBackend(policy(tmp_path), runner),
        registry,
        preview_publisher=publisher,  # type: ignore[arg-type]
    )
    service.create(spec(tmp_path))

    published = service.publish_preview("job-1")

    assert published.preview_url == "https://m5.tailnet.ts.net:20001/"
    assert registry.require("job-1").preview_https_port == 20_001
    assert service.logs("job-1", tail=20) == "application ready\n"

    removed = service.destroy("job-1")

    assert removed.state is SandboxState.REMOVED
    assert publisher.removed[0].https_port == 20_001
    assert registry.require("job-1").preview_url is None
    assert not runner.containers


def test_active_sandbox_limit_is_enforced_atomically(tmp_path: Path) -> None:
    runner = FakeDockerRunner()
    service = SandboxService(
        DockerCLIBackend(policy(tmp_path), runner),
        JsonSandboxRegistry(tmp_path / "registry.json"),
        max_active_sandboxes=1,
    )
    service.create(spec(tmp_path))

    with pytest.raises(RecordExistsError, match="active sandbox limit"):
        service.create(spec(tmp_path, sandbox_id="job-2"))

    service.destroy("job-1")
    assert service.create(spec(tmp_path, sandbox_id="job-2")).state is SandboxState.RUNNING


def _pair(argv: tuple[str, ...], option: str) -> tuple[str, str]:
    index = argv.index(option)
    return argv[index], argv[index + 1]
