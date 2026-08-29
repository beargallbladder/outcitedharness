from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass

from .backend import CommandRunner, SubprocessCommandRunner


_DNS_PORT = re.compile(
    r"^(?P<dns>[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?):(?P<port>[0-9]{1,5})$"
)


class PreviewError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreviewRoute:
    host_port: int
    https_port: int
    url: str


class TailscalePreviewPublisher:
    """Publish loopback sandbox ingress on a root-path tailnet-only HTTPS port."""

    def __init__(
        self,
        runner: CommandRunner | None = None,
        *,
        executable: str = "tailscale",
        timeout: float = 30.0,
    ) -> None:
        if not executable or "/" in executable or "\x00" in executable:
            raise ValueError("Tailscale executable must be a bare command name")
        if not 1 <= timeout <= 120:
            raise ValueError("preview timeout must be between 1 and 120 seconds")
        self.runner = runner or SubprocessCommandRunner()
        self.executable = executable
        self.timeout = timeout

    def publish(
        self,
        host_port: int,
        *,
        https_port: int | None = None,
    ) -> PreviewRoute:
        https_port = https_port or host_port
        self._validate_port(host_port)
        self._validate_port(https_port)
        self._execute(
            (
                self.executable,
                "serve",
                "--bg",
                "--yes",
                f"--https={https_port}",
                str(host_port),
            )
        )
        try:
            route = self.route(https_port)
        except Exception:
            self.runner.run(
                (
                    self.executable,
                    "serve",
                    "--yes",
                    f"--https={https_port}",
                    "off",
                ),
                timeout=self.timeout,
            )
            raise
        expected_proxy = f"http://127.0.0.1:{host_port}"
        status = self._status()
        host = route.url.removeprefix("https://").rstrip("/")
        handler = (
            status.get("Web", {})
            .get(host, {})
            .get("Handlers", {})
            .get("/", {})
        )
        if handler.get("Proxy") != expected_proxy:
            self.remove(route)
            raise PreviewError("Tailscale Serve published an unexpected proxy target")
        return route

    def route(self, https_port: int) -> PreviewRoute:
        self._validate_port(https_port)
        status = self._status()
        for key, value in status.get("Web", {}).items():
            match = _DNS_PORT.fullmatch(str(key))
            if not match or int(match.group("port")) != https_port:
                continue
            handler = value.get("Handlers", {}).get("/", {})
            proxy = str(handler.get("Proxy") or "")
            expected_prefix = "http://127.0.0.1:"
            if not proxy.startswith(expected_prefix):
                continue
            try:
                host_port = int(proxy.removeprefix(expected_prefix))
            except ValueError:
                continue
            return PreviewRoute(
                host_port=host_port,
                https_port=https_port,
                url=f"https://{key}/",
            )
        raise PreviewError(f"no root-path Tailscale preview exists on {https_port}")

    def remove(self, route: PreviewRoute) -> None:
        self._validate_port(route.https_port)
        self._execute(
            (
                self.executable,
                "serve",
                "--yes",
                f"--https={route.https_port}",
                "off",
            )
        )

    def probe(self, route: PreviewRoute, *, expected: str | None = None) -> str:
        request = urllib.request.Request(route.url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read(1024 * 1024).decode("utf-8", errors="replace")
        except Exception as exc:
            raise PreviewError(f"preview health check failed: {exc}") from exc
        if expected is not None and expected not in body:
            raise PreviewError("preview response did not contain the expected marker")
        return body

    def _status(self) -> dict[str, object]:
        result = self._execute((self.executable, "serve", "status", "--json"))
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise PreviewError("Tailscale Serve returned malformed status JSON") from exc
        if not isinstance(payload, dict):
            raise PreviewError("Tailscale Serve status must be an object")
        return payload

    def _execute(self, argv: tuple[str, ...]):
        result = self.runner.run(argv, timeout=self.timeout)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().replace("\n", " ")
            raise PreviewError(
                f"Tailscale Serve command failed ({result.returncode}): {detail[:600]}"
            )
        return result

    @staticmethod
    def _validate_port(port: int) -> None:
        if not 20_000 <= port <= 45_000:
            raise ValueError("preview port must be between 20000 and 45000")
