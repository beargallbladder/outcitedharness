from __future__ import annotations

import json

import pytest

from harness.sandbox import (
    CommandResult,
    PreviewError,
    TailscalePreviewPublisher,
)


class Runner:
    def __init__(self, results: list[CommandResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[tuple[str, ...], float]] = []

    def run(self, argv, *, timeout: float) -> CommandResult:
        self.calls.append((tuple(argv), timeout))
        return self.results.pop(0)


def _status(*, proxy_port: int = 20_000, https_port: int = 20_000) -> str:
    return json.dumps(
        {
            "TCP": {str(https_port): {"HTTPS": True}},
            "Web": {
                f"m5.tailnet.ts.net:{https_port}": {
                    "Handlers": {
                        "/": {"Proxy": f"http://127.0.0.1:{proxy_port}"}
                    }
                }
            },
        }
    )


def test_publish_uses_root_path_dedicated_https_port() -> None:
    runner = Runner(
        [
            CommandResult(0),
            CommandResult(0, _status()),
            CommandResult(0, _status()),
            CommandResult(0),
        ]
    )
    publisher = TailscalePreviewPublisher(runner)

    route = publisher.publish(20_000)
    publisher.remove(route)

    assert route.url == "https://m5.tailnet.ts.net:20000/"
    assert route.host_port == 20_000
    assert runner.calls[0][0] == (
        "tailscale",
        "serve",
        "--bg",
        "--yes",
        "--https=20000",
        "20000",
    )
    assert runner.calls[-1][0] == (
        "tailscale",
        "serve",
        "--yes",
        "--https=20000",
        "off",
    )


def test_publish_removes_route_when_proxy_target_is_wrong() -> None:
    runner = Runner(
        [
            CommandResult(0),
            CommandResult(0, _status(proxy_port=20_001)),
            CommandResult(0, _status(proxy_port=20_001)),
            CommandResult(0),
        ]
    )
    publisher = TailscalePreviewPublisher(runner)

    with pytest.raises(PreviewError, match="unexpected proxy target"):
        publisher.publish(20_000)

    assert runner.calls[-1][0][-1] == "off"


def test_route_rejects_malformed_status() -> None:
    publisher = TailscalePreviewPublisher(
        Runner([CommandResult(0, "not-json")])
    )

    with pytest.raises(PreviewError, match="malformed"):
        publisher.route(20_000)


def test_preview_ports_are_bounded() -> None:
    publisher = TailscalePreviewPublisher(Runner([]))

    with pytest.raises(ValueError, match="between 20000 and 45000"):
        publisher.publish(19_999)
