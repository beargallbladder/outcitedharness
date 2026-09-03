#!/usr/bin/env python3
"""Run a work-stealing datasheet extraction pool over durable chunk leases."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from harness.electronics.factory_control import ElectronicsFactoryState


DEFAULT_WORKERS = (
    ("dgx2", "http://192.168.4.45:8912/v1"),
    ("asus1", "http://192.168.4.58:8912/v1"),
    ("dgx3", "http://192.168.4.49:8912/v1"),
    ("asus3", "http://192.168.4.32:8912/v1"),
    ("asus2", "http://192.168.4.39:8912/v1"),
    ("asus4", "http://192.168.4.56:8912/v1"),
)


def _worker(value: str) -> tuple[str, str]:
    try:
        name, url = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("worker must be NAME=BASE_URL") from exc
    if not name or name != name.strip() or not url.startswith(("http://", "https://")):
        raise argparse.ArgumentTypeError("worker must have a name and HTTP(S) URL")
    return name, url.rstrip("/")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--structural-queue", type=Path, required=True)
    parser.add_argument("--page-evidence", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--worker", action="append", type=_worker, default=[])
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--lease-seconds", type=int, default=1800)
    parser.add_argument("--render-dpi", type=int, default=220)
    parser.add_argument("--request-timeout-seconds", type=float, default=900)
    parser.add_argument("--readiness-timeout-seconds", type=int, default=1200)
    parser.add_argument(
        "--vision-policy",
        choices=("fallback", "always"),
        default="always",
    )
    return parser


def _wait_ready(
    workers: list[tuple[str, str]],
    model: str,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    pending = dict(workers)
    while pending and time.monotonic() < deadline:
        for name, base_url in list(pending.items()):
            try:
                response = httpx.get(f"{base_url}/models", timeout=5)
                response.raise_for_status()
                models = {
                    str(item.get("id"))
                    for item in response.json().get("data", [])
                    if isinstance(item, dict)
                }
                if model in models:
                    del pending[name]
            except (httpx.HTTPError, ValueError, TypeError):
                continue
        if pending:
            time.sleep(5)
    if pending:
        raise RuntimeError(
            f"workers did not become ready for {model}: {sorted(pending)}"
        )


def _worker_command(
    args: argparse.Namespace,
    name: str,
    base_url: str,
) -> list[str]:
    repository = Path(__file__).resolve().parents[1]
    return [
        sys.executable,
        str(repository / "scripts" / "run_datasheet_factory_worker.py"),
        "--state-root",
        str(args.state_root),
        "--node",
        name,
        "--base-url",
        base_url,
        "--model",
        args.model,
        "--page-evidence",
        str(args.page_evidence),
        "--render-dpi",
        str(args.render_dpi),
        "--request-timeout-seconds",
        str(args.request_timeout_seconds),
        "--lease-seconds",
        str(args.lease_seconds),
        "--max-jobs",
        "10000",
        "--vision-policy",
        args.vision_policy,
    ]


def main() -> int:
    args = _parser().parse_args()
    workers = list(args.worker or DEFAULT_WORKERS)
    if len({name for name, _url in workers}) != len(workers):
        raise ValueError("worker names must be unique")
    if not workers:
        raise ValueError("at least one worker is required")
    state = ElectronicsFactoryState(args.state_root)
    chunk_ids = state.register_queue(
        args.structural_queue,
        args.output_root,
        chunk_size=args.chunk_size,
        max_attempts=args.max_attempts,
        start_offset=args.start_offset,
    )
    _wait_ready(workers, args.model, args.readiness_timeout_seconds)
    logs = state.root / "pool-logs"
    logs.mkdir(mode=0o700, exist_ok=True)
    os.chmod(logs, 0o700)

    round_number = 0
    while True:
        state.recover_expired()
        snapshot = state.status()
        queued = int(snapshot["chunks"].get("queued", 0))
        leased = int(snapshot["chunks"].get("leased", 0))
        failed = int(snapshot["chunks"].get("failed", 0))
        if failed:
            print(json.dumps(snapshot, indent=2, sort_keys=True))
            return 1
        if not queued and not leased:
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "registered_chunks": len(chunk_ids),
                        "factory": snapshot,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if not queued and leased:
            raise RuntimeError(
                "pool has active leases but no local worker processes"
            )

        round_number += 1
        processes: list[tuple[str, subprocess.Popen[bytes], object]] = []
        for name, base_url in workers:
            log_path = logs / f"round-{round_number:04d}-{name}.log"
            descriptor = os.open(
                log_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            handle = os.fdopen(descriptor, "wb")
            process = subprocess.Popen(
                _worker_command(args, name, base_url),
                cwd=Path(__file__).resolve().parents[1],
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((name, process, handle))

        return_codes: dict[str, int] = {}
        for name, process, handle in processes:
            try:
                return_codes[name] = process.wait()
            finally:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
        if any(code not in {0, 1} for code in return_codes.values()):
            raise RuntimeError(f"worker process failed unexpectedly: {return_codes}")
        if any(code == 1 for code in return_codes.values()):
            time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
