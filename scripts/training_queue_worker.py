#!/usr/bin/env python3
"""Run one allowlisted worker for the durable learning-factory queue."""

from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import asdict
from pathlib import Path

from harness.storage.db import Store
from harness.training.worker import TrainingWorker, load_handlers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--handlers", required=True, type=Path)
    parser.add_argument("--node", required=True)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--lease-seconds", type=int, default=1800)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    worker = TrainingWorker(
        Store(args.database),
        node=args.node,
        handlers=load_handlers(args.handlers),
        log_root=args.log_root,
        lease_seconds=args.lease_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    running = True

    def stop(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while running:
        result = worker.run_once()
        payload = asdict(result)
        if payload["log_path"] is not None:
            payload["log_path"] = str(payload["log_path"])
        print(json.dumps(payload, sort_keys=True), flush=True)
        if args.once:
            return 0
        if result.status == "idle":
            time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
