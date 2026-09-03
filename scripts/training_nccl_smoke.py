#!/usr/bin/env python3
"""Verify multi-rank NCCL correctness and measure collective bandwidth."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.distributed as dist


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--expected-world-size", type=int, default=2)
    parser.add_argument(
        "--sizes-mib", type=int, nargs="+", default=[1, 64, 256]
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 3:
        parser.error("warmup must be >=1 and iterations must be >=3")
    if args.expected_world_size < 2 or args.expected_world_size > 6:
        parser.error("expected world size must be between 2 and 6")
    if any(size < 1 or size > 1024 for size in args.sizes_mib):
        parser.error("sizes must be between 1 and 1024 MiB")

    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"expected exactly {args.expected_world_size} ranks, "
            f"observed {world_size}"
        )
    torch.cuda.set_device(0)
    expected_sum = world_size * (world_size + 1) / 2

    rows: list[dict[str, float | int]] = []
    for size_mib in args.sizes_mib:
        elements = size_mib * 1024 * 1024 // 4
        tensor = torch.full(
            (elements,),
            float(rank + 1),
            dtype=torch.float32,
            device="cuda",
        )
        for _ in range(args.warmup):
            dist.all_reduce(tensor)
            tensor.fill_(float(rank + 1))
        dist.barrier()

        durations: list[float] = []
        for _ in range(args.iterations):
            tensor.fill_(float(rank + 1))
            torch.cuda.synchronize()
            started = time.perf_counter()
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
            if not bool(torch.all(tensor == expected_sum)):
                raise RuntimeError(
                    f"rank {rank} observed an incorrect all-reduce result"
                )
        dist.barrier()
        median_seconds = statistics.median(durations)
        bytes_per_collective = size_mib * 1024 * 1024
        algorithm_gbps = bytes_per_collective * 8 / median_seconds / 1e9
        bus_factor = 2 * (world_size - 1) / world_size
        rows.append(
            {
                "size_mib": size_mib,
                "iterations": args.iterations,
                "median_ms": round(median_seconds * 1000, 4),
                "p95_ms": round(percentile(durations, 0.95) * 1000, 4),
                "algorithm_gbps": round(algorithm_gbps, 4),
                "bus_gbps": round(algorithm_gbps * bus_factor, 4),
            }
        )

    payload = {
        "schema": "harness.training.nccl-qualification.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "world_size": world_size,
        "backend": dist.get_backend(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": ".".join(str(value) for value in torch.cuda.nccl.version()),
        "hostname": os.uname().nodename,
        "interface": os.environ.get("NCCL_SOCKET_IFNAME", ""),
        "hcas": os.environ.get("NCCL_IB_HCA", ""),
        "gid_index": os.environ.get("NCCL_IB_GID_INDEX", ""),
        "correct": True,
        "measurements": rows,
    }
    if rank == 0:
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        print(encoded, end="")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, args.output)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
