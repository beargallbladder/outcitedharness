#!/usr/bin/env python3
"""Four-node native-TP load/forward probe for Qwen3.8 with sharded PLE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM


SCHEMA = "harness.qwen38-tp-load-smoke.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequence-length", type=int, default=32)
    arguments = parser.parse_args()
    if not 8 <= arguments.sequence_length <= 256:
        raise ValueError("sequence length must be between 8 and 256")
    if arguments.output.exists() or arguments.output.is_symlink():
        raise ValueError("smoke output already exists")
    config_path = arguments.model / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("model config must be a regular file")

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 4 or rank not in range(4):
        raise ValueError("Qwen3.8 TP probe requires exactly four ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        arguments.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        tp_plan="auto",
    )
    model.config.use_cache = False
    load_seconds = time.perf_counter() - load_started

    ple_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.endswith("ple.ple_embedding.ngram_embedding.weight")
    ]
    if len(ple_parameters) != 1:
        raise RuntimeError(
            f"expected one runtime PLE table, observed {len(ple_parameters)}"
        )
    ple_name, ple_parameter = ple_parameters[0]
    local_ple = _local_tensor(ple_parameter)
    ple_sharded = (
        local_ple.numel() * world_size == ple_parameter.numel()
        and tuple(local_ple.shape) != tuple(ple_parameter.shape)
    )
    if not ple_sharded:
        raise RuntimeError("native TP did not shard the runtime PLE table")

    input_ids = torch.arange(
        1,
        arguments.sequence_length + 1,
        dtype=torch.long,
        device=torch.device("cuda", local_rank),
    ).unsqueeze(0)
    forward_started = time.perf_counter()
    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=False)
    torch.cuda.synchronize()
    logits = _local_tensor(output.logits)
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Qwen3.8 forward produced non-finite logits")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    rank_result: dict[str, Any] = {
        "rank": rank,
        "hostname": os.uname().nodename,
        "world_size": world_size,
        "model_config_sha256": _sha256(config_path),
        "load_passed": True,
        "forward_passed": True,
        "ple_sharded": ple_sharded,
        "ple_parameter": ple_name,
        "ple_global_shape": list(ple_parameter.shape),
        "ple_local_shape": list(local_ple.shape),
        "ple_placements": [str(value) for value in getattr(ple_parameter, "placements", ())],
        "load_seconds": round(load_seconds, 3),
        "forward_seconds": round(time.perf_counter() - forward_started, 3),
        "peak_cuda_memory_gib": round(
            torch.cuda.max_memory_allocated() / 1024**3,
            3,
        ),
        "free_cuda_memory_gib": round(free_bytes / 1024**3, 3),
        "total_cuda_memory_gib": round(total_bytes / 1024**3, 3),
        "gid_index": os.environ.get("NCCL_IB_GID_INDEX", ""),
        "hcas": os.environ.get("NCCL_IB_HCA", ""),
    }
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, rank_result)
    if rank == 0:
        core: dict[str, Any] = {
            "schema": SCHEMA,
            "world_size": world_size,
            "sequence_length": arguments.sequence_length,
            "all_passed": all(
                row is not None
                and row["load_passed"]
                and row["forward_passed"]
                and row["ple_sharded"]
                for row in gathered
            ),
            "ranks": gathered,
        }
        core["evidence_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
        temporary.write_text(json.dumps(core, indent=2, sort_keys=True) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, arguments.output)
        os.chmod(arguments.output, 0o444)
        print(json.dumps(core, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
