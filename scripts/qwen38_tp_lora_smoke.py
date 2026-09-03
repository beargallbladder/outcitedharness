#!/usr/bin/env python3
"""Four-node native-TP Qwen3.8 replicated-LoRA load/step/save smoke."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM


SCHEMA = "harness.qwen38-tp-lora-smoke.v1"
TARGET_MODULES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "out_proj",
    }
)
EXPECTED_MODULES = 228
EXPECTED_TRAINABLE_ELEMENTS = 13_052_928


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class ReplicatedLoRALinear(nn.Module):
    """Add a replicated LoRA branch after a TP layer's gathered output."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        device = base_layer.weight.device
        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                base_layer.in_features,
                dtype=torch.bfloat16,
                device=device,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base_layer.out_features,
                rank,
                dtype=torch.bfloat16,
                device=device,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(inputs)
        if hasattr(base_output, "to_local"):
            raise RuntimeError(
                "target layer returned a sharded DTensor; replicated LoRA "
                "requires a gathered local output"
            )
        delta = F.linear(F.linear(inputs, self.lora_A), self.lora_B)
        return base_output + delta * self.scaling


def _inject_lora(model: nn.Module, rank: int, alpha: int) -> dict[str, ReplicatedLoRALinear]:
    selected: dict[str, ReplicatedLoRALinear] = {}
    for name, module in list(model.named_modules()):
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in TARGET_MODULES:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA target {name} is not nn.Linear")
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        replacement = ReplicatedLoRALinear(module, rank=rank, alpha=alpha)
        setattr(parent, child_name, replacement)
        selected[name] = replacement
    return selected


def _adapter_state(
    modules: dict[str, ReplicatedLoRALinear],
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in sorted(modules.items()):
        state[f"{name}.lora_A.weight"] = module.lora_A.detach().cpu().contiguous()
        state[f"{name}.lora_B.weight"] = module.lora_B.detach().cpu().contiguous()
    return state


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _all_rank_values(value: Any, world_size: int) -> list[Any]:
    values: list[Any] = [None] * world_size
    dist.all_gather_object(values, value)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter-output", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    arguments = parser.parse_args()
    if not 8 <= arguments.sequence_length <= 256:
        raise ValueError("sequence length must be between 8 and 256")
    if arguments.rank != 8 or arguments.alpha != 16:
        raise ValueError("qualification requires rank 8 and alpha 16")
    for output in (arguments.adapter_output, arguments.receipt_output):
        if output.exists() or output.is_symlink():
            raise ValueError(f"immutable output already exists: {output}")
    config_path = arguments.model / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("model config must be a regular file")

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    process_rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 4 or process_rank not in range(4):
        raise ValueError("Qwen3.8 TP LoRA smoke requires exactly four ranks")
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    load_seconds = time.perf_counter() - load_started

    ple = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("ple.ple_embedding.ngram_embedding.weight")
    ]
    if len(ple) != 1:
        raise RuntimeError(f"expected one PLE table, observed {len(ple)}")
    ple_local = ple[0].to_local() if hasattr(ple[0], "to_local") else ple[0]
    ple_sharded = (
        ple_local.numel() * world_size == ple[0].numel()
        and tuple(ple_local.shape) != tuple(ple[0].shape)
    )
    if not ple_sharded:
        raise RuntimeError("native TP did not shard the runtime PLE table")
    gc.collect()
    torch.cuda.empty_cache()

    torch.manual_seed(1788172800)
    torch.cuda.manual_seed_all(1788172800)
    modules = _inject_lora(model, rank=arguments.rank, alpha=arguments.alpha)
    trainable = [
        parameter
        for module in modules.values()
        for parameter in (module.lora_A, module.lora_B)
    ]
    trainable_elements = sum(parameter.numel() for parameter in trainable)
    if len(modules) != EXPECTED_MODULES:
        raise RuntimeError(
            f"expected {EXPECTED_MODULES} LoRA modules, observed {len(modules)}"
        )
    if trainable_elements != EXPECTED_TRAINABLE_ELEMENTS:
        raise RuntimeError(
            "unexpected LoRA parameter count: "
            f"{trainable_elements} != {EXPECTED_TRAINABLE_ELEMENTS}"
        )
    initial_digest = _state_digest(_adapter_state(modules))
    if len(set(_all_rank_values(initial_digest, world_size))) != 1:
        raise RuntimeError("replicated adapters did not initialize identically")

    input_ids = torch.arange(
        1,
        arguments.sequence_length + 1,
        dtype=torch.long,
        device=torch.device("cuda", local_rank),
    ).unsqueeze(0)
    labels = input_ids.clone()
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    model.train()
    step_started = time.perf_counter()
    output = model(input_ids=input_ids, labels=labels, use_cache=False)
    output.loss.backward()
    for parameter in trainable:
        if parameter.grad is None:
            raise RuntimeError("a LoRA parameter did not receive a gradient")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError("a LoRA parameter received a non-finite gradient")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - step_started

    state = _adapter_state(modules)
    updated_digest = _state_digest(state)
    if updated_digest == initial_digest:
        raise RuntimeError("optimizer step did not update the adapter")
    if len(set(_all_rank_values(updated_digest, world_size))) != 1:
        raise RuntimeError("replicated adapters diverged across TP ranks")

    adapter_sha256 = ""
    config_sha256 = _sha256(config_path)
    if process_rank == 0:
        arguments.adapter_output.mkdir(parents=True)
        adapter_path = arguments.adapter_output / "adapter_model.safetensors"
        adapter_config_path = arguments.adapter_output / "adapter_config.json"
        save_file(
            state,
            adapter_path,
            metadata={"format": "harness-replicated-tp-lora-v1"},
        )
        adapter_config = {
            "schema": "harness.replicated-tp-lora.v1",
            "base_model_config_sha256": config_sha256,
            "rank": arguments.rank,
            "alpha": arguments.alpha,
            "target_modules": sorted(TARGET_MODULES),
            "module_count": len(modules),
            "trainable_elements": trainable_elements,
            "adapter_state_sha256": updated_digest,
        }
        adapter_config_path.write_text(
            json.dumps(adapter_config, indent=2, sort_keys=True) + "\n"
        )
        os.chmod(adapter_path, 0o444)
        os.chmod(adapter_config_path, 0o444)
        adapter_sha256 = _sha256(adapter_path)
    dist.barrier()
    adapter_sha256 = _all_rank_values(adapter_sha256, world_size)[0]

    peak_memory_gib = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    rank_result = {
        "rank": process_rank,
        "hostname": os.uname().nodename,
        "loss": float(output.loss.detach()),
        "load_seconds": round(load_seconds, 3),
        "step_seconds": round(step_seconds, 3),
        "peak_memory_gib": peak_memory_gib,
        "free_memory_gib": round(free_bytes / 1024**3, 3),
        "total_memory_gib": round(total_bytes / 1024**3, 3),
        "adapter_state_sha256": updated_digest,
        "gid_index": os.environ.get("NCCL_IB_GID_INDEX", ""),
    }
    ranks = _all_rank_values(rank_result, world_size)
    if process_rank == 0:
        core: dict[str, Any] = {
            "schema": SCHEMA,
            "model_config_sha256": config_sha256,
            "world_size": world_size,
            "sequence_length": arguments.sequence_length,
            "ple_sharded": ple_sharded,
            "load_passed": True,
            "optimizer_step_passed": True,
            "finite_gradients": True,
            "adapter_save_passed": bool(adapter_sha256),
            "adapters_identical": len(
                {row["adapter_state_sha256"] for row in ranks}
            )
            == 1,
            "module_count": len(modules),
            "adapter_tensor_count": len(state),
            "trainable_elements": trainable_elements,
            "adapter_artifact_sha256": adapter_sha256,
            "max_peak_memory_gib": max(
                float(row["peak_memory_gib"]) for row in ranks
            ),
            "ranks": ranks,
        }
        core["evidence_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
        arguments.receipt_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.receipt_output.with_suffix(".tmp")
        temporary.write_text(json.dumps(core, indent=2, sort_keys=True) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, arguments.receipt_output)
        os.chmod(arguments.receipt_output, 0o444)
        print(json.dumps(core, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
