#!/usr/bin/env python3
"""Two-node native tensor-parallel load/LoRA-step probe for Qwen3-Next."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


SCHEMA = "harness.qwen3-next-tp-lora-smoke.v1"
TARGET_MODULES = (
    "in_proj_qkvz",
    "in_proj_ba",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _allocated_gib() -> float:
    return round(torch.cuda.memory_allocated() / 1024**3, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adapter-output", type=Path)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--load-only", action="store_true")
    arguments = parser.parse_args()

    if not 8 <= arguments.sequence_length <= 1024:
        raise ValueError("smoke sequence length must be between 8 and 1024")
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 2 or rank not in {0, 1}:
        raise ValueError("probe requires exactly two torchrun ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        arguments.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        tp_plan="auto",
    )
    model.config.use_cache = False
    load_seconds = time.perf_counter() - started
    base_memory_gib = _allocated_gib()
    result: dict[str, object] = {
        "schema": SCHEMA,
        "rank": rank,
        "world_size": world_size,
        "hostname": os.uname().nodename,
        "model_config_sha256": _sha256(
            (arguments.model / "config.json").read_bytes()
        ),
        "load_seconds": round(load_seconds, 3),
        "base_memory_gib": base_memory_gib,
        "sequence_length": arguments.sequence_length,
        "load_passed": True,
        "lora_injection_passed": False,
        "optimizer_step_passed": False,
        "adapter_save_passed": False,
    }
    if not arguments.load_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(TARGET_MODULES),
            ),
        )
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if trainable <= 0:
            raise RuntimeError("LoRA injection produced no trainable parameters")
        result["lora_injection_passed"] = True
        result["trainable_parameters"] = trainable
        tokenizer = AutoTokenizer.from_pretrained(
            arguments.model,
            local_files_only=True,
            trust_remote_code=True,
        )
        inputs = tokenizer(
            "Repair this Python function and return a unified diff.\n"
            "def add(a, b):\n    return a - b\n",
            return_tensors="pt",
            truncation=True,
            max_length=arguments.sequence_length,
            padding="max_length",
        )
        input_ids = inputs["input_ids"].to(local_rank)
        attention_mask = inputs["attention_mask"].to(local_rank)
        labels = input_ids.masked_fill(attention_mask == 0, -100)
        model.train()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-4,
        )
        step_started = time.perf_counter()
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        output.loss.backward()
        finite_gradients = all(
            bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        if not finite_gradients:
            raise RuntimeError("LoRA gradients are not finite")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        result["optimizer_step_passed"] = True
        result["loss"] = float(output.loss.detach())
        result["step_seconds"] = round(time.perf_counter() - step_started, 3)
        result["peak_memory_gib"] = round(
            torch.cuda.max_memory_allocated() / 1024**3,
            3,
        )
        if arguments.adapter_output is not None:
            destination = arguments.adapter_output / f"rank-{rank}"
            if destination.exists():
                shutil.rmtree(destination)
            model.save_pretrained(destination, safe_serialization=True)
            required = destination / "adapter_config.json"
            if not required.is_file():
                raise RuntimeError("adapter save omitted adapter_config.json")
            result["adapter_save_passed"] = True
            result["adapter_config_sha256"] = _sha256(required.read_bytes())

    gathered: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(gathered, result)
    if rank == 0:
        core = {
            "schema": SCHEMA,
            "world_size": world_size,
            "all_passed": all(
                row is not None
                and row["load_passed"]
                and (
                    arguments.load_only
                    or (
                        row["lora_injection_passed"]
                        and row["optimizer_step_passed"]
                        and row["adapter_save_passed"]
                    )
                )
                for row in gathered
            ),
            "ranks": gathered,
        }
        core["evidence_sha256"] = _sha256(
            json.dumps(
                core,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        if arguments.output.exists() or arguments.output.is_symlink():
            raise ValueError("smoke output already exists")
        temporary = arguments.output.with_suffix(".tmp")
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
