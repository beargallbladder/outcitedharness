#!/usr/bin/env python3
"""Decisive four-node Qwen3.8 file-PLE parity and real-data training gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen38_file_backed_ple import replace_ple_embedding
from qwen38_tp_lora_smoke import (
    EXPECTED_MODULES,
    EXPECTED_TRAINABLE_ELEMENTS,
    TARGET_MODULES,
    _adapter_state,
    _inject_lora,
    _state_digest,
)


SCHEMA = "harness.qwen38-file-ple-training-gate.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _all_rank_values(value: Any, world_size: int) -> list[Any]:
    values: list[Any] = [None] * world_size
    dist.all_gather_object(values, value)
    return values


def _ple_module(model: torch.nn.Module) -> torch.nn.Module:
    modules = [
        module
        for name, module in model.named_modules()
        if name.endswith("ple.ple_embedding.ngram_embedding")
    ]
    if len(modules) != 1:
        raise RuntimeError(f"expected one resident PLE, observed {len(modules)}")
    return modules[0]


def _real_training_example(
    *,
    tokenizer: Any,
    dataset_path: Path,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    rows = json.loads(dataset_path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("owned coding dataset is empty or malformed")
    candidates = []
    for index, row in enumerate(rows):
        messages = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"owned coding row {index} is malformed")
        prompt = tokenizer.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        response_tokens = len(full_ids) - len(prompt_ids)
        if len(full_ids) <= max_length and response_tokens >= 64:
            candidates.append(
                (
                    len(full_ids),
                    index,
                    len(prompt_ids),
                    response_tokens,
                    full_ids,
                )
            )
    if not candidates:
        raise RuntimeError(
            f"no complete owned coding record fits {max_length} tokens"
        )
    total_tokens, index, prompt_tokens, response_tokens, full_ids = min(candidates)
    input_ids = torch.tensor([full_ids], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :prompt_tokens] = -100
    return input_ids, labels, {
        "record_index": index,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": total_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--adapter-output", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    arguments = parser.parse_args()
    if arguments.max_sequence_length != 1024:
        raise ValueError("go/no-go gate requires a 1024-token ceiling")
    for path in (arguments.dataset, arguments.model / "config.json"):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"gate input is absent or unsafe: {path}")
    for output in (arguments.adapter_output, arguments.receipt_output):
        if output.exists() or output.is_symlink():
            raise ValueError(f"immutable gate output already exists: {output}")

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    process_rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 4 or process_rank not in range(4):
        raise ValueError("file-PLE gate requires exactly four ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    config_path = arguments.model / "config.json"
    config_sha256 = _sha256(config_path)
    dataset_sha256 = _sha256(arguments.dataset)

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
    model.eval()
    load_seconds = time.perf_counter() - load_started

    rows_per_shard = 2_500_012
    probe_values = [
        0,
        1,
        rows_per_shard - 1,
        rows_per_shard,
        rows_per_shard + 1,
        17 * rows_per_shard - 1,
        17 * rows_per_shard,
        63 * rows_per_shard - 1,
        63 * rows_per_shard,
        64 * rows_per_shard,
        127 * rows_per_shard - 1,
        127 * rows_per_shard,
        320_001_535,
    ]
    probe_ids = torch.tensor([probe_values], dtype=torch.long, device=device)
    resident_ple = _ple_module(model)
    with torch.no_grad():
        resident_rows = resident_ple(probe_ids).detach().cpu()
        parity_input = torch.arange(1, 33, dtype=torch.long, device=device).unsqueeze(0)
        resident_logits = (
            model(input_ids=parity_input, use_cache=False)
            .logits[:, -1, :]
            .detach()
            .cpu()
        )
    del resident_ple
    gc.collect()
    torch.cuda.empty_cache()

    file_ple, memory = replace_ple_embedding(
        model,
        model_path=arguments.model,
    )
    with torch.no_grad():
        file_rows = file_ple(probe_ids).detach().cpu()
        file_logits = (
            model(input_ids=parity_input, use_cache=False)
            .logits[:, -1, :]
            .detach()
            .cpu()
        )
    row_parity_exact = torch.equal(resident_rows, file_rows)
    logit_delta = (resident_logits.float() - file_logits.float()).abs()
    logit_max_abs = float(logit_delta.max())
    logit_cosine = float(
        torch.nn.functional.cosine_similarity(
            resident_logits.float(),
            file_logits.float(),
            dim=-1,
        ).item()
    )
    logit_top1_equal = bool(
        resident_logits.argmax(dim=-1).item() == file_logits.argmax(dim=-1).item()
    )
    if not row_parity_exact:
        raise RuntimeError("file-backed PLE rows are not bit-identical")
    if logit_cosine < 0.999999 or not logit_top1_equal or logit_max_abs > 0.015625:
        raise RuntimeError(
            "file-backed PLE changed model logits: "
            f"cos={logit_cosine}, max_abs={logit_max_abs}, top1={logit_top1_equal}"
        )
    del resident_rows, file_rows, resident_logits, file_logits, parity_input, probe_ids
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(
        arguments.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    input_ids, labels, example = _real_training_example(
        tokenizer=tokenizer,
        dataset_path=arguments.dataset,
        max_length=arguments.max_sequence_length,
    )
    del tokenizer
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    torch.manual_seed(1788172800)
    torch.cuda.manual_seed_all(1788172800)
    modules = _inject_lora(model, rank=8, alpha=16)
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
            f"unexpected LoRA parameter count: {trainable_elements}"
        )
    initial_digest = _state_digest(_adapter_state(modules))
    if len(set(_all_rank_values(initial_digest, world_size))) != 1:
        raise RuntimeError("replicated adapters initialized differently")

    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    step_started = time.perf_counter()
    output = model(input_ids=input_ids, labels=labels, use_cache=False)
    if not bool(torch.isfinite(output.loss)):
        raise RuntimeError("real-data loss is not finite")
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
        raise RuntimeError("real-data optimizer step did not update the adapter")
    if len(set(_all_rank_values(updated_digest, world_size))) != 1:
        raise RuntimeError("real-data adapters diverged across ranks")

    adapter_sha256 = ""
    if process_rank == 0:
        arguments.adapter_output.mkdir(parents=True)
        adapter_path = arguments.adapter_output / "adapter_model.safetensors"
        config_output = arguments.adapter_output / "adapter_config.json"
        save_file(
            state,
            adapter_path,
            metadata={"format": "harness-file-ple-tp-lora-v1"},
        )
        config_output.write_text(
            json.dumps(
                {
                    "schema": "harness.file-ple-tp-lora.v1",
                    "base_model_config_sha256": config_sha256,
                    "rank": 8,
                    "alpha": 16,
                    "target_modules": sorted(TARGET_MODULES),
                    "module_count": len(modules),
                    "trainable_elements": trainable_elements,
                    "adapter_state_sha256": updated_digest,
                    "training_dataset_sha256": dataset_sha256,
                    "training_record": example,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        os.chmod(adapter_path, 0o444)
        os.chmod(config_output, 0o444)
        adapter_sha256 = _sha256(adapter_path)
    dist.barrier()
    adapter_sha256 = _all_rank_values(adapter_sha256, world_size)[0]
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    rank_result = {
        "rank": process_rank,
        "hostname": os.uname().nodename,
        "load_seconds": round(load_seconds, 3),
        "real_step_seconds": round(step_seconds, 3),
        "real_loss": float(output.loss.detach()),
        "peak_training_memory_gib": round(
            torch.cuda.max_memory_allocated() / 1024**3,
            3,
        ),
        "free_memory_gib": round(free_bytes / 1024**3, 3),
        "total_memory_gib": round(total_bytes / 1024**3, 3),
        "memory": memory,
        "row_parity_exact": row_parity_exact,
        "logit_cosine": logit_cosine,
        "logit_max_abs": logit_max_abs,
        "logit_top1_equal": logit_top1_equal,
        "adapter_state_sha256": updated_digest,
        "gid_index": os.environ.get("NCCL_IB_GID_INDEX", ""),
    }
    ranks = _all_rank_values(rank_result, world_size)
    if process_rank == 0:
        core: dict[str, Any] = {
            "schema": SCHEMA,
            "passed": True,
            "model_config_sha256": config_sha256,
            "dataset_sha256": dataset_sha256,
            "world_size": world_size,
            "example": example,
            "row_parity_exact": all(row["row_parity_exact"] for row in ranks),
            "minimum_logit_cosine": min(row["logit_cosine"] for row in ranks),
            "maximum_logit_abs_error": max(row["logit_max_abs"] for row in ranks),
            "logit_top1_equal": all(row["logit_top1_equal"] for row in ranks),
            "minimum_recovered_gib": min(
                row["memory"]["recovered_gib"] for row in ranks
            ),
            "maximum_peak_training_memory_gib": max(
                row["peak_training_memory_gib"] for row in ranks
            ),
            "minimum_free_memory_gib": min(row["free_memory_gib"] for row in ranks),
            "finite_gradients": True,
            "optimizer_step_passed": True,
            "adapters_identical": len(
                {row["adapter_state_sha256"] for row in ranks}
            )
            == 1,
            "module_count": len(modules),
            "adapter_tensor_count": len(state),
            "trainable_elements": trainable_elements,
            "adapter_artifact_sha256": adapter_sha256,
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
