#!/usr/bin/env python3
"""Fail closed on Qwen3.8 training unless its 95 GiB PLE stays sharded."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "harness.qwen38-training-preflight.v1"
TP_RECEIPT_SCHEMA = "harness.qwen38-tp-lora-smoke.v1"
FILE_RECEIPT_SCHEMA = "harness.qwen38-file-ple-lora-smoke.v1"
STRATEGIES = (
    "vanilla-zero3",
    "native-tp-load",
    "native-tp-lora",
    "file-backed-ple-lora",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    value = start
    for _ in range(count):
        value += 1
        while not _is_prime(value):
            value += 1
    return value


def ple_geometry(config: dict[str, Any], world_size: int) -> dict[str, Any]:
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("Qwen3.8 text_config must be an object")
    ple_layers = text.get("ple_layer_ids")
    if not isinstance(ple_layers, list) or not ple_layers:
        raise ValueError("Qwen3.8 config does not declare a PLE layer")
    ngram_size = int(text["ngram_size"])
    heads_per_ngram = int(text["heads_per_ngram"])
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    embedding_dim = int(text["ple_embed_dim"])
    if ngram_heads <= 0 or embedding_dim % ngram_heads:
        raise ValueError("invalid PLE head geometry")
    row_width = embedding_dim // ngram_heads
    base = int(text["ngram_vocab_size_base"])
    divisor = int(text["make_ngram_vocab_size_divisible_by"])
    if divisor <= 0:
        raise ValueError("invalid PLE vocabulary divisor")

    rows_per_table: list[int] = []
    for ple_index, _layer_id in enumerate(ple_layers):
        total_rows = sum(
            _nth_prime_after(
                base - 1,
                ple_index * ngram_heads + head_index + 1,
            )
            for head_index in range(ngram_heads)
        )
        rows_per_table.append(math.ceil(total_rows / divisor) * divisor)
    full_bytes = sum(rows_per_table) * row_width * 2
    return {
        "layers": len(ple_layers),
        "layer_ids": ple_layers,
        "ngram_heads": ngram_heads,
        "rows_per_table": rows_per_table,
        "row_width": row_width,
        "checkpoint_parts": int(text["split_ngram_parts"]),
        "full_bf16_bytes": full_bytes,
        "full_bf16_gib": round(full_bytes / 1024**3, 6),
        "ideal_per_rank_gib": round(full_bytes / world_size / 1024**3, 6),
        "column_shard_width": (
            row_width // world_size if row_width % world_size == 0 else None
        ),
    }


def _load_receipt(
    path: Path | None,
    *,
    expected_schema: str,
    config_sha256: str,
    world_size: int,
    node_memory_gib: float,
) -> tuple[dict[str, Any] | None, list[str]]:
    if path is None:
        return None, ["a mechanically verified LoRA compatibility receipt is absent"]
    if path.is_symlink() or not path.is_file():
        return None, ["the LoRA compatibility receipt is not a regular file"]
    receipt = json.loads(path.read_text())
    if not isinstance(receipt, dict):
        return None, ["the LoRA compatibility receipt is not an object"]
    claimed_digest = receipt.get("evidence_sha256")
    core = {key: value for key, value in receipt.items() if key != "evidence_sha256"}
    actual_digest = hashlib.sha256(_canonical(core)).hexdigest()
    blockers: list[str] = []
    if claimed_digest != actual_digest:
        blockers.append("the LoRA compatibility receipt digest is invalid")
    if receipt.get("schema") != expected_schema:
        blockers.append("the LoRA compatibility receipt schema is wrong")
    if receipt.get("model_config_sha256") != config_sha256:
        blockers.append("the receipt covers a different model config")
    if receipt.get("world_size") != world_size:
        blockers.append("the receipt covers a different world size")
    for field in (
        "ple_sharded",
        "load_passed",
        "optimizer_step_passed",
        "finite_gradients",
        "adapter_save_passed",
    ):
        if receipt.get(field) is not True:
            blockers.append(f"the receipt does not prove {field}")
    peak = receipt.get("max_peak_memory_gib")
    if (
        isinstance(peak, bool)
        or not isinstance(peak, int | float)
        or peak <= 0
        or peak >= node_memory_gib
    ):
        blockers.append("the receipt does not prove safe per-rank peak memory")
    return receipt, blockers


def verify(
    *,
    config_path: Path,
    strategy: str,
    world_size: int,
    node_memory_gib: float,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown Qwen3.8 strategy: {strategy}")
    if world_size < 2 or world_size > 8:
        raise ValueError("world size must be between 2 and 8")
    if node_memory_gib <= 0:
        raise ValueError("node memory must be positive")
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("model config must be a regular file")
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict) or config.get("model_type") != "qwen4_exp":
        raise ValueError("model config is not Qwen3.8 qwen4_exp")
    config_sha256 = _sha256(config_path)
    ple = ple_geometry(config, world_size)
    blockers: list[str] = []
    receipt: dict[str, Any] | None = None

    if strategy == "vanilla-zero3":
        blockers.append(
            "vanilla ZeRO-3 gathers the full PLE embedding for lookup and "
            "exceeds a 128 GiB node with the resident model shard"
        )
    elif world_size != 4:
        blockers.append("the qualified Qwen3.8 topology requires exactly four ranks")

    if ple["column_shard_width"] is None and strategy.startswith("native-tp"):
        blockers.append("PLE width is not divisible by the tensor-parallel world size")
    if (
        strategy.startswith("native-tp")
        and ple["ideal_per_rank_gib"] >= node_memory_gib
    ):
        blockers.append("the tensor-parallel PLE shard alone exceeds node memory")

    if strategy == "native-tp-load":
        scope = "load-and-forward-probe-only"
    else:
        scope = "training"

    if strategy in {"native-tp-lora", "file-backed-ple-lora"}:
        expected_schema = (
            TP_RECEIPT_SCHEMA
            if strategy == "native-tp-lora"
            else FILE_RECEIPT_SCHEMA
        )
        receipt, receipt_blockers = _load_receipt(
            receipt_path,
            expected_schema=expected_schema,
            config_sha256=config_sha256,
            world_size=world_size,
            node_memory_gib=node_memory_gib,
        )
        blockers.extend(receipt_blockers)

    core: dict[str, Any] = {
        "schema": SCHEMA,
        "passed": not blockers,
        "scope": scope,
        "strategy": strategy,
        "world_size": world_size,
        "node_memory_gib": node_memory_gib,
        "model_type": config["model_type"],
        "model_config_sha256": config_sha256,
        "ple": ple,
        "blockers": blockers,
        "receipt_sha256": _sha256(receipt_path) if receipt is not None else None,
    }
    core["evidence_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    return core


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("preflight output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o444)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--node-memory-gib", type=float, default=128.0)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = verify(
        config_path=arguments.config,
        strategy=arguments.strategy,
        world_size=arguments.world_size,
        node_memory_gib=arguments.node_memory_gib,
        receipt_path=arguments.receipt,
    )
    if arguments.output is not None:
        _write_once(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
