#!/usr/bin/env python3
"""Score every held-out DesignWins response under teacher forcing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from evaluate_designwins_text import (
    _record_identity,
    load_records,
    target_token_lengths,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fingerprint(details: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        details,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _summarize(details: list[dict[str, Any]], elapsed_seconds: float) -> dict:
    token_count = sum(int(row["scored_tokens"]) for row in details)
    total_nll = sum(float(row["total_nll"]) for row in details)
    correct_tokens = sum(int(row["correct_tokens"]) for row in details)
    mean_nll = total_nll / token_count
    return {
        "samples": len(details),
        "scored_tokens": token_count,
        "total_nll": total_nll,
        "mean_token_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 50)),
        "token_accuracy": correct_tokens / token_count,
        "elapsed_seconds": elapsed_seconds,
        "fingerprint": _fingerprint(details),
    }


def evaluate_pass(
    *,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    cutoff_len: int,
    max_response_tokens: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    details: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, record in enumerate(records):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(record["instruction"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=True,
            max_length=cutoff_len,
        )["input_ids"]
        target_ids = tokenizer(
            str(record["output"]),
            add_special_tokens=False,
        )["input_ids"]
        if len(target_ids) > max_response_tokens:
            raise ValueError(
                f"record {index} target has {len(target_ids)} tokens; "
                f"limit is {max_response_tokens}"
            )
        target_ids = [*target_ids, tokenizer.eos_token_id]
        input_ids = torch.tensor(
            [prompt_ids + target_ids],
            dtype=torch.long,
            device=model.device,
        )
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor(
            [[-100] * len(prompt_ids) + target_ids],
            dtype=torch.long,
            device=model.device,
        )
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, :-1, :].float()
        shifted_labels = labels[:, 1:]
        response_start = len(prompt_ids) - 1
        selected_logits = logits[:, response_start:, :].reshape(
            -1, logits.shape[-1]
        )
        selected_labels = shifted_labels[:, response_start:].reshape(-1)
        total_nll = functional.cross_entropy(
            selected_logits,
            selected_labels,
            reduction="sum",
        ).item()
        correct_tokens = int(
            selected_logits.argmax(dim=-1).eq(selected_labels).sum().item()
        )
        part, family = _record_identity(record, index)
        detail = {
            "index": index,
            "part": part,
            "family": family,
            "prompt_tokens": len(prompt_ids),
            "scored_tokens": len(target_ids),
            "total_nll": total_nll,
            "mean_token_nll": total_nll / len(target_ids),
            "correct_tokens": correct_tokens,
            "token_accuracy": correct_tokens / len(target_ids),
        }
        details.append(detail)
        print(
            json.dumps(
                {
                    "index": index,
                    "mean_token_nll": detail["mean_token_nll"],
                    "token_accuracy": detail["token_accuracy"],
                    "scored_tokens": detail["scored_tokens"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del input_ids, attention_mask, labels, logits
        del shifted_labels, selected_logits, selected_labels
    elapsed = time.monotonic() - started
    return {
        "summary": _summarize(details, elapsed),
        "details": details,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records = load_records(args.dataset)[: args.max_samples]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )
    longest_target = max(target_token_lengths(tokenizer, records))
    if longest_target > args.max_response_tokens:
        raise ValueError(
            f"longest target has {longest_target} tokens; "
            f"limit is {args.max_response_tokens}"
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    if args.adapter is not None:
        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            is_trainable=False,
        )
    model.eval()
    passes = [
        evaluate_pass(
            model=model,
            tokenizer=tokenizer,
            records=records,
            cutoff_len=args.cutoff_len,
            max_response_tokens=args.max_response_tokens,
        )
        for _ in range(args.passes)
    ]
    return {
        "schema": "harness.designwins-teacher-forced-evaluation.v1",
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "dataset": str(args.dataset),
        "max_samples": args.max_samples,
        "cutoff_len": args.cutoff_len,
        "max_response_tokens": args.max_response_tokens,
        "passes": passes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=141)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--max-response-tokens", type=int, default=8192)
    parser.add_argument("--passes", type=int, default=1)
    args = parser.parse_args()
    if (
        args.max_samples < 1
        or args.cutoff_len < 256
        or args.max_response_tokens < 32
        or args.passes not in (1, 2)
    ):
        parser.error("invalid evaluation limits")
    return args


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    _write_json(args.output, result)
    print(
        json.dumps([row["summary"] for row in result["passes"]], sort_keys=True)
    )
    print("DESIGNWINS_TEACHER_FORCED_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
