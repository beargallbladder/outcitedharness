#!/usr/bin/env python3
"""Evaluate base or LoRA-adapted MCU JSON extraction on a held-out split."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def extract_json(text: str) -> Any | None:
    cleaned = text
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def _leaves(value: Any, path: str = "$") -> set[tuple[str, str]]:
    if isinstance(value, dict):
        output: set[tuple[str, str]] = set()
        for key, child in value.items():
            output.update(_leaves(child, f"{path}.{key}"))
        return output
    if isinstance(value, list):
        output = set()
        for index, child in enumerate(value):
            output.update(_leaves(child, f"{path}[{index}]"))
        return output
    return {(path, json.dumps(value, ensure_ascii=False, sort_keys=True))}


def score_json(expected: Any, predicted: Any | None) -> dict[str, float | bool]:
    if predicted is None:
        return {
            "valid_json": False,
            "exact": False,
            "leaf_precision": 0.0,
            "leaf_recall": 0.0,
            "leaf_f1": 0.0,
        }
    expected_leaves = _leaves(expected)
    predicted_leaves = _leaves(predicted)
    intersection = len(expected_leaves & predicted_leaves)
    precision = intersection / len(predicted_leaves) if predicted_leaves else 0.0
    recall = intersection / len(expected_leaves) if expected_leaves else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "valid_json": True,
        "exact": predicted == expected,
        "leaf_precision": precision,
        "leaf_recall": recall,
        "leaf_f1": f1,
    }


def target_token_lengths(tokenizer: Any, records: list[dict[str, Any]]) -> list[int]:
    return [
        len(
            tokenizer(
                str(record["output"]),
                add_special_tokens=False,
            )["input_ids"]
        )
        for record in records
    ]


def _write_json(path: Path, value: Any) -> None:
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
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records = json.loads(args.dataset.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("evaluation dataset must be a non-empty JSON list")
    records = records[: args.max_samples]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )
    expected_lengths = target_token_lengths(tokenizer, records)
    required_generation_budget = max(expected_lengths) + 32
    if args.max_new_tokens < required_generation_budget:
        raise ValueError(
            f"max_new_tokens={args.max_new_tokens} cannot cover the longest "
            f"held-out target; require at least {required_generation_budget}"
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    details: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        instruction = str(record["instruction"])
        expected = json.loads(str(record["output"]))
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=args.cutoff_len,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        generated_tokens = int(
            generated.shape[1] - inputs["input_ids"].shape[1]
        )
        predicted = extract_json(response)
        details.append(
            {
                "index": index,
                "expected_tokens": expected_lengths[index],
                "generated_tokens": generated_tokens,
                "hit_generation_limit": generated_tokens >= args.max_new_tokens,
                "score": score_json(expected, predicted),
                "response": response,
            }
        )
        print(
            json.dumps(
                {
                    "index": index,
                    "valid_json": details[-1]["score"]["valid_json"],
                    "leaf_f1": details[-1]["score"]["leaf_f1"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    score_names = ("leaf_precision", "leaf_recall", "leaf_f1")
    summary: dict[str, Any] = {
        "samples": len(details),
        "valid_json_rate": sum(row["score"]["valid_json"] for row in details)
        / len(details),
        "generation_limit_hits": sum(
            row["hit_generation_limit"] for row in details
        ),
        "exact_rate": sum(row["score"]["exact"] for row in details) / len(details),
        **{
            f"mean_{name}": sum(float(row["score"][name]) for row in details)
            / len(details)
            for name in score_names
        },
    }
    return {
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "dataset": str(args.dataset),
        "summary": summary,
        "details": details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.max_samples < 1 or args.cutoff_len < 256 or args.max_new_tokens < 32:
        parser.error("sample and token limits are too small")
    return args


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    _write_json(args.output, result)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
