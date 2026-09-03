#!/usr/bin/env python3
"""Evaluate the frozen DesignWins holdout with continuous vLLM batching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_designwins_text import (
    _record_identity,
    _write_json,
    extract_json,
    load_records,
    score_json,
    summarize_details,
    target_token_lengths,
)


def evaluate(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    records = load_records(args.dataset)[: args.max_samples]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )
    expected_lengths = target_token_lengths(tokenizer, records)
    required_budget = max(expected_lengths) + args.generation_slack_tokens
    if args.max_new_tokens < required_budget:
        raise ValueError(
            f"max_new_tokens={args.max_new_tokens} cannot cover the longest "
            f"held-out target; require at least {required_budget}"
        )
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": str(record["instruction"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for record in records
    ]
    budgets = [
        min(args.max_new_tokens, length + args.generation_slack_tokens)
        for length in expected_lengths
    ]
    sampling = [
        SamplingParams(temperature=0, max_tokens=budget) for budget in budgets
    ]
    model = LLM(
        model=str(args.model),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.cutoff_len + args.max_new_tokens,
        max_num_seqs=args.batch_size,
        enable_lora=args.adapter is not None,
        max_lora_rank=args.max_lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    lora_request = (
        LoRARequest("designwins", 1, str(args.adapter))
        if args.adapter is not None
        else None
    )
    generated = model.generate(
        prompts,
        sampling,
        lora_request=lora_request,
        use_tqdm=True,
    )

    details: list[dict] = []
    for index, result in enumerate(generated):
        record = records[index]
        part, family = _record_identity(record, index)
        expected = json.loads(str(record["output"]))
        output = result.outputs[0]
        response = output.text
        predicted = extract_json(response)
        generated_tokens = len(output.token_ids)
        detail = {
            "index": index,
            "part": part,
            "family": family,
            "expected_tokens": expected_lengths[index],
            "generated_tokens": generated_tokens,
            "generation_budget": budgets[index],
            "hit_generation_limit": output.finish_reason == "length",
            "score": score_json(expected, predicted),
            "response": response,
        }
        details.append(detail)
        print(
            json.dumps(
                {
                    "index": index,
                    "valid_json": detail["score"]["valid_json"],
                    "leaf_f1": detail["score"]["leaf_f1"],
                    "generated_tokens": generated_tokens,
                    "generation_budget": budgets[index],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return {
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "dataset": str(args.dataset),
        "summary": summarize_details(details),
        "details": details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=141)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--generation-slack-tokens", type=int, default=256)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    if (
        args.max_samples < 1
        or args.cutoff_len < 256
        or args.max_new_tokens < 32
        or args.batch_size < 1
        or args.generation_slack_tokens < 32
        or args.max_lora_rank < 1
        or not 0 < args.gpu_memory_utilization < 1
    ):
        parser.error("invalid evaluation limits")
    return args


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    _write_json(args.output, result)
    print(json.dumps(result["summary"], sort_keys=True))
    print("DESIGNWINS_VLLM_EVALUATION_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
