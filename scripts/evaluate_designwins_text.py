#!/usr/bin/env python3
"""Evaluate base or LoRA-adapted MCU JSON extraction on a held-out split."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load either LlamaFactory JSON or canonical lineage-bearing JSONL."""

    if path.suffix == ".jsonl":
        raw_records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raw_records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("evaluation dataset must be a non-empty JSON list or JSONL")

    records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise ValueError(f"evaluation record {index} must be an object")
        instruction = raw.get("instruction", raw.get("prompt"))
        output = raw.get("output", raw.get("response"))
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"evaluation record {index} lacks an instruction")
        if not isinstance(output, str) or not output.strip():
            raise ValueError(f"evaluation record {index} lacks an output")
        metadata = raw.get("metadata")
        records.append(
            {
                "instruction": instruction,
                "output": output,
                "metadata": dict(metadata) if isinstance(metadata, dict) else {},
            }
        )
    return records


def _record_identity(record: dict[str, Any], index: int) -> tuple[str, str]:
    metadata = record.get("metadata")
    part = (
        str(metadata.get("part") or "").strip()
        if isinstance(metadata, dict)
        else ""
    )
    if not part:
        part = f"record-{index:04d}"
    family = part.split("_", 1)[0].casefold()
    return part, family


def extract_json(text: str) -> Any | None:
    cleaned = text
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    starts = [
        index
        for character in ("[", "{")
        if (index := cleaned.find(character)) >= 0
    ]
    if not starts:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(cleaned[min(starts) :])
    except json.JSONDecodeError:
        return None
    return value


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
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize_details(details: list[dict[str, Any]]) -> dict[str, Any]:
    if not details:
        raise ValueError("cannot summarize an empty evaluation")
    score_names = ("leaf_precision", "leaf_recall", "leaf_f1")

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "samples": len(rows),
            "valid_json_rate": sum(
                bool(row["score"]["valid_json"]) for row in rows
            )
            / len(rows),
            "generation_limit_hits": sum(
                bool(row["hit_generation_limit"]) for row in rows
            ),
            "exact_rate": sum(bool(row["score"]["exact"]) for row in rows)
            / len(rows),
            **{
                f"mean_{name}": sum(float(row["score"][name]) for row in rows)
                / len(rows)
                for name in score_names
            },
        }

    families: dict[str, list[dict[str, Any]]] = {}
    for row in details:
        families.setdefault(str(row["family"]), []).append(row)
    return {
        **aggregate(details),
        "by_family": {
            family: aggregate(rows) for family, rows in sorted(families.items())
        },
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    class JsonCompletionCriteria(StoppingCriteria):
        def __init__(self, prompt_width: int, batch_size: int):
            self.prompt_width = prompt_width
            self.done = [False] * batch_size
            self.calls = 0

        def __call__(self, input_ids, _scores, **_kwargs):
            self.calls += 1
            if self.calls % 16 == 0:
                for row in range(len(self.done)):
                    if self.done[row]:
                        continue
                    text = tokenizer.decode(
                        input_ids[row, self.prompt_width :],
                        skip_special_tokens=True,
                    )
                    self.done[row] = extract_json(text) is not None
            return torch.tensor(
                self.done,
                dtype=torch.bool,
                device=input_ids.device,
            ).unsqueeze(1)

    records = load_records(args.dataset)
    records = records[: args.max_samples]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )
    expected_lengths = target_token_lengths(tokenizer, records)
    required_generation_budget = max(expected_lengths) + args.generation_slack_tokens
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
    tokenizer.padding_side = "left"

    details: list[dict[str, Any]] = []
    ordered_indices = sorted(
        range(len(records)),
        key=lambda index: (expected_lengths[index], index),
    )
    batches = [
        ordered_indices[offset : offset + args.batch_size]
        for offset in range(0, len(ordered_indices), args.batch_size)
    ]
    for batch_indices in batches:
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": str(records[index]["instruction"])}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for index in batch_indices
        ]
        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.cutoff_len,
        ).to(model.device)
        prompt_width = int(inputs["input_ids"].shape[1])
        generation_budget = min(
            args.max_new_tokens,
            max(expected_lengths[index] for index in batch_indices)
            + args.generation_slack_tokens,
        )
        stopping = JsonCompletionCriteria(prompt_width, len(batch_indices))
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=generation_budget,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([stopping]),
            )
        for row, index in enumerate(batch_indices):
            record = records[index]
            part, family = _record_identity(record, index)
            expected = json.loads(str(record["output"]))
            token_ids = generated[row, prompt_width:].tolist()
            eos_position = (
                token_ids.index(tokenizer.eos_token_id)
                if tokenizer.eos_token_id in token_ids
                else None
            )
            generated_tokens = (
                eos_position + 1 if eos_position is not None else len(token_ids)
            )
            response = tokenizer.decode(
                token_ids[:generated_tokens],
                skip_special_tokens=True,
            )
            predicted = extract_json(response)
            detail = {
                "index": index,
                "part": part,
                "family": family,
                "expected_tokens": expected_lengths[index],
                "generated_tokens": generated_tokens,
                "generation_budget": generation_budget,
                "hit_generation_limit": (
                    generated_tokens >= generation_budget
                    and eos_position is None
                    and predicted is None
                ),
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
                        "generation_budget": generation_budget,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    details.sort(key=lambda row: int(row["index"]))

    summary = summarize_details(details)
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--generation-slack-tokens", type=int, default=256)
    args = parser.parse_args()
    if (
        args.max_samples < 1
        or args.cutoff_len < 256
        or args.max_new_tokens < 32
        or args.batch_size < 1
        or args.generation_slack_tokens < 32
    ):
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
