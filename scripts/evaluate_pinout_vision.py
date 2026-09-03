#!/usr/bin/env python3
"""Freeze and evaluate a PDF-lineage-safe Qwen3-VL pinout vision cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


COHORT_SCHEMA = "harness.pinout-vision-eval-cohort.v1"
EVALUATION_SCHEMA = "harness.pinout-vision-evaluation.v1"
DATASET_RELATIVE = Path("datasets/pinout-vision-row-v1")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, kind: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{kind} must be a regular file: {path}")
    return path


def _json_object(path: Path, kind: str) -> dict[str, Any]:
    value = json.loads(_regular(path, kind).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{kind} must contain a JSON object")
    return value


def _verify_evidence(value: dict[str, Any], schema: str, kind: str) -> None:
    if value.get("schema") != schema:
        raise ValueError(f"{kind} schema is not supported")
    expected = value.get("evidence_sha256")
    core = {
        key: item
        for key, item in value.items()
        if key not in ("created_at", "evidence_sha256")
    }
    if hashlib.sha256(_canonical(core)).hexdigest() != expected:
        raise ValueError(f"{kind} evidence digest is invalid")


def _records(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(_regular(path, "canonical split").read_text().splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"canonical row {line_number} is malformed")
        yield value


def _family_key(record_id: str) -> str:
    """Collapse known automotive/package suffixes without guessing MCU series."""

    return re.sub(
        r"(?i)(?:-?q1|-(?:tr|reel|tape))$",
        "",
        record_id.strip(),
    ).casefold()


def write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_cohort(
    *,
    root: Path,
    examples_per_lineage: int,
    dataset_root: Path | None = None,
) -> dict[str, Any]:
    if examples_per_lineage < 1:
        raise ValueError("examples per lineage must be positive")
    root = root.resolve(strict=True)
    dataset_root = (
        dataset_root.resolve(strict=True)
        if dataset_root is not None
        else root / DATASET_RELATIVE
    )
    manifest_path = dataset_root / "manifest.json"
    manifest = _json_object(manifest_path, "dataset manifest")
    _verify_evidence(
        manifest,
        "harness.pinout-vision-row-dataset.v1",
        "dataset manifest",
    )
    if manifest.get("authorization", {}).get("training_authorized") is not True:
        raise ValueError("dataset is not authorized")
    train_path = dataset_root / "canonical" / "train.jsonl"
    test_path = dataset_root / "canonical" / "test.jsonl"
    for split, path in (("train", train_path), ("test", test_path)):
        artifact = manifest["artifacts"][f"canonical/{split}.jsonl"]
        if (
            _sha256(path) != artifact["sha256"]
            or path.stat().st_size != artifact["bytes"]
        ):
            raise ValueError(
                f"canonical {split} split differs from dataset manifest"
            )

    train_families = {
        _family_key(str(row["record_id"]))
        for row in _records(train_path)
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _records(test_path):
        if row.get("split") != "test" or not isinstance(row.get("lineage_id"), str):
            raise ValueError("canonical test split contains an invalid row")
        grouped[row["lineage_id"]].append(row)
    selected: list[dict[str, Any]] = []
    excluded_lineages: dict[str, list[str]] = {}
    for lineage, rows in sorted(grouped.items()):
        rows = [
            row
            for row in rows
            if _family_key(str(row["record_id"])) not in train_families
        ]
        if not rows:
            excluded_lineages[lineage] = sorted(
                {
                    _family_key(str(row["record_id"]))
                    for row in grouped[lineage]
                }
            )
            continue
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"pinout-frozen-v1:{row['example_id']}".encode()
            ).hexdigest(),
        )
        for row in ranked[:examples_per_lineage]:
            images = []
            for relative, expected_sha256 in zip(
                row["images"],
                row["image_sha256"],
                strict=True,
            ):
                path = _regular(dataset_root / relative, "cohort image")
                if _sha256(path) != expected_sha256:
                    raise ValueError(f"cohort image hash mismatch: {relative}")
                images.append(
                    {
                        "path": relative,
                        "sha256": expected_sha256,
                        "bytes": path.stat().st_size,
                    }
                )
            selected.append(
                {
                    "example_id": row["example_id"],
                    "lineage_id": lineage,
                    "record_id": row["record_id"],
                    "vendor": str(row["record_id"]).split("_", 1)[0].casefold(),
                    "family_key": _family_key(str(row["record_id"])),
                    "prompt": row["prompt"],
                    "reference": row["response"],
                    "images": images,
                }
            )
    selected_families = {row["family_key"] for row in selected}
    if selected_families & train_families:
        raise RuntimeError("frozen cohort contains a training family")
    core: dict[str, Any] = {
        "schema": COHORT_SCHEMA,
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "evidence_sha256": manifest["evidence_sha256"],
        },
        "selection": {
            "split": "test",
            "lineage_key": "source_pdf_sha256",
            "examples_per_lineage": examples_per_lineage,
            "test_lineages": len(grouped),
            "lineages": len(grouped) - len(excluded_lineages),
            "examples": len(selected),
            "excluded_train_family_lineages": excluded_lineages,
            "family_key": "record_id with automotive/package suffix removed",
            "family_overlap_check_passed": True,
            "algorithm": (
                "exclude train family keys, then minimum "
                "sha256(pinout-frozen-v1:example_id) per PDF lineage"
            ),
        },
        "examples": sorted(selected, key=lambda row: row["example_id"]),
    }
    core["evidence_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **core,
    }


def _normalize_number(value: Any) -> str:
    return re.sub(r"\s+", "", str("" if value is None else value).upper())


def _normalize_name(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str("" if value is None else value).upper())


def _pin_map(value: Any) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("pins"), list):
        raise ValueError("response has no pins list")
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in value["pins"]:
        if not isinstance(row, dict):
            raise ValueError("response pin is not an object")
        identity = (
            _normalize_number(row.get("pin_no")),
            _normalize_name(row.get("name")),
        )
        if not all(identity) or identity in output:
            raise ValueError("response pin identity is invalid or duplicated")
        output[identity] = row
    return output


def _parse_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    decoder = json.JSONDecoder()
    start = stripped.find("{")
    if start < 0:
        raise ValueError("model output has no JSON object")
    value, _end = decoder.raw_decode(stripped[start:])
    if not isinstance(value, dict):
        raise ValueError("model output JSON is not an object")
    return value


def _set_metric(
    predicted: set[tuple[str, str]],
    reference: set[tuple[str, str]],
) -> dict[str, Any]:
    true_positive = len(predicted & reference)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(reference) if reference else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive": true_positive,
        "predicted": len(predicted),
        "reference": len(reference),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": predicted == reference,
    }


def _rich_metric(
    predicted: Mapping[tuple[str, str], dict[str, Any]],
    reference: Mapping[tuple[str, str], dict[str, Any]],
) -> dict[str, int]:
    output = {
        "matched_identities": 0,
        "type_correct": 0,
        "direction_correct": 0,
        "functions_exact": 0,
    }
    for identity in predicted.keys() & reference.keys():
        actual = predicted[identity]
        expected = reference[identity]
        output["matched_identities"] += 1
        output["type_correct"] += str(actual.get("type") or "").casefold() == str(
            expected.get("type") or ""
        ).casefold()
        output["direction_correct"] += str(actual.get("dir") or "").casefold() == str(
            expected.get("dir") or ""
        ).casefold()
        actual_functions = {
            re.sub(r"\s+", " ", str(item).casefold()).strip()
            for item in actual.get("functions") or []
        }
        expected_functions = {
            re.sub(r"\s+", " ", str(item).casefold()).strip()
            for item in expected.get("functions") or []
        }
        output["functions_exact"] += actual_functions == expected_functions
    return output


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_positive = sum(row["identity"]["true_positive"] for row in rows)
    predicted = sum(row["identity"]["predicted"] for row in rows)
    reference = sum(row["identity"]["reference"] for row in rows)
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / reference if reference else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    rich = {
        key: sum(row["rich"][key] for row in rows)
        for key in (
            "matched_identities",
            "type_correct",
            "direction_correct",
            "functions_exact",
        )
    }
    matched = rich["matched_identities"]
    return {
        "examples": len(rows),
        "json_valid": sum(row["json_valid"] for row in rows),
        "identity_exact": sum(row["identity"]["exact"] for row in rows),
        "identity": {
            "true_positive": true_positive,
            "predicted": predicted,
            "reference": reference,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "rich": {
            **rich,
            "type_accuracy": rich["type_correct"] / matched if matched else 0.0,
            "direction_accuracy": (
                rich["direction_correct"] / matched if matched else 0.0
            ),
            "functions_exact_rate": (
                rich["functions_exact"] / matched if matched else 0.0
            ),
        },
    }


def evaluate(
    *,
    root: Path,
    cohort_path: Path,
    model_path: Path,
    adapter_path: Path | None,
    maximum_new_tokens: int,
    limit: int | None,
    dataset_root: Path | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    cohort = _json_object(cohort_path, "frozen cohort")
    _verify_evidence(cohort, COHORT_SCHEMA, "frozen cohort")
    dataset_root = (
        dataset_root.resolve(strict=True)
        if dataset_root is not None
        else root / DATASET_RELATIVE
    )
    if _sha256(dataset_root / "manifest.json") != cohort["dataset_manifest"]["sha256"]:
        raise ValueError("dataset manifest differs from frozen cohort")
    examples = cohort["examples"]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        examples = examples[:limit]
    if not examples:
        raise ValueError("evaluation cohort is empty")
    if not 64 <= maximum_new_tokens <= 2048:
        raise ValueError("maximum new tokens must be between 64 and 2048")
    if not 1 <= batch_size <= 32:
        raise ValueError("batch size must be between 1 and 32")

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_path = model_path.resolve(strict=True)
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=True,
    )
    adapter: dict[str, Any] | None = None
    if adapter_path is not None:
        from peft import PeftModel

        adapter_path = adapter_path.resolve(strict=True)
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
        )
        adapter_model = _regular(
            adapter_path / "adapter_model.safetensors",
            "adapter model",
        )
        adapter = {
            "path": str(adapter_path),
            "sha256": _sha256(adapter_model),
            "bytes": adapter_model.stat().st_size,
        }
    model.eval()
    torch.manual_seed(1788172800)
    torch.cuda.manual_seed_all(1788172800)
    torch.cuda.reset_peak_memory_stats()

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch_start in range(0, len(examples), batch_size):
        batch = examples[batch_start : batch_start + batch_size]
        opened_by_example = [
            [
                Image.open(
                    _regular(
                        dataset_root / image["path"],
                        "evaluation image",
                    )
                ).convert("RGB")
                for image in example["images"]
            ]
            for example in batch
        ]
        try:
            texts = [
                processor.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [
                                *[
                                    {"type": "image"}
                                    for _image in opened
                                ],
                                {
                                    "type": "text",
                                    "text": example["prompt"],
                                },
                            ],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for example, opened in zip(
                    batch,
                    opened_by_example,
                    strict=True,
                )
            ]
            inputs = processor(
                text=texts,
                images=[
                    image
                    for opened in opened_by_example
                    for image in opened
                ],
                padding=True,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(model.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=maximum_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            prompt_length = int(inputs["input_ids"].shape[1])
            output_text = processor.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        finally:
            for opened in opened_by_example:
                for image in opened:
                    image.close()

        for example, text_output in zip(batch, output_text, strict=True):
            reference_value = json.loads(example["reference"])
            reference_pins = _pin_map(reference_value)
            json_valid = True
            error = None
            try:
                predicted_value = _parse_response(text_output)
                predicted_pins = _pin_map(predicted_value)
            except (ValueError, json.JSONDecodeError) as caught:
                json_valid = False
                error = str(caught)
                predicted_pins = {}
            identity = _set_metric(set(predicted_pins), set(reference_pins))
            rich = _rich_metric(predicted_pins, reference_pins)
            results.append(
                {
                    "example_id": example["example_id"],
                    "lineage_id": example["lineage_id"],
                    "record_id": example["record_id"],
                    "vendor": example["vendor"],
                    "family_key": example["family_key"],
                    "json_valid": json_valid,
                    "error": error,
                    "identity": identity,
                    "rich": rich,
                    "output": text_output,
                }
            )
        batch_end = batch_start + len(batch)
        print(
            f"evaluated {batch_end}/{len(examples)} "
            f"batch_size={len(batch)}",
            flush=True,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    by_vendor = {
        vendor: _aggregate([row for row in results if row["vendor"] == vendor])
        for vendor in sorted({row["vendor"] for row in results})
    }
    core: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "cohort": {
            "path": str(cohort_path.resolve(strict=True)),
            "sha256": _sha256(cohort_path),
            "evidence_sha256": cohort["evidence_sha256"],
            "limited": limit is not None,
        },
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
            "adapter": adapter,
        },
        "generation": {
            "batch_size": batch_size,
            "do_sample": False,
            "maximum_new_tokens": maximum_new_tokens,
            "seed": 1788172800,
        },
        "runtime": {
            "seconds": round(elapsed, 3),
            "examples_per_second": len(results) / elapsed,
            "peak_cuda_memory_gib": round(
                torch.cuda.max_memory_allocated() / 1024**3,
                3,
            ),
        },
        "aggregate": _aggregate(results),
        "by_vendor": by_vendor,
        "results": results,
    }
    if not all(math.isfinite(value) for value in (
        core["runtime"]["seconds"],
        core["runtime"]["examples_per_second"],
        core["aggregate"]["identity"]["f1"],
    )):
        raise RuntimeError("evaluation produced a non-finite metric")
    core["evidence_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **core,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--root", required=True, type=Path)
    freeze.add_argument("--dataset-root", type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--examples-per-lineage", type=int, default=1)
    run = subparsers.add_parser("evaluate")
    run.add_argument("--root", required=True, type=Path)
    run.add_argument("--dataset-root", type=Path)
    run.add_argument("--cohort", required=True, type=Path)
    run.add_argument("--model", required=True, type=Path)
    run.add_argument("--adapter", type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--maximum-new-tokens", type=int, default=768)
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.command == "freeze":
        result = freeze_cohort(
            root=arguments.root,
            examples_per_lineage=arguments.examples_per_lineage,
            dataset_root=arguments.dataset_root,
        )
    else:
        result = evaluate(
            root=arguments.root,
            cohort_path=arguments.cohort,
            model_path=arguments.model,
            adapter_path=arguments.adapter,
            maximum_new_tokens=arguments.maximum_new_tokens,
            limit=arguments.limit,
            dataset_root=arguments.dataset_root,
            batch_size=arguments.batch_size,
        )
    write_new(arguments.output, result)
    if arguments.command == "evaluate":
        print(json.dumps(result["aggregate"], sort_keys=True))
    else:
        print(json.dumps(result["selection"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
