#!/usr/bin/env python3
"""Build tokenizer-verified, source-grounded DesignWins pin-table chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


MARKER = "\n\nPIN TABLE TEXT:\n"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"record {line_number} is not an object")
            rows.append(row)
    return rows


def _pin_number_text(pin: dict[str, Any]) -> tuple[str, ...]:
    value = pin.get("pin_no")
    if isinstance(value, list):
        return tuple(str(item).casefold() for item in value)
    if value is None:
        return ()
    return (str(value).casefold(),)


def _anchor(
    source: str,
    pin: dict[str, Any],
    *,
    used_positions: set[int],
) -> int | None:
    name = pin.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    lowered = source.casefold()
    needle = name.strip().casefold()
    positions: list[int] = []
    offset = 0
    while True:
        position = lowered.find(needle, offset)
        if position < 0:
            break
        positions.append(position)
        offset = position + max(1, len(needle))
    if not positions:
        return None
    numbers = _pin_number_text(pin)
    preferred: list[int] = []
    if numbers:
        for position in positions:
            nearby = lowered[max(0, position - 120) : position + len(needle) + 120]
            if any(
                re.search(
                    rf"(?<![a-z0-9]){re.escape(number)}(?![a-z0-9])",
                    nearby,
                )
                for number in numbers
            ):
                preferred.append(position)
    ranked = preferred + [position for position in positions if position not in preferred]
    return next(
        (position for position in ranked if position not in used_positions),
        ranked[0],
    )


def _render_prompt(part: str, source: str) -> str:
    return (
        f"Extract the pin table for the exact {part} package variant from the "
        "source fragment below. Extract every complete pin definition present "
        "in this fragment and no pins that are absent. Return one entry per "
        "physical package pin; repeated names are allowed only when their pin "
        "numbers differ. Return ONLY valid JSON in this shape: "
        '{"pins": [{"pin_no": 1, "name": "PA0", "type": "gpio", '
        '"functions": ["..."], "dir": "I/O"}]}'
        f"{MARKER}{source.strip()}"
    )


def _sequence_tokens(
    tokenizer: Any,
    prompt: str,
    response: str,
) -> int:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    return len(prompt_ids) + len(response_ids) + 1


def _candidate(
    *,
    part: str,
    source: str,
    anchored: list[tuple[int, dict[str, Any]]],
    start_index: int,
    end_index: int,
    tokenizer: Any,
) -> tuple[str, str, int, int, int]:
    first_anchor = anchored[start_index][0]
    previous_end = (
        anchored[start_index - 1][0]
        + len(str(anchored[start_index - 1][1].get("name") or ""))
        if start_index
        else 0
    )
    source_start = max(previous_end, first_anchor - 240)
    next_index = end_index + 1
    if next_index < len(anchored):
        source_end = max(
            anchored[end_index][0] + 1,
            anchored[next_index][0] - 1,
        )
    else:
        source_end = min(len(source), anchored[end_index][0] + 2000)
    pins = [row for _position, row in anchored[start_index : end_index + 1]]
    prompt = _render_prompt(part, source[source_start:source_end])
    response = json.dumps(
        {"pins": pins},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        prompt,
        response,
        _sequence_tokens(tokenizer, prompt, response),
        source_start,
        source_end,
    )


def chunk_records(
    records: Iterable[dict[str, Any]],
    *,
    tokenizer: Any,
    cutoff_len: int,
    max_pins_per_chunk: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        pair_id = str(record.get("pair_id") or f"record-{record_index}")
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or MARKER not in prompt:
            rejections.append(
                {"pair_id": pair_id, "reason": "prompt lacks PIN TABLE TEXT marker"}
            )
            continue
        source = prompt.split(MARKER, 1)[1]
        try:
            response = json.loads(str(record.get("response") or ""))
        except json.JSONDecodeError:
            rejections.append(
                {"pair_id": pair_id, "reason": "response is not valid JSON"}
            )
            continue
        pins = response.get("pins") if isinstance(response, dict) else None
        if not isinstance(pins, list) or not pins:
            rejections.append(
                {"pair_id": pair_id, "reason": "response has no pin records"}
            )
            continue
        anchored: list[tuple[int, dict[str, Any]]] = []
        used_positions: set[int] = set()
        seen_physical_pins: set[tuple[str, str]] = set()
        for pin_index, pin in enumerate(pins):
            if not isinstance(pin, dict):
                rejections.append(
                    {
                        "pair_id": pair_id,
                        "pin_index": pin_index,
                        "reason": "pin record is not an object",
                    }
                )
                continue
            physical_pin = (
                str(pin.get("name") or ""),
                json.dumps(
                    pin.get("pin_no"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            if physical_pin in seen_physical_pins:
                rejections.append(
                    {
                        "pair_id": pair_id,
                        "pin_index": pin_index,
                        "pin_name": physical_pin[0],
                        "reason": "duplicate physical pin label",
                    }
                )
                continue
            seen_physical_pins.add(physical_pin)
            position = _anchor(source, pin, used_positions=used_positions)
            if position is None:
                rejections.append(
                    {
                        "pair_id": pair_id,
                        "pin_index": pin_index,
                        "pin_name": str(pin.get("name") or ""),
                        "reason": "pin name is not grounded in source text",
                    }
                )
                continue
            used_positions.add(position)
            anchored.append((position, pin))
        anchored.sort(key=lambda item: (item[0], str(item[1].get("name") or "")))
        start = 0
        record_chunks: list[dict[str, Any]] = []
        while start < len(anchored):
            accepted: tuple[str, str, int, int, int] | None = None
            accepted_end = start
            stop = min(len(anchored), start + max_pins_per_chunk)
            end = start
            while end < stop:
                unit_end = end
                while (
                    unit_end + 1 < len(anchored)
                    and anchored[unit_end + 1][0] == anchored[end][0]
                ):
                    unit_end += 1
                candidate = _candidate(
                    part=str((record.get("metadata") or {}).get("part") or pair_id),
                    source=source,
                    anchored=anchored,
                    start_index=start,
                    end_index=unit_end,
                    tokenizer=tokenizer,
                )
                if candidate[2] > cutoff_len:
                    break
                source_fragment = candidate[0].split(MARKER, 1)[1].casefold()
                if any(
                    str(pin.get("name") or "").casefold() not in source_fragment
                    for _position, pin in anchored[start : unit_end + 1]
                ):
                    break
                accepted = candidate
                accepted_end = unit_end
                end = unit_end + 1
            if accepted is None:
                pin = anchored[start][1]
                rejections.append(
                    {
                        "pair_id": pair_id,
                        "pin_name": str(pin.get("name") or ""),
                        "reason": "single grounded pin block exceeds cutoff",
                    }
                )
                start += 1
                continue
            chunk_index = len(record_chunks)
            chunk_prompt, chunk_response, tokens, source_start, source_end = accepted
            provenance = dict(record.get("provenance") or {})
            provenance["source_record_id"] = (
                f"{provenance.get('source_record_id') or pair_id}:chunk:{chunk_index}"
            )
            provenance["content_sha256"] = hashlib.sha256(
                _canonical(
                    {
                        "prompt": chunk_prompt,
                        "response": chunk_response,
                    }
                )
            ).hexdigest()
            metadata = {
                **dict(record.get("metadata") or {}),
                "parent_pair_id": pair_id,
                "chunk_index": chunk_index,
                "sequence_tokens": tokens,
                "source_start": source_start,
                "source_end": source_end,
            }
            record_chunks.append(
                {
                    **record,
                    "pair_id": f"{pair_id}-chunk-{chunk_index:04d}",
                    "prompt": chunk_prompt,
                    "response": chunk_response,
                    "provenance": provenance,
                    "metadata": metadata,
                }
            )
            start = accepted_end + 1
        chunk_count = len(record_chunks)
        for chunk in record_chunks:
            chunk["metadata"]["chunk_count"] = chunk_count
        chunks.extend(record_chunks)
    return chunks, rejections


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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    source = args.source.expanduser().resolve(strict=True)
    model = args.model.expanduser().resolve(strict=True)
    destination = args.destination.expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite destination: {destination}")
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        local_files_only=True,
        trust_remote_code=True,
    )
    records = _load_jsonl(source)
    chunks, rejections = chunk_records(
        records,
        tokenizer=tokenizer,
        cutoff_len=args.cutoff_len,
        max_pins_per_chunk=args.max_pins_per_chunk,
    )
    if not chunks:
        raise ValueError("chunking produced no training records")
    if any(
        int(row["metadata"]["sequence_tokens"]) > args.cutoff_len for row in chunks
    ):
        raise ValueError("chunking produced a sequence over cutoff")
    destination.mkdir(parents=True)
    canonical = destination / "canonical" / f"{args.split}.jsonl"
    llamafactory = (
        destination
        / "llamafactory"
        / f"designwins_text_{args.split}.json"
    )
    rejection_path = destination / "quarantine" / "rejections.json"
    _write_jsonl(canonical, chunks)
    _write_json(
        llamafactory,
        [
            {
                "instruction": row["prompt"],
                "input": "",
                "output": row["response"],
            }
            for row in chunks
        ],
    )
    _write_json(
        destination / "llamafactory" / "dataset_info.json",
        {
            f"designwins_text_{args.split}": {
                "file_name": llamafactory.name,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                },
            }
        },
    )
    _write_json(rejection_path, rejections)
    manifest = {
        "schema": "harness.dataset.designwins-chunked.v1",
        "source_sha256": _sha256(source),
        "chunker_sha256": _sha256(Path(__file__)),
        "cutoff_len": args.cutoff_len,
        "split": args.split,
        "max_pins_per_chunk": args.max_pins_per_chunk,
        "source_records": len(records),
        "chunks": len(chunks),
        "rejections": len(rejections),
        "maximum_sequence_tokens": max(
            int(row["metadata"]["sequence_tokens"]) for row in chunks
        ),
        "artifacts": {
            path.relative_to(destination).as_posix(): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (canonical, llamafactory, rejection_path)
        },
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--max-pins-per-chunk", type=int, default=16)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    args = parser.parse_args()
    if args.cutoff_len < 512 or args.max_pins_per_chunk < 1:
        parser.error("invalid chunking limits")
    return args


def main() -> int:
    result = build(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
