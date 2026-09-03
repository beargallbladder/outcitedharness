#!/usr/bin/env python3
"""Normalize the frozen DesignWins holdout prompt to its physical-pin labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


OLD_CONTRACT = "One entry per unique pin name."
NEW_CONTRACT = (
    "One entry per physical package pin; repeated names are allowed only when "
    "their pin numbers differ."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"holdout record {index} is not an object")
        instruction = record.get("instruction")
        output = record.get("output")
        if not isinstance(instruction, str) or instruction.count(OLD_CONTRACT) != 1:
            raise ValueError(
                f"holdout record {index} does not have the inherited contract"
            )
        if not isinstance(output, str):
            raise ValueError(f"holdout record {index} has no output")
        value = json.loads(output)
        pins = value.get("pins") if isinstance(value, dict) else None
        if not isinstance(pins, list) or not pins:
            raise ValueError(f"holdout record {index} has no pins")
        physical: set[tuple[str, str]] = set()
        for pin in pins:
            if not isinstance(pin, dict) or not isinstance(pin.get("name"), str):
                raise ValueError(f"holdout record {index} has an invalid pin")
            identity = (
                pin["name"],
                json.dumps(
                    pin.get("pin_no"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            if identity in physical:
                raise ValueError(
                    f"holdout record {index} duplicates a physical pin"
                )
            physical.add(identity)
        normalized.append(
            {
                **record,
                "instruction": instruction.replace(OLD_CONTRACT, NEW_CONTRACT),
            }
        )
    return normalized


def _write_once(path: Path, value: Any) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-records", required=True, type=int)
    args = parser.parse_args()
    source = args.source.expanduser().resolve(strict=True)
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != args.expected_records:
        raise ValueError("holdout record count does not match the frozen split")
    normalized = normalize(records)
    _write_once(args.output, normalized)
    _write_once(
        args.manifest,
        {
            "schema": "harness.designwins-physical-pin-holdout.v1",
            "records": len(normalized),
            "source_sha256": _sha256(source),
            "output_sha256": _sha256(args.output),
            "normalizer_sha256": _sha256(Path(__file__)),
            "transformation": "prompt_contract_only",
        },
    )
    print(args.manifest.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
