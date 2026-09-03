#!/usr/bin/env python3
"""Bind a completed DesignWins evaluation to immutable input/runtime hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_once(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == payload:
            return
        raise FileExistsError(f"refusing to overwrite sealed evaluation: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seal(
    source: Path,
    output: Path,
    *,
    dataset: Path,
    model_manifest: Path,
    scorer: Path,
    runtime_image_id: str,
    adapter_manifest: Path | None,
    max_samples: int,
    cutoff_len: int,
    max_new_tokens: int,
    batch_size: int,
    generation_slack_tokens: int,
) -> dict[str, Any]:
    value = json.loads(source.read_text(encoding="utf-8"))
    details = value.get("details")
    if not isinstance(details, list) or len(details) != max_samples:
        raise ValueError(f"evaluation must contain exactly {max_samples} details")
    if value.get("identity") is not None:
        raise ValueError("source evaluation is already sealed")
    core_sha256 = hashlib.sha256(_canonical(value)).hexdigest()
    value["identity"] = {
        "schema": "harness.designwins-evaluation-identity.v1",
        "core_sha256": core_sha256,
        "dataset_sha256": _sha256(dataset),
        "model_manifest_sha256": _sha256(model_manifest),
        "scorer_sha256": _sha256(scorer),
        "adapter_manifest_sha256": (
            _sha256(adapter_manifest) if adapter_manifest is not None else None
        ),
        "runtime_image_id": runtime_image_id,
        "max_samples": max_samples,
        "cutoff_len": cutoff_len,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "generation_slack_tokens": generation_slack_tokens,
    }
    _write_once(output, value)
    return value["identity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--scorer", required=True, type=Path)
    parser.add_argument("--runtime-image-id", required=True)
    parser.add_argument("--adapter-manifest", type=Path)
    parser.add_argument("--max-samples", type=int, default=141)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--generation-slack-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    identity = seal(
        args.source,
        args.output,
        dataset=args.dataset,
        model_manifest=args.model_manifest,
        scorer=args.scorer,
        runtime_image_id=args.runtime_image_id,
        adapter_manifest=args.adapter_manifest,
        max_samples=args.max_samples,
        cutoff_len=args.cutoff_len,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        generation_slack_tokens=args.generation_slack_tokens,
    )
    print(json.dumps(identity, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
