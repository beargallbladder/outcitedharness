#!/usr/bin/env python3
"""Prepare or verify an isolated one-step LlamaFactory resume smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_checkpoint(root: Path) -> tuple[int, Path]:
    rows: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint-*"):
        try:
            rows.append((int(path.name.removeprefix("checkpoint-")), path))
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"no checkpoint found below {root}")
    return max(rows)


def prepare(
    source_config: Path,
    checkpoint_root: Path,
    output_dir: Path,
    destination_config: Path,
) -> int:
    source_step, checkpoint = _latest_checkpoint(checkpoint_root)
    for name in ("adapter_model.safetensors", "optimizer.pt", "trainer_state.json"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"resume checkpoint lacks {name}")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if int(state["global_step"]) != source_step:
        raise ValueError("checkpoint name and trainer state disagree")
    final_state_path = checkpoint_root / "trainer_state.json"
    final_adapter = checkpoint_root / "adapter_model.safetensors"
    if not final_state_path.is_file() or not final_adapter.is_file():
        raise ValueError("final candidate lacks trainer state or adapter")
    final_step = int(json.loads(final_state_path.read_text())["global_step"])
    if final_step != source_step:
        raise ValueError("final candidate is not backed by an exact resume checkpoint")
    if _sha256(final_adapter) != _sha256(
        checkpoint / "adapter_model.safetensors"
    ):
        raise ValueError("final adapter differs from its resume checkpoint")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config must be a mapping")
    config.update(
        {
            "output_dir": str(output_dir),
            "resume_from_checkpoint": str(checkpoint),
            "max_steps": source_step + 1,
            "save_steps": 1,
            "save_total_limit": 1,
            "overwrite_output_dir": False,
        }
    )
    _write_text(
        destination_config,
        yaml.safe_dump(config, sort_keys=False),
    )
    print(
        json.dumps(
            {
                "source_checkpoint": str(checkpoint),
                "source_step": source_step,
                "expected_step": source_step + 1,
            },
            sort_keys=True,
        )
    )
    return source_step


def verify(
    source_config: Path,
    checkpoint_root: Path,
    output_dir: Path,
    destination_config: Path,
    summary: Path,
) -> None:
    source_step, source_checkpoint = _latest_checkpoint(checkpoint_root)
    resumed_step, resumed_checkpoint = _latest_checkpoint(output_dir)
    if resumed_step != source_step + 1:
        raise ValueError(
            f"resume advanced to {resumed_step}, expected {source_step + 1}"
        )
    state = json.loads(
        (resumed_checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    if int(state["global_step"]) != resumed_step:
        raise ValueError("resumed checkpoint and trainer state disagree")
    adapter = resumed_checkpoint / "adapter_model.safetensors"
    if not adapter.is_file():
        raise ValueError("resume did not save an adapter")
    candidate_adapter = checkpoint_root / "adapter_model.safetensors"
    candidate_sha256 = _sha256(candidate_adapter)
    resumed_sha256 = _sha256(adapter)
    source_adapter_sha256 = _sha256(
        source_checkpoint / "adapter_model.safetensors"
    )
    if candidate_sha256 != source_adapter_sha256:
        raise ValueError(
            "resume checkpoint does not reproduce evaluated candidate"
        )
    value: dict[str, Any] = {
        "schema": "harness.designwins.resume-smoke.v1",
        "passed": True,
        "source_step": source_step,
        "resumed_step": resumed_step,
        "source_checkpoint": str(source_checkpoint),
        "resumed_checkpoint": str(resumed_checkpoint),
        "sha256": {
            "source_config": _sha256(source_config),
            "resume_config": _sha256(destination_config),
            "source_trainer_state": _sha256(
                source_checkpoint / "trainer_state.json"
            ),
            "source_optimizer": _sha256(source_checkpoint / "optimizer.pt"),
            "candidate_adapter": candidate_sha256,
            "source_adapter": source_adapter_sha256,
            "resumed_adapter": resumed_sha256,
        },
    }
    _write_text(summary, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--destination-config", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.action == "verify" and args.summary is None:
        parser.error("verify requires --summary")
    return args


def main() -> int:
    args = parse_args()
    if args.action == "prepare":
        prepare(
            args.source_config,
            args.checkpoint_root,
            args.output_dir,
            args.destination_config,
        )
    else:
        verify(
            args.source_config,
            args.checkpoint_root,
            args.output_dir,
            args.destination_config,
            args.summary,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
