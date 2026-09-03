#!/usr/bin/env python3
"""Build immutable, lineage-split training artifacts from approved sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.training import (
    Split,
    assert_no_lineage_leakage,
    grouped_lineage_split,
    load_native_designwins_text_pairs,
    load_native_designwins_vision_pairs,
    write_pairs_jsonl,
)
from harness.training.models import TextPair, VisionPair


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _partition(pairs: list[TextPair]) -> dict[Split, list[TextPair]]:
    partitions = grouped_lineage_split(
        pairs,
        lineage_key=lambda pair: pair.provenance.lineage_id,
    )
    assert_no_lineage_leakage(
        partitions,
        lineage_key=lambda pair: pair.provenance.lineage_id,
    )
    return partitions


def _llamafactory_text(pair: TextPair) -> dict[str, str]:
    return {
        "instruction": pair.prompt,
        "input": "",
        "output": pair.response,
    }


def _llamafactory_vision(pair: VisionPair) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "\n".join("<image>" for _ in pair.image_uris)
                    + "\n"
                    + pair.prompt
                ),
            },
            {"role": "assistant", "content": pair.response},
        ],
        "images": [
            "../" + uri.removeprefix("dataset://designwins/")
            for uri in pair.image_uris
        ],
    }


def _designwins_eligibility(
    audit_root: Path,
    *,
    include_provenance_suspect: bool,
) -> tuple[set[str], dict[str, str], dict[str, Path]]:
    audit_root = audit_root.resolve(strict=True)
    quality_path = audit_root / "gt_audit.json"
    provenance_path = audit_root / "gt_provenance_audit.json"
    quality = json.loads(quality_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    if not isinstance(quality, dict) or not isinstance(provenance, dict):
        raise ValueError("DesignWins audit files must contain JSON objects")

    eligible: set[str] = set()
    reasons: dict[str, str] = {}
    for part, result in quality.items():
        if not isinstance(result, dict):
            reasons[str(part)] = "ground-truth audit record is malformed"
        elif result.get("ok") is True:
            eligible.add(str(part))
        else:
            problems = result.get("problems")
            detail = "; ".join(str(value) for value in problems or [])
            reasons[str(part)] = f"ground-truth audit failed: {detail or 'unspecified'}"

    if not include_provenance_suspect:
        suspect = provenance.get("suspect")
        if not isinstance(suspect, list):
            raise ValueError("provenance audit suspect list is missing")
        for record in suspect:
            if not isinstance(record, dict) or not record.get("part"):
                raise ValueError("provenance audit contains a malformed suspect record")
            part = str(record["part"])
            eligible.discard(part)
            reasons[part] = "ground-truth provenance audit marked this part suspect"
    return eligible, reasons, {
        "gt_audit.json": quality_path,
        "gt_provenance_audit.json": provenance_path,
    }


def build_designwins(
    source_root: Path,
    destination: Path,
    *,
    audit_root: Path | None = None,
    include_provenance_suspect: bool = False,
) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    eligible_parts: set[str] | None = None
    exclusion_reasons: dict[str, str] = {}
    audit_paths: dict[str, Path] = {}
    if audit_root is not None:
        eligible_parts, exclusion_reasons, audit_paths = _designwins_eligibility(
            audit_root,
            include_provenance_suspect=include_provenance_suspect,
        )
    text_source = source_root / "text_pairs.jsonl"
    vision_source = source_root / "vision_pairs.jsonl"
    text_rejections: list[dict[str, Any]] = []
    vision_rejections: list[dict[str, Any]] = []
    text_pairs = load_native_designwins_text_pairs(
        text_source,
        strict=False,
        rejections=text_rejections,
        eligible_parts=eligible_parts,
        exclusion_reasons=exclusion_reasons,
    )
    vision_pairs = load_native_designwins_vision_pairs(
        vision_source,
        strict=False,
        rejections=vision_rejections,
        eligible_parts=eligible_parts,
        exclusion_reasons=exclusion_reasons,
    )
    text_parts = _partition(text_pairs)
    vision_parts = _partition(vision_pairs)

    destination.mkdir(parents=True)
    source_images = source_root / "images"
    if source_images.is_dir():
        destination_images = destination / "images"
        if eligible_parts is None:
            shutil.copytree(
                source_images, destination_images, copy_function=shutil.copy2
            )
        else:
            destination_images.mkdir()
            for part in sorted(eligible_parts):
                source_part = source_images / part
                if source_part.is_dir():
                    shutil.copytree(
                        source_part,
                        destination_images / part,
                        copy_function=shutil.copy2,
                    )

    generated: list[Path] = []
    dataset_info: dict[str, Any] = {}
    for split in Split:
        text_path = destination / "canonical" / "text" / f"{split.value}.jsonl"
        vision_path = destination / "canonical" / "vision" / f"{split.value}.jsonl"
        write_pairs_jsonl(text_path, text_parts[split])
        write_pairs_jsonl(vision_path, vision_parts[split])
        generated.extend((text_path, vision_path))

        lf_text = destination / "llamafactory" / f"designwins_text_{split.value}.json"
        lf_vision = destination / "llamafactory" / f"designwins_vision_{split.value}.json"
        _write_json(lf_text, [_llamafactory_text(pair) for pair in text_parts[split]])
        _write_json(
            lf_vision,
            [_llamafactory_vision(pair) for pair in vision_parts[split]],
        )
        generated.extend((lf_text, lf_vision))
        dataset_info[f"designwins_text_{split.value}"] = {
            "file_name": lf_text.name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
        dataset_info[f"designwins_vision_{split.value}"] = {
            "file_name": lf_vision.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    dataset_info_path = destination / "llamafactory" / "dataset_info.json"
    _write_json(dataset_info_path, dataset_info)
    generated.append(dataset_info_path)
    rejection_path = destination / "quarantine" / "rejections.json"
    _write_json(
        rejection_path,
        {
            "text": text_rejections,
            "vision": vision_rejections,
        },
    )
    generated.append(rejection_path)

    manifest = {
        "schema": "harness.dataset.designwins.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "text_pairs.jsonl": {
                "sha256": _sha256(text_source),
                "bytes": text_source.stat().st_size,
            },
            "vision_pairs.jsonl": {
                "sha256": _sha256(vision_source),
                "bytes": vision_source.stat().st_size,
            },
            **{
                name: {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in sorted(audit_paths.items())
            },
        },
        "eligibility": {
            "audit_required": audit_root is not None,
            "provenance_suspect_included": include_provenance_suspect,
            "eligible_parts": len(eligible_parts or set())
            if audit_root is not None
            else None,
        },
        "counts": {
            "text": {split.value: len(text_parts[split]) for split in Split},
            "vision": {split.value: len(vision_parts[split]) for split in Split},
            "quarantine": {
                "text": len(text_rejections),
                "vision": len(vision_rejections),
            },
        },
        "artifacts": {
            path.relative_to(destination).as_posix(): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(generated)
        },
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    designwins = subparsers.add_parser("designwins")
    designwins.add_argument("--source-root", required=True, type=Path)
    designwins.add_argument("--destination", required=True, type=Path)
    designwins.add_argument(
        "--audit-root",
        type=Path,
        help="directory containing gt_audit.json and gt_provenance_audit.json",
    )
    designwins.add_argument(
        "--include-provenance-suspect",
        action="store_true",
        help="include quality-passing parts flagged by the provenance audit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "designwins":
            audit_root = args.audit_root or args.source_root.parent
            manifest = build_designwins(
                args.source_root,
                args.destination,
                audit_root=audit_root,
                include_provenance_suspect=args.include_provenance_suspect,
            )
            print(json.dumps(manifest["counts"], sort_keys=True))
        return 0
    except Exception as error:
        print(f"training dataset build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
