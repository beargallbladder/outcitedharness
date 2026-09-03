#!/usr/bin/env python3
"""Build immutable row-crop vision examples from a sealed alignment audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from harness.training.security import assert_value_no_secrets


SCHEMA = "harness.pinout-vision-row-dataset.v1"
ALIGNMENT_SCHEMA = "harness.pinout-vision-alignment-audit.v1"
SOURCE_SCHEMA = "harness.pinout-vision-source-audit.v1"
AUTHORIZATION_SCHEMA = "harness.training-authorization.v1"
SPLIT_FRACTIONS = {"train": 0.82, "validation": 0.09, "test": 0.09}


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


def _json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input must be a regular file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"input must contain a JSON object: {path}")
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


def _pin_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    pinout = value.get("pinout")
    rows = pinout.get("pin_functions_summary") if isinstance(pinout, dict) else None
    if not isinstance(rows, list):
        raise ValueError("frontier target has no pin_functions_summary")
    return rows


def _authorization(path: Path) -> dict[str, Any]:
    receipt = _json_object(path)
    if receipt.get("schema") != AUTHORIZATION_SCHEMA:
        raise ValueError("authorization receipt schema is not supported")
    if receipt.get("training_authorized") is not True:
        raise ValueError("authorization receipt does not authorize training")
    scope = receipt.get("scope")
    if (
        not isinstance(scope, dict)
        or scope.get("dataset_kind") != "frontier-validated-pinout-row-crops"
        or scope.get("model") != "Qwen3-VL-8B-Instruct"
        or scope.get("method") != "offline-lora-sft"
    ):
        raise ValueError("authorization scope does not match this dataset")
    constraints = receipt.get("constraints")
    if (
        not isinstance(constraints, dict)
        or constraints.get("required_validation") != "VALIDATED_GROUND_TRUTH"
        or constraints.get("require_exact_rendered-page_alignment") is not True
        or constraints.get("network_during_training") != "none"
    ):
        raise ValueError("authorization constraints are incomplete")
    evidence_found = False
    evidence_files: list[dict[str, str]] = []
    for basis in receipt.get("basis") or []:
        if not isinstance(basis, dict) or basis.get("kind") != "corpus-owner-response":
            continue
        evidence_path = Path(str(basis.get("path") or ""))
        expected_sha256 = str(basis.get("sha256") or "")
        if _sha256(evidence_path) != expected_sha256:
            raise ValueError("authorization evidence changed or is missing")
        evidence_found = True
        evidence_files.append(
            {
                "source_path": str(evidence_path.resolve(strict=True)),
                "sha256": expected_sha256,
            }
        )
    if not evidence_found:
        raise ValueError("authorization has no hash-bound corpus-owner response")
    return {
        "status": "authorized",
        "training_authorized": True,
        "receipt": {
            "source_path": str(path.resolve(strict=True)),
            "sha256": _sha256(path),
        },
        "evidence": evidence_files,
    }


def _split_lineages(weights: dict[str, int]) -> dict[str, str]:
    total = sum(weights.values())
    targets = {
        split: total * fraction for split, fraction in SPLIT_FRACTIONS.items()
    }
    counts = {split: 0 for split in SPLIT_FRACTIONS}
    assignments: dict[str, str] = {}
    ordered = sorted(
        weights,
        key=lambda lineage: (
            -weights[lineage],
            hashlib.sha256(f"pinout-v1:{lineage}".encode()).hexdigest(),
        ),
    )
    for lineage in ordered:
        split = max(
            SPLIT_FRACTIONS,
            key=lambda candidate: (
                targets[candidate] - counts[candidate],
                SPLIT_FRACTIONS[candidate],
            ),
        )
        assignments[lineage] = split
        counts[split] += weights[lineage]
    if len(weights) >= len(SPLIT_FRACTIONS) and set(assignments.values()) != set(
        SPLIT_FRACTIONS
    ):
        raise RuntimeError("lineage splitter produced an empty split")
    return assignments


def _expanded_clip(
    bbox: Iterable[float],
    page_rect: Any,
    *,
    padding_points: float = 3.0,
) -> Any:
    import pymupdf as fitz

    rectangle = fitz.Rect(tuple(float(value) for value in bbox))
    rectangle.x0 -= padding_points
    rectangle.y0 -= padding_points
    rectangle.x1 += padding_points
    rectangle.y1 += padding_points
    rectangle &= page_rect
    if rectangle.is_empty or rectangle.width < 1 or rectangle.height < 1:
        raise ValueError("alignment produced an empty image crop")
    return rectangle


def _render(page: Any, bbox: Iterable[float], destination: Path, dpi: int) -> None:
    import pymupdf as fitz

    destination.parent.mkdir(parents=True, exist_ok=True)
    clip = _expanded_clip(bbox, page.rect)
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(dpi / 72, dpi / 72),
        clip=clip,
        alpha=False,
        annots=False,
    )
    if pixmap.width < 16 or pixmap.height < 16:
        raise ValueError("rendered image crop is too small")
    pixmap.save(destination)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def build_dataset(
    *,
    alignment_audit_path: Path,
    destination: Path,
    dpi: int = 240,
    authorization_receipt: Path | None = None,
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"immutable destination already exists: {destination}")
    if not 144 <= dpi <= 400:
        raise ValueError("DPI must be between 144 and 400")
    alignment = _json_object(alignment_audit_path)
    _verify_evidence(alignment, ALIGNMENT_SCHEMA, "alignment audit")
    if alignment["policy"].get("limited_probe") is True:
        raise ValueError("cannot build a dataset from a limited alignment probe")
    if alignment["policy"].get("training_authorized") is not True:
        raise ValueError("alignment audit did not meet the minimum example gate")

    source_audit_path = Path(alignment["source_audit"]["path"])
    if _sha256(source_audit_path) != alignment["source_audit"]["sha256"]:
        raise ValueError("source audit changed after alignment")
    source = _json_object(source_audit_path)
    _verify_evidence(source, SOURCE_SCHEMA, "source audit")
    source_candidates = {
        str(row["record_id"]): row for row in source["candidates"]
    }
    source_paths = source["sources"]
    ground_truth_root = Path(source_paths["ground_truth_root"]).resolve(strict=True)
    pdf_root = Path(source_paths["pdf_root"]).resolve(strict=True)

    authorization: dict[str, Any] = {
        "status": "pending",
        "training_authorized": False,
        "receipt": None,
    }
    if authorization_receipt is not None:
        authorization = _authorization(authorization_receipt)

    prepared: list[dict[str, Any]] = []
    lineage_weights: Counter[str] = Counter()
    for record in alignment["records"]:
        if record.get("row_crop_status") != "eligible":
            continue
        record_id = str(record["record_id"])
        source_candidate = source_candidates.get(record_id)
        if source_candidate is None:
            raise ValueError(f"alignment record is absent from source audit: {record_id}")
        ground_truth_path = ground_truth_root / source_candidate["ground_truth_path"]
        if _sha256(ground_truth_path) != source_candidate["ground_truth_sha256"]:
            raise ValueError(f"ground truth changed after source audit: {record_id}")
        target_rows = _pin_rows(_json_object(ground_truth_path))
        used_indices: set[int] = set()
        for chunk_number, chunk in enumerate(record["row_crop_chunks"]):
            target_indices = [int(index) for index in chunk["target_indices"]]
            if (
                not target_indices
                or len(set(target_indices)) != len(target_indices)
                or used_indices.intersection(target_indices)
                or any(index < 0 or index >= len(target_rows) for index in target_indices)
            ):
                raise ValueError(f"invalid row-crop target indices: {record_id}")
            used_indices.update(target_indices)
            selected_rows = [target_rows[index] for index in target_indices]
            assert_value_no_secrets(selected_rows, field=f"{record_id}.target")
            identity = {
                "record_id": record_id,
                "pdf_sha256": record["pdf_sha256"],
                "page_1based": chunk["page_1based"],
                "table_index": chunk["table_index"],
                "source_rows": chunk["source_rows"],
                "target_indices": target_indices,
            }
            example_id = hashlib.sha256(_canonical(identity)).hexdigest()[:24]
            prepared.append(
                {
                    "example_id": example_id,
                    "record_id": record_id,
                    "pdf_sha256": record["pdf_sha256"],
                    "pdf_path": source_candidate["pdf_path"],
                    "ground_truth_sha256": source_candidate["ground_truth_sha256"],
                    "frontier_batch_id": source_candidate["frontier_batch_id"],
                    "published_sha256": source_candidate["published_sha256"],
                    "page_1based": int(chunk["page_1based"]),
                    "table_index": int(chunk["table_index"]),
                    "package": chunk["package_candidate"],
                    "package_header": chunk["package_header"],
                    "header_bbox": chunk["header_bbox"],
                    "body_bbox": chunk["body_bbox"],
                    "source_rows": chunk["source_rows"],
                    "target_indices": target_indices,
                    "target_rows": selected_rows,
                    "chunk_number": chunk_number,
                }
            )
            lineage_weights[str(record["pdf_sha256"])] += 1
    if not prepared:
        raise ValueError("alignment audit produced no row-crop examples")
    assignments = _split_lineages(dict(lineage_weights))

    destination = destination.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        import pymupdf as fitz

        if authorization["training_authorized"]:
            evidence_root = temporary / "evidence"
            evidence_root.mkdir()
            receipt_source = Path(authorization["receipt"]["source_path"])
            receipt_relative = Path("evidence") / "training-authorization.json"
            shutil.copy2(receipt_source, temporary / receipt_relative)
            authorization["receipt"]["bundled_path"] = receipt_relative.as_posix()
            for index, evidence in enumerate(authorization["evidence"], 1):
                source_path = Path(evidence["source_path"])
                suffix = source_path.suffix or ".txt"
                bundled = (
                    Path("evidence")
                    / f"corpus-owner-response-{index:02d}{suffix}"
                )
                shutil.copy2(source_path, temporary / bundled)
                evidence["bundled_path"] = bundled.as_posix()

        examples: list[dict[str, Any]] = []
        image_artifacts: dict[str, dict[str, Any]] = {}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in prepared:
            grouped[row["pdf_sha256"]].append(row)
        for lineage_index, (pdf_sha256, rows) in enumerate(sorted(grouped.items()), 1):
            pdf_path = pdf_root / rows[0]["pdf_path"]
            if _sha256(pdf_path) != pdf_sha256:
                raise ValueError(f"PDF changed after source audit: {pdf_path}")
            split = assignments[pdf_sha256]
            document = fitz.open(pdf_path)
            rendered_headers: dict[str, str] = {}
            try:
                for row in rows:
                    page = document[row["page_1based"] - 1]
                    image_root = (
                        temporary / "images" / split / pdf_sha256[:16]
                    )
                    header_key = hashlib.sha256(
                        _canonical(
                            {
                                "page": row["page_1based"],
                                "table": row["table_index"],
                                "bbox": row["header_bbox"],
                            }
                        )
                    ).hexdigest()[:20]
                    header_relative = (
                        Path("images")
                        / split
                        / pdf_sha256[:16]
                        / f"header-{header_key}.png"
                    )
                    if header_key not in rendered_headers:
                        _render(
                            page,
                            row["header_bbox"],
                            temporary / header_relative,
                            dpi,
                        )
                        rendered_headers[header_key] = header_relative.as_posix()
                    body_relative = (
                        Path("images")
                        / split
                        / pdf_sha256[:16]
                        / f"body-{row['example_id']}.png"
                    )
                    _render(
                        page,
                        row["body_bbox"],
                        temporary / body_relative,
                        dpi,
                    )
                    image_paths = [
                        rendered_headers[header_key],
                        body_relative.as_posix(),
                    ]
                    image_sha256 = []
                    for relative in image_paths:
                        path = temporary / relative
                        digest = _sha256(path)
                        image_sha256.append(digest)
                        image_artifacts.setdefault(
                            relative,
                            {"sha256": digest, "bytes": path.stat().st_size},
                        )
                    prompt = (
                        "Image 1 is the source table header. Image 2 is a "
                        "contiguous row crop from the same definition table. "
                        f"Extract exactly the {len(row['target_rows'])} visible "
                        f"rows for package column {json.dumps(row['package'])}. "
                        "Use the physical pin/ball identifier and signal name from "
                        "the images. Return only valid JSON with schema "
                        '{"pins":[{"pin_no":"...","name":"...","type":"...",'
                        '"functions":["..."],"dir":"..."}]}.'
                    )
                    response = json.dumps(
                        {"pins": row["target_rows"]},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    examples.append(
                        {
                            "example_id": row["example_id"],
                            "split": split,
                            "lineage_id": f"pdf-sha256:{pdf_sha256}",
                            "record_id": row["record_id"],
                            "prompt": prompt,
                            "response": response,
                            "images": image_paths,
                            "image_sha256": image_sha256,
                            "provenance": {
                                "pdf_sha256": pdf_sha256,
                                "ground_truth_sha256": row["ground_truth_sha256"],
                                "published_sha256": row["published_sha256"],
                                "frontier_batch_id": row["frontier_batch_id"],
                            },
                            "alignment": {
                                "page_1based": row["page_1based"],
                                "table_index": row["table_index"],
                                "package": row["package"],
                                "package_header": row["package_header"],
                                "header_bbox": row["header_bbox"],
                                "body_bbox": row["body_bbox"],
                                "source_rows": row["source_rows"],
                                "target_indices": row["target_indices"],
                            },
                        }
                    )
            finally:
                document.close()
            if lineage_index % 25 == 0:
                print(
                    f"rendered {lineage_index}/{len(grouped)} PDF lineages "
                    f"({len(examples)} examples)",
                    flush=True,
                )

        examples.sort(key=lambda row: (row["split"], row["example_id"]))
        split_rows = {
            split: [row for row in examples if row["split"] == split]
            for split in SPLIT_FRACTIONS
        }
        artifacts: dict[str, dict[str, Any]] = {}
        dataset_info: dict[str, Any] = {}
        for split, rows in split_rows.items():
            canonical_path = temporary / "canonical" / f"{split}.jsonl"
            _write_jsonl(canonical_path, rows)
            llamafactory_path = temporary / "llamafactory" / f"{split}.json"
            _write_json(
                llamafactory_path,
                [
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "<image>\n<image>\n" + row["prompt"],
                            },
                            {"role": "assistant", "content": row["response"]},
                        ],
                        "images": ["../" + image for image in row["images"]],
                    }
                    for row in rows
                ],
            )
            dataset_info[f"pinout_rows_{split}"] = {
                "file_name": f"{split}.json",
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            }
            for path in (canonical_path, llamafactory_path):
                artifacts[path.relative_to(temporary).as_posix()] = {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
        dataset_info_path = temporary / "llamafactory" / "dataset_info.json"
        _write_json(dataset_info_path, dataset_info)
        artifacts[dataset_info_path.relative_to(temporary).as_posix()] = {
            "sha256": _sha256(dataset_info_path),
            "bytes": dataset_info_path.stat().st_size,
        }
        image_manifest_path = temporary / "image-manifest.jsonl"
        _write_jsonl(
            image_manifest_path,
            [
                {"path": path, **metadata}
                for path, metadata in sorted(image_artifacts.items())
            ],
        )
        artifacts[image_manifest_path.name] = {
            "sha256": _sha256(image_manifest_path),
            "bytes": image_manifest_path.stat().st_size,
        }
        evidence_directory = temporary / "evidence"
        if evidence_directory.is_dir():
            for path in sorted(evidence_directory.glob("*")):
                artifacts[path.relative_to(temporary).as_posix()] = {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }

        split_lineages = {
            split: {
                row["lineage_id"]
                for row in split_rows[split]
            }
            for split in SPLIT_FRACTIONS
        }
        if any(
            split_lineages[left] & split_lineages[right]
            for left in SPLIT_FRACTIONS
            for right in SPLIT_FRACTIONS
            if left < right
        ):
            raise RuntimeError("PDF lineage leaked across dataset splits")
        manifest_core: dict[str, Any] = {
            "schema": SCHEMA,
            "authorization": authorization,
            "rendering": {
                "dpi": dpi,
                "resize": "none",
                "format": "png",
                "images_per_example": 2,
            },
            "sources": {
                "alignment_audit": {
                    "path": str(alignment_audit_path.resolve(strict=True)),
                    "sha256": _sha256(alignment_audit_path),
                    "evidence_sha256": alignment["evidence_sha256"],
                },
                "source_audit": {
                    "path": str(source_audit_path.resolve(strict=True)),
                    "sha256": _sha256(source_audit_path),
                    "evidence_sha256": source["evidence_sha256"],
                },
            },
            "split_policy": {
                "lineage_key": "source_pdf_sha256",
                "fractions": SPLIT_FRACTIONS,
                "leakage_check_passed": True,
            },
            "counts": {
                "examples": {
                    split: len(rows) for split, rows in split_rows.items()
                },
                "pdf_lineages": {
                    split: len(split_lineages[split])
                    for split in SPLIT_FRACTIONS
                },
                "unique_images": len(image_artifacts),
                "target_rows": sum(
                    len(json.loads(row["response"])["pins"]) for row in examples
                ),
            },
            "artifacts": artifacts,
            "images_aggregate_sha256": hashlib.sha256(
                _canonical(image_artifacts)
            ).hexdigest(),
        }
        manifest_core["evidence_sha256"] = hashlib.sha256(
            _canonical(manifest_core)
        ).hexdigest()
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **manifest_core,
        }
        _write_json(temporary / "manifest.json", manifest)
        temporary.rename(destination)
        _make_read_only(destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-audit", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--authorization-receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    manifest = build_dataset(
        alignment_audit_path=arguments.alignment_audit,
        destination=arguments.destination,
        dpi=arguments.dpi,
        authorization_receipt=arguments.authorization_receipt,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    if manifest["authorization"]["training_authorized"]:
        print("DATASET_SEALED_AND_TRAINING_AUTHORIZED")
    else:
        print("DATASET_SEALED_IN_QUARANTINE_PENDING_AUTHORIZATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
