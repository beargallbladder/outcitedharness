"""Local-first import and work planning for datasheet extraction."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from harness.electronics.claims import canonical_json, make_claim
from harness.electronics.corpus import sha256_file, verify_corpus_registry
from harness.electronics.models import (
    ClaimClass,
    EntityGrain,
    EntityReference,
    EvidenceKind,
    EvidenceReference,
    FactClaim,
    ModelIdentity,
)


WORK_QUEUE_SCHEMA = "harness.electronics-extraction-work-queue.v1"
SUPPORTED_ROW_DATASET_SCHEMA = "harness.pinout-vision-row-dataset.v1"
GT_SECTIONS = (
    "interface_atoms",
    "peripheral_depth",
    "clock_specs",
    "abs_max_ratings",
    "power_modes",
    "overview",
)
UNIT_SUFFIXES = {
    "_mhz": "MHz",
    "_khz": "kHz",
    "_hz": "Hz",
    "_kb": "KiB",
    "_mb": "MiB",
    "_ua": "uA",
    "_ma": "mA",
    "_c": "degC",
    "_v": "V",
    "_bits": "bit",
}


def _safe_component(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "-", str(value).strip())
    cleaned = cleaned.strip("-.")
    if not cleaned:
        return hashlib.sha256(str(value).encode()).hexdigest()[:12]
    return cleaned[:80]


def _field_component(value: str) -> str:
    output = re.sub(r"[^a-z0-9_.-]+", "_", value.casefold()).strip("_.-")
    if not output or not output[0].isalpha():
        output = f"value_{output}"
    return output[:128]


def _unit(field: str) -> str | None:
    lowered = field.casefold()
    for suffix, unit in UNIT_SUFFIXES.items():
        if lowered.endswith(suffix):
            return unit
    return None


def _valid_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, dict)) and not value:
        return False
    return True


def _pdf_uris(registry: Mapping[str, Any]) -> dict[str, str]:
    root = Path(str(registry["sources"]["pdf_root"])).expanduser().resolve()
    output: dict[str, str] = {}
    for document in registry["documents"]:
        paths = document.get("paths") or []
        if paths:
            output[str(document["document_sha256"])] = (
                root / str(paths[0])
            ).resolve().as_uri()
    return output


def _document_records(
    registry: Mapping[str, Any],
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    gt_root = Path(
        str(registry["sources"]["ground_truth_root"])
    ).expanduser().resolve()
    for document in registry["documents"]:
        for summary in document.get("ground_truth") or []:
            path = (gt_root / str(summary["path"])).resolve()
            if not path.is_relative_to(gt_root):
                raise ValueError("ground-truth path escapes configured root")
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"ground truth is not an object: {path}")
            if sha256_file(path) != summary["sha256"]:
                raise ValueError(f"ground truth changed after corpus seal: {path}")
            yield document, {"summary": summary, "value": value, "path": path}


def iter_ground_truth_claims(
    registry: Mapping[str, Any],
    *,
    created_at: datetime,
) -> Iterator[FactClaim]:
    verify_corpus_registry(registry)
    pdf_uris = _pdf_uris(registry)
    for document, record in _document_records(registry):
        value = record["value"]
        summary = record["summary"]
        overview = value.get("overview")
        part_number = (
            overview.get("part_number")
            if isinstance(overview, Mapping)
            else None
        ) or record["path"].stem
        vendor = (
            overview.get("manufacturer")
            if isinstance(overview, Mapping)
            else None
        ) or summary.get("vendor")
        entity = EntityReference(
            entity_id=(
                f"opn:{_safe_component(vendor or 'unknown')}:"
                f"{_safe_component(part_number)}"
            ),
            grain=EntityGrain.OPN,
            canonical_id=str(part_number),
            vendor=str(vendor) if vendor else None,
        )
        source_uri = pdf_uris[str(document["document_sha256"])]
        evidence = (
            EvidenceReference(
                kind=EvidenceKind.SOURCE_RECORD,
                document_sha256=document["document_sha256"],
                source_uri=source_uri,
                artifact_sha256=summary["sha256"],
            ),
        )
        for section in GT_SECTIONS:
            section_value = value.get(section)
            if not isinstance(section_value, Mapping):
                continue
            for name, fact_value in sorted(section_value.items()):
                if not _valid_value(fact_value):
                    continue
                field = _field_component(f"{section}.{name}")
                yield make_claim(
                    entity=entity,
                    field=field,
                    value=fact_value,
                    unit=_unit(str(name)),
                    claim_class=ClaimClass.SEMANTIC_LABEL,
                    extraction_method="imported_frontier_ground_truth",
                    evidence=evidence,
                    conditions={
                        "source_record_id": summary["record_id"],
                        "page_evidence_status": "not_localized",
                    },
                    created_at=created_at,
                )
        pinout = value.get("pinout")
        if isinstance(pinout, Mapping):
            for name in ("packages", "declared_pin_total", "rows_emitted"):
                fact_value = pinout.get(name)
                if not _valid_value(fact_value):
                    continue
                yield make_claim(
                    entity=entity,
                    field=_field_component(f"pinout.{name}"),
                    value=fact_value,
                    claim_class=ClaimClass.SEMANTIC_LABEL,
                    extraction_method="imported_frontier_ground_truth",
                    evidence=evidence,
                    conditions={
                        "source_record_id": summary["record_id"],
                        "page_evidence_status": "not_localized",
                    },
                    created_at=created_at,
                )


def _load_row_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SUPPORTED_ROW_DATASET_SCHEMA:
        raise ValueError("pinout row dataset schema is not supported")
    for relative, receipt in value.get("artifacts", {}).items():
        if not relative.startswith("canonical/"):
            continue
        artifact = root / relative
        if sha256_file(artifact) != receipt.get("sha256"):
            raise ValueError(f"pinout row artifact hash mismatch: {relative}")
        if artifact.stat().st_size != receipt.get("bytes"):
            raise ValueError(f"pinout row artifact size mismatch: {relative}")
    return value


def _row_records(
    root: Path,
    splits: Sequence[str],
) -> Iterator[dict[str, Any]]:
    allowed = {"train", "validation", "test"}
    if not splits or set(splits) - allowed:
        raise ValueError("splits must contain train, validation, or test")
    _load_row_manifest(root)
    for split in splits:
        path = root / "canonical" / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}: invalid JSON at line {line_number}"
                    ) from exc
                if not isinstance(value, dict) or value.get("split") != split:
                    raise ValueError(f"{path}: invalid row at line {line_number}")
                yield value


def iter_pinout_row_claims(
    registry: Mapping[str, Any],
    row_dataset_root: Path,
    *,
    created_at: datetime,
    splits: Sequence[str] = ("train", "validation", "test"),
) -> Iterator[FactClaim]:
    verify_corpus_registry(registry)
    root = row_dataset_root.expanduser().resolve(strict=True)
    pdf_uris = _pdf_uris(registry)
    for example in _row_records(root, splits):
        provenance = example.get("provenance") or {}
        alignment = example.get("alignment") or {}
        document_sha = provenance.get("pdf_sha256")
        if document_sha not in pdf_uris:
            raise ValueError(
                f"row example references a PDF outside the corpus: {document_sha}"
            )
        response = json.loads(str(example.get("response") or ""))
        pins = response.get("pins") if isinstance(response, dict) else None
        if not isinstance(pins, list) or not pins:
            raise ValueError(f"row example has invalid response: {example.get('example_id')}")
        images = example.get("images") or []
        image_hashes = example.get("image_sha256") or []
        if len(images) != 2 or len(image_hashes) != 2:
            raise ValueError("row example must include header and body image receipts")
        body_path = (root / str(images[1])).resolve()
        if not body_path.is_relative_to(root):
            raise ValueError("row image escapes dataset root")
        if sha256_file(body_path) != image_hashes[1]:
            raise ValueError(f"row image hash mismatch: {body_path}")
        batch_id = provenance.get("frontier_batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError("row example is missing frontier batch identity")
        request_sha = hashlib.sha256(
            canonical_json(
                {
                    "prompt": example.get("prompt"),
                    "image_sha256": image_hashes,
                }
            )
        ).hexdigest()
        model = ModelIdentity(
            provider="anthropic",
            model="claude-sonnet-5",
            request_sha256=request_sha,
            batch_id=batch_id,
        )
        bbox = alignment.get("body_bbox")
        evidence = (
            EvidenceReference(
                kind=EvidenceKind.IMAGE_REGION,
                document_sha256=document_sha,
                source_uri=pdf_uris[document_sha],
                artifact_sha256=image_hashes[1],
                page_1based=alignment.get("page_1based"),
                bbox=tuple(bbox) if isinstance(bbox, list) else bbox,
            ),
        )
        package = str(alignment.get("package") or "").strip()
        if not package:
            raise ValueError("row example is missing exact package identity")
        for pin in pins:
            if not isinstance(pin, Mapping):
                raise ValueError("row example pin is not an object")
            pin_no = pin.get("pin_no")
            pin_name = pin.get("name")
            if not _valid_value(pin_no) or not _valid_value(pin_name):
                raise ValueError("row example pin has invalid physical identity")
            record_id = str(example.get("record_id") or "")
            entity = EntityReference(
                entity_id=(
                    f"pin:{_safe_component(record_id)}:"
                    f"{hashlib.sha256(package.encode()).hexdigest()[:12]}:"
                    f"{_safe_component(pin_no)}"
                ),
                grain=EntityGrain.PIN_OR_BALL,
                canonical_id=str(pin_no),
                vendor=record_id.split("_", 1)[0] or None,
                family=record_id,
                package=package,
            )
            conditions = {
                "package": package,
                "source_split": example["split"],
                "source_example_id": example["example_id"],
            }
            for field, fact_value, claim_class in (
                ("pin.number", pin_no, ClaimClass.VISIBLE_FACT),
                ("pin.name", pin_name, ClaimClass.VISIBLE_FACT),
                ("pin.type", pin.get("type"), ClaimClass.SEMANTIC_LABEL),
                ("pin.functions", pin.get("functions"), ClaimClass.SEMANTIC_LABEL),
                ("pin.direction", pin.get("dir"), ClaimClass.SEMANTIC_LABEL),
            ):
                if not _valid_value(fact_value):
                    continue
                yield make_claim(
                    entity=entity,
                    field=field,
                    value=fact_value,
                    claim_class=claim_class,
                    extraction_method="frontier_vision_teacher",
                    evidence=evidence,
                    conditions=conditions,
                    model=model,
                    created_at=created_at,
                )


def build_extraction_work_queue(
    registry: Mapping[str, Any],
    *,
    aligned_document_sha256: Iterable[str],
) -> dict[str, Any]:
    verify_corpus_registry(registry)
    aligned = set(aligned_document_sha256)
    rows: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    for document in registry["documents"]:
        digest = document["document_sha256"]
        has_gt = bool(document.get("ground_truth"))
        gt_sections = {
            section
            for record in document.get("ground_truth") or []
            for section in record.get("sections") or []
        }
        if digest in aligned:
            pin_route = "owned_exact_vision_pair"
        elif has_gt:
            pin_route = "pymupdf_localize_then_local_vision"
        else:
            pin_route = "pymupdf_localize_then_local_text_or_vision"
        semantic_route = (
            "existing_ground_truth_then_page_verify"
            if has_gt
            else "pymupdf_then_local_text"
        )
        parametric_route = (
            "existing_ground_truth_then_page_verify"
            if gt_sections
            & {"interface_atoms", "peripheral_depth", "clock_specs", "abs_max_ratings", "power_modes"}
            else "pymupdf_then_local_text"
        )
        summary_route = (
            "existing_ground_truth_then_page_verify"
            if "overview" in gt_sections
            else "pymupdf_then_local_text"
        )
        routes = {
            "pin_or_ball": pin_route,
            "pin_semantics": semantic_route,
            "parametrics": parametric_route,
            "series_summary": summary_route,
        }
        route_counts.update(routes.values())
        rows.append(
            {
                "document_sha256": digest,
                "paths": document.get("paths") or [],
                "vendors": document.get("vendors") or [],
                "record_ids": document.get("record_ids") or [],
                "routes": routes,
                "frontier_batch_eligibility": {
                    "eligible": False,
                    "required_failures": [
                        "deterministic_extractor_failed",
                        "local_model_failed_or_disagreed",
                    ],
                    "required_evidence": [
                        "source_document_sha256",
                        "page_or_region_receipt",
                        "local_attempt_receipt",
                    ],
                },
            }
        )
    core = {
        "schema": WORK_QUEUE_SCHEMA,
        "policy": {
            "ordering": [
                "existing_verified_facts",
                "pymupdf",
                "local_text_or_vision",
                "anthropic_message_batch",
            ],
            "frontier_purpose": "local_training_pair_generation",
            "direct_database_writes": False,
        },
        "corpus_evidence_sha256": registry["evidence_sha256"],
        "counts": {
            "documents": len(rows),
            "routes": dict(sorted(route_counts.items())),
        },
        "work": rows,
    }
    core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
    return core


__all__ = [
    "WORK_QUEUE_SCHEMA",
    "build_extraction_work_queue",
    "iter_ground_truth_claims",
    "iter_pinout_row_claims",
]
