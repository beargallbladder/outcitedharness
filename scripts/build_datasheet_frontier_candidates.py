#!/usr/bin/env python3
"""Derive teacher candidates for every completed sealed local extraction."""

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
from typing import Any

from harness.electronics.claims import canonical_json
from harness.electronics.corpus import sha256_file
from harness.electronics.frontier_batch import (
    COMPLETED_LOCAL_STATUSES,
    FrontierCandidate,
    FrontierEvidence,
    LocalAttempt,
    candidate_id,
    candidate_identity_payload,
)
from harness.electronics.local_model import RESPONSE_SCHEMAS, local_prompt
from harness.electronics.models import PairCapability


QUEUE_SCHEMA = "harness.electronics-structural-local-work.v1"
LOCAL_SCHEMA = "harness.electronics-structural-local-extraction.v1"
OUTPUT_SCHEMA = "harness.electronics-frontier-candidate-cohort.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _queue_core(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "evidence_sha256"}
    }


def _verify_artifact(bundle: Path, receipt: dict[str, Any], name: str) -> Path:
    path = bundle / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"bundle artifact is missing or unsafe: {path}")
    if path.stat().st_size != receipt.get("bytes") or _sha256(path) != receipt.get(
        "sha256"
    ):
        raise ValueError(f"bundle artifact hash mismatch: {path}")
    return path


def _evidence_bbox(item: dict[str, Any]) -> tuple[float, float, float, float]:
    bboxes = [
        region["bbox"] for region in item["structural_evidence"]["regions"]
    ]
    if not bboxes:
        raise ValueError(f"work item has no structural region: {item['work_id']}")
    return (
        min(float(bbox[0]) for bbox in bboxes),
        min(float(bbox[1]) for bbox in bboxes),
        max(float(bbox[2]) for bbox in bboxes),
        max(float(bbox[3]) for bbox in bboxes),
    )


def _candidate(
    item: dict[str, Any],
    page: dict[str, Any],
    image_path: Path,
    image_sha256: str,
    attempts: list[dict[str, Any]],
) -> FrontierCandidate:
    completed = [
        LocalAttempt(
            provider="local",
            model=str(attempt["model"]),
            status=attempt["status"],
            receipt_sha256=attempt["receipt_sha256"],
            output_sha256=attempt.get("response_sha256"),
            reason=(
                attempt.get("reason")
                or "local output passed the source-evidence gate"
            ),
        )
        for attempt in attempts
        if attempt.get("status") in COMPLETED_LOCAL_STATUSES
    ]
    if not completed:
        raise ValueError(f"work item has no completed attempt: {item['work_id']}")
    prompt = local_prompt(
        item["capability"],
        page,
        include_page_evidence=False,
    )
    draft = FrontierCandidate(
        candidate_id="candidate-" + ("0" * 32),
        capability=PairCapability(item["capability"]),
        document_sha256=item["document_sha256"],
        entity_hint=item["work_id"],
        prompt=prompt,
        response_schema=RESPONSE_SCHEMAS[item["capability"]],
        evidence=(
            FrontierEvidence(
                path=image_path,
                sha256=image_sha256,
                media_type="image/png",
                page_1based=item["page_1based"],
                bbox=_evidence_bbox(item),
            ),
        ),
        local_attempts=tuple(completed),
        estimated_input_tokens=len(prompt) // 4 + 1600,
        max_output_tokens=8192,
    )
    return draft.model_copy(
        update={"candidate_id": candidate_id(candidate_identity_payload(draft))}
    )


def _write(path: Path, payload: bytes) -> dict[str, Any]:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    return {"sha256": _sha256(path), "bytes": path.stat().st_size}


def build(
    work_queue: Path,
    local_bundles: list[Path],
    output_directory: Path,
    require_complete: bool,
) -> dict[str, Any]:
    queue_path = work_queue.expanduser().resolve(strict=True)
    queue = _json(queue_path)
    if queue.get("schema") != QUEUE_SCHEMA:
        raise ValueError("unsupported structural work queue")
    if hashlib.sha256(canonical_json(_queue_core(queue))).hexdigest() != queue.get(
        "evidence_sha256"
    ):
        raise ValueError("structural work queue evidence hash is invalid")
    ordered_work = list(queue.get("work") or [])
    work_by_id = {item["work_id"]: item for item in ordered_work}
    if len(work_by_id) != len(ordered_work):
        raise ValueError("structural work queue contains duplicate work IDs")

    attempts_by_work: dict[str, list[dict[str, Any]]] = defaultdict(list)
    results_by_work: dict[str, dict[str, Any]] = {}
    pages_by_work: dict[str, dict[str, Any]] = {}
    image_by_work: dict[str, tuple[Path, str]] = {}
    source_receipts: list[dict[str, Any]] = []
    covered: set[str] = set()

    for source in local_bundles:
        bundle = source.expanduser().resolve(strict=True)
        manifest_path = bundle / "manifest.json"
        manifest = _json(manifest_path)
        if manifest.get("schema") != LOCAL_SCHEMA:
            raise ValueError(f"unsupported local extraction bundle: {bundle}")
        if manifest.get("sources", {}).get(
            "structural_queue_sha256"
        ) != _sha256(queue_path):
            raise ValueError(f"local bundle used a different work queue: {bundle}")
        artifacts = manifest.get("artifacts") or {}
        required = (
            "attempts.jsonl",
            "local-results.jsonl",
            "pillar-evidence.jsonl",
        )
        paths = {
            name: _verify_artifact(bundle, artifacts[name], name)
            for name in required
        }
        selection = manifest.get("selection") or {}
        offset = int(selection["offset"])
        count = int(selection["work_items"])
        expected_ids = {
            item["work_id"] for item in ordered_work[offset : offset + count]
        }
        if len(expected_ids) != count:
            raise ValueError(f"invalid local bundle selection: {bundle}")
        overlap = covered.intersection(expected_ids)
        if overlap:
            raise ValueError(f"local bundles overlap work IDs: {sorted(overlap)[:3]}")
        covered.update(expected_ids)

        attempts = _jsonl(paths["attempts.jsonl"])
        results = _jsonl(paths["local-results.jsonl"])
        pillars = _jsonl(paths["pillar-evidence.jsonl"])
        pillar_map = {row["work_id"]: row["page"] for row in pillars}
        if len(pillar_map) != len(pillars) or set(pillar_map) != expected_ids:
            raise ValueError(f"pillar evidence does not match selection: {bundle}")
        for row in attempts:
            work_id = row["work_id"]
            if work_id not in expected_ids:
                raise ValueError(f"attempt escapes bundle selection: {work_id}")
            attempts_by_work[work_id].append(row)
        for row in results:
            work_id = row["work_id"]
            if work_id not in expected_ids or work_id in results_by_work:
                raise ValueError(f"invalid or duplicate local result: {work_id}")
            results_by_work[work_id] = row
        pages_by_work.update(pillar_map)

        image_receipts = manifest.get("images") or {}
        images_by_sha: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(
            list
        )
        for name, receipt in image_receipts.items():
            images_by_sha[str(receipt["sha256"])].append((name, receipt))
        for work_id in expected_ids:
            result = results_by_work.get(work_id)
            image_sha = (
                result.get("evidence_image_sha256") if result is not None else None
            )
            if image_sha is None:
                image_sha = next(
                    (
                        attempt.get("image_sha256")
                        for attempt in reversed(attempts_by_work[work_id])
                        if attempt.get("image_sha256")
                    ),
                    None,
                )
            completed = [
                attempt
                for attempt in attempts_by_work[work_id]
                if attempt.get("status") in COMPLETED_LOCAL_STATUSES
            ]
            if not completed:
                continue
            matches = images_by_sha.get(str(image_sha), [])
            if len(matches) > 1:
                suffix = f"-{work_id[-8:]}-focused.png"
                matches = [
                    match for match in matches if match[0].endswith(suffix)
                ]
            if len(matches) != 1:
                raise ValueError(f"work item image is not unique: {work_id}")
            image_name, image_receipt = matches[0]
            image_path = _verify_artifact(
                bundle, image_receipt, f"images/{image_name}"
            )
            image_by_work[work_id] = (image_path, str(image_sha))

        source_receipts.append(
            {
                "path": str(bundle),
                "manifest_sha256": _sha256(manifest_path),
                "offset": offset,
                "work_items": count,
            }
        )

    if require_complete and covered != set(work_by_id):
        missing = set(work_by_id).difference(covered)
        raise ValueError(f"local bundles do not cover the queue: {len(missing)} missing")
    if set(pages_by_work) != covered:
        raise ValueError("merged pillar evidence does not match covered work")

    candidates: list[FrontierCandidate] = []
    withheld: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for item in ordered_work:
        work_id = item["work_id"]
        if work_id not in covered:
            continue
        completed = [
            attempt
            for attempt in attempts_by_work[work_id]
            if attempt.get("status") in COMPLETED_LOCAL_STATUSES
        ]
        if not completed:
            withheld.append(
                {
                    "work_id": work_id,
                    "reason": "no_completed_local_model_attempt",
                    "attempt_statuses": [
                        attempt.get("status")
                        for attempt in attempts_by_work[work_id]
                    ],
                }
            )
            counts["withheld:no_completed_local_model_attempt"] += 1
            continue
        image_path, image_sha = image_by_work[work_id]
        candidate = _candidate(
            item,
            pages_by_work[work_id],
            image_path,
            image_sha,
            attempts_by_work[work_id],
        )
        candidates.append(candidate)
        counts[
            "candidate:with_parseable_local_result"
            if work_id in results_by_work
            else "candidate:without_parseable_local_result"
        ] += 1
        for attempt in completed:
            counts[f"attempt:{attempt['status']}"] += 1
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("derived frontier candidates are not unique")

    output = output_directory.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        merged_results = [
            results_by_work[item["work_id"]]
            for item in ordered_work
            if item["work_id"] in results_by_work
        ]
        merged_pillars = [
            {
                "schema": "harness.electronics-local-pillar-evidence.v1",
                "work_id": item["work_id"],
                "page": pages_by_work[item["work_id"]],
            }
            for item in ordered_work
            if item["work_id"] in covered
        ]
        merged_attempts = [
            attempt
            for item in ordered_work
            for attempt in attempts_by_work[item["work_id"]]
            if item["work_id"] in covered
        ]
        payloads = {
            "candidates.jsonl": b"".join(
                canonical_json(
                    candidate.model_dump(mode="json", by_alias=True)
                )
                + b"\n"
                for candidate in candidates
            ),
            "local-results.jsonl": b"".join(
                canonical_json(row) + b"\n" for row in merged_results
            ),
            "pillar-evidence.jsonl": b"".join(
                canonical_json(row) + b"\n" for row in merged_pillars
            ),
            "attempts.jsonl": b"".join(
                canonical_json(row) + b"\n" for row in merged_attempts
            ),
            "withheld.jsonl": b"".join(
                canonical_json(row) + b"\n" for row in withheld
            ),
        }
        artifact_receipts = {
            name: _write(temporary / name, payload)
            for name, payload in payloads.items()
        }
        core = {
            "schema": OUTPUT_SCHEMA,
            "purpose": "teacher_for_every_completed_local_output",
            "counts": {
                "covered_work": len(covered),
                "local_results": len(merged_results),
                "candidates": len(candidates),
                "withheld": len(withheld),
                **dict(sorted(counts.items())),
            },
            "sources": {
                "work_queue_path": str(queue_path),
                "work_queue_sha256": _sha256(queue_path),
                "work_queue_evidence_sha256": queue["evidence_sha256"],
                "local_bundles": source_receipts,
            },
            "artifacts": artifact_receipts,
        }
        core["evidence_sha256"] = hashlib.sha256(
            canonical_json(core)
        ).hexdigest()
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **core,
        }
        _write(
            temporary / "manifest.json",
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode()
            + b"\n",
        )
        os.chmod(temporary, 0o555)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-queue", type=Path, required=True)
    parser.add_argument(
        "--local-bundle", type=Path, action="append", required=True
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    manifest = build(
        args.work_queue,
        args.local_bundle,
        args.output_directory,
        args.require_complete,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
