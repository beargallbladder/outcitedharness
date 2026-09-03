#!/usr/bin/env python3
"""Verify local datasheet output and route weak results to teacher batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from harness.electronics.claims import canonical_json
from harness.electronics.frontier_batch import (
    FrontierCandidate,
    FrontierEvidence,
    LocalAttempt,
    candidate_id,
    candidate_identity_payload,
)
from harness.electronics.ground_truth import (
    load_ground_truth_records,
    rows_for_package,
)
from harness.electronics.local_model import RESPONSE_SCHEMAS, local_prompt
from harness.electronics.local_verification import (
    verify_opn_decoder,
    verify_parametrics,
    verify_pin_or_ball,
    verify_pin_semantics,
    verify_series_summary,
)
from harness.electronics.models import PairCapability


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} is not an object")
            yield value


def _verify_artifact(
    root: Path,
    name: str,
    receipt: dict[str, Any],
) -> Path:
    path = root / name
    artifact = receipt["artifacts"][name]
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != artifact["bytes"]
        or _sha256(path) != artifact["sha256"]
    ):
        raise ValueError(f"local artifact differs from manifest: {name}")
    return path


def _write(path: Path, payload: bytes) -> dict[str, Any]:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-bundle", type=Path, required=True)
    parser.add_argument("--priority-queue", type=Path, required=True)
    parser.add_argument("--page-evidence", type=Path, required=True)
    parser.add_argument("--corpus-registry", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument(
        "--frontier-policy",
        choices=("all", "failed-only"),
        default="all",
    )
    parser.add_argument(
        "--exclude-frontier-candidates",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    local_root = args.local_bundle.expanduser().resolve(strict=True)
    local_manifest = json.loads(
        (local_root / "manifest.json").read_text(encoding="utf-8")
    )
    local_results_path = _verify_artifact(
        local_root,
        "local-results.jsonl",
        local_manifest,
    )
    pillar_pages: dict[str, dict[str, Any]] = {}
    if "pillar-evidence.jsonl" in local_manifest.get("artifacts", {}):
        pillar_path = _verify_artifact(
            local_root,
            "pillar-evidence.jsonl",
            local_manifest,
        )
        pillar_pages = {
            value["work_id"]: value["page"]
            for value in _jsonl(pillar_path)
        }
    priority_path = args.priority_queue.expanduser().resolve(strict=True)
    page_root = args.page_evidence.expanduser().resolve(strict=True)
    corpus_path = args.corpus_registry.expanduser().resolve(strict=True)
    gt_root = args.ground_truth_root.expanduser().resolve(strict=True)
    priority = json.loads(priority_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    work = {item["work_id"]: item for item in priority["work"]}
    pages = {
        (item["document_sha256"], int(item["page_1based"])): item
        for item in _jsonl(page_root / "page-evidence.jsonl")
    }
    ground_truth = load_ground_truth_records(corpus, gt_root)
    image_by_sha = {
        value["sha256"]: local_root / "images" / name
        for name, value in local_manifest["images"].items()
    }
    excluded_candidate_ids = {
        value["candidate_id"]
        for path in args.exclude_frontier_candidates
        for value in _jsonl(path.expanduser().resolve(strict=True))
    }

    verifications: list[dict[str, Any]] = []
    passed: list[dict[str, Any]] = []
    candidates: list[FrontierCandidate] = []
    for result in _jsonl(local_results_path):
        item = work.get(result["work_id"])
        if item is None:
            raise ValueError(f"local result has unknown work ID: {result['work_id']}")
        page = pillar_pages.get(result["work_id"]) or pages.get(
            (item["document_sha256"], int(item["page_1based"]))
        )
        if page is None or page["evidence_sha256"] != item["page_evidence_sha256"]:
            raise ValueError(f"page evidence is unavailable: {result['work_id']}")
        package_scope = (
            item.get("structural_evidence", {}).get("package_scope")
            or {}
        )
        if item["capability"] == "pin_or_ball":
            verdict = verify_pin_or_ball(
                result,
                page,
                rows_for_package(
                    ground_truth.get(item["document_sha256"], []),
                    package_scope.get("package"),
                ),
            )
        elif item["capability"] == "pin_semantics":
            verdict = verify_pin_semantics(
                result,
                page,
                rows_for_package(
                    ground_truth.get(item["document_sha256"], []),
                    package_scope.get("package"),
                ),
            )
        elif item["capability"] == "series_summary":
            verdict = verify_series_summary(result, page)
        elif item["capability"] == "opn_decoder":
            verdict = verify_opn_decoder(result, page)
        elif item["capability"] == "parametrics":
            verdict = verify_parametrics(result, page)
        else:
            raise ValueError(
                f"unsupported local verification lane: {item['capability']}"
            )
        evidence_image_sha256 = (
            result.get("image_sha256")
            or result.get("evidence_image_sha256")
        )
        if not isinstance(evidence_image_sha256, str):
            raise ValueError(
                f"local result has no evidence image: {result['work_id']}"
            )
        core = {
            "schema": "harness.electronics-local-verification.v1",
            "work_id": result["work_id"],
            "document_sha256": item["document_sha256"],
            "page_1based": item["page_1based"],
            "capability": item["capability"],
            "status": (
                "passed_pending_claim_materialization"
                if verdict.passed
                else "failed_frontier_eligible"
            ),
            "terminal_status": verdict.terminal_status,
            "reason": verdict.reason,
            "checks": list(verdict.checks),
            "metrics": verdict.metrics,
            "response_sha256": result["response_sha256"],
            "page_evidence_sha256": page["evidence_sha256"],
            "image_sha256": evidence_image_sha256,
        }
        verification_sha = hashlib.sha256(canonical_json(core)).hexdigest()
        verification = {
            **core,
            "verification_id": f"local-verify-{verification_sha[:32]}",
            "receipt_sha256": verification_sha,
        }
        verifications.append(verification)
        if verdict.passed:
            passed.append(
                {
                    "schema": "harness.electronics-verified-local-result.v1",
                    "verification_id": verification["verification_id"],
                    "verification_receipt_sha256": verification_sha,
                    "work": item,
                    "local_result": result,
                }
            )
            if args.frontier_policy == "failed-only":
                continue

        image_path = image_by_sha.get(evidence_image_sha256)
        if image_path is None or _sha256(image_path) != evidence_image_sha256:
            raise ValueError(f"local image is unavailable: {result['work_id']}")
        attempt = LocalAttempt(
            provider="local",
            model=result["model"],
            status=verdict.terminal_status or "low_confidence",
            receipt_sha256=verification_sha,
            output_sha256=result["response_sha256"],
            reason=verdict.reason
            or (
                "bootstrap teacher comparison is required before "
                "claim or training admission"
            ),
        )
        prompt = local_prompt(item["capability"], page)
        regions = (
            item.get("structural_evidence", {}).get("regions") or []
        )
        evidence_bbox = (
            (
                min(float(region["bbox"][0]) for region in regions),
                min(float(region["bbox"][1]) for region in regions),
                max(float(region["bbox"][2]) for region in regions),
                max(float(region["bbox"][3]) for region in regions),
            )
            if regions
            else None
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
                    sha256=evidence_image_sha256,
                    media_type="image/png",
                    page_1based=item["page_1based"],
                    bbox=evidence_bbox,
                ),
            ),
            local_attempts=(attempt,),
            estimated_input_tokens=len(prompt) // 4 + 1600,
            max_output_tokens=8192,
        )
        candidate = draft.model_copy(
            update={
                "candidate_id": candidate_id(
                    candidate_identity_payload(draft)
                )
            }
        )
        if candidate.candidate_id not in excluded_candidate_ids:
            candidates.append(candidate)

    output = args.output_directory.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        payloads = {
            "verifications.jsonl": b"".join(
                canonical_json(value) + b"\n" for value in verifications
            ),
            "verified-local-results.jsonl": b"".join(
                canonical_json(value) + b"\n" for value in passed
            ),
            "frontier-candidates.jsonl": b"".join(
                canonical_json(
                    value.model_dump(mode="json", by_alias=True)
                )
                + b"\n"
                for value in candidates
            ),
        }
        artifacts = {
            name: _write(temporary / name, payload)
            for name, payload in payloads.items()
        }
        core = {
            "schema": "harness.electronics-local-verification-bundle.v1",
            "sources": {
                "local_bundle_evidence_sha256": local_manifest[
                    "evidence_sha256"
                ],
                "priority_queue_sha256": _sha256(priority_path),
                "page_evidence_sha256": _sha256(
                    page_root / "manifest.json"
                ),
                "corpus_registry_sha256": _sha256(corpus_path),
            },
            "counts": {
                "evaluated": len(verifications),
                "passed_pending_claim_materialization": len(passed),
                "frontier_candidates": len(candidates),
                "excluded_frontier_candidates": len(excluded_candidate_ids),
            },
            "policy": {
                "frontier": args.frontier_policy,
                "local_extraction_required_before_frontier": True,
                "teacher_outputs_require_source_grounding": True,
            },
            "artifacts": artifacts,
        }
        core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
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
            ).encode("utf-8")
            + b"\n",
        )
        os.chmod(temporary, 0o555)
        os.rename(temporary, output)
    except BaseException:
        os.chmod(temporary, 0o755)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
