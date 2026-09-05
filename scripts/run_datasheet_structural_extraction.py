#!/usr/bin/env python3
"""Run text-first extraction with focused vision only when text is insufficient."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

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
from harness.electronics.local_model import (
    LocalExtractionClient,
    RESPONSE_SCHEMAS,
    local_prompt,
)
from harness.electronics.local_verification import (
    verify_opn_decoder,
    verify_parametrics,
    verify_pin_or_ball,
    verify_pin_semantics,
    verify_series_summary,
)
from harness.electronics.models import PairCapability
from harness.electronics.regions import (
    pdftotext_layout_page,
    render_focused_regions,
    render_full_page,
)
from harness.electronics.table_extractors import (
    normalize_parametric_facts,
    parse_parametric_table,
)


SCHEMA = "harness.electronics-structural-local-extraction.v1"
DETERMINISTIC_PARAMETRIC_MODEL = "pymupdf-parametric-normalizer-v1"


def _verify(
    capability: str,
    result: dict[str, Any],
    page: dict[str, Any],
):
    if capability == "pin_or_ball":
        return verify_pin_or_ball(result, page)
    if capability == "pin_semantics":
        return verify_pin_semantics(result, page)
    if capability == "parametrics":
        return verify_parametrics(result, page)
    if capability == "series_summary":
        return verify_series_summary(result, page)
    if capability == "opn_decoder":
        return verify_opn_decoder(result, page)
    raise ValueError(f"unsupported structural capability: {capability}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verified_core(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "evidence_sha256"}
    }


def _terminal_status(exc: BaseException) -> str:
    return "no_answer" if "no JSON object" in str(exc) else "schema_failed"


def _deterministic_parametric_result(
    item: dict[str, Any],
    page: dict[str, Any],
) -> dict[str, Any]:
    selected_tables = {
        int(region["table_index"])
        for region in item["structural_evidence"]["regions"]
        if region.get("table_index") is not None
    }
    facts_by_identity: dict[bytes, dict[str, Any]] = {}
    for table in page.get("tables") or []:
        table_index = table.get("table_index")
        if selected_tables and table_index not in selected_tables:
            continue
        rows = parse_parametric_table(
            table,
            document_sha256=item["document_sha256"],
            page_1based=int(item["page_1based"]),
        )
        for row in rows:
            for fact in normalize_parametric_facts(row):
                facts_by_identity[canonical_json(fact)] = fact
    parsed = {
        "facts": [
            facts_by_identity[identity]
            for identity in sorted(facts_by_identity)
        ]
    }
    request = {
        "model": DETERMINISTIC_PARAMETRIC_MODEL,
        "capability": "parametrics",
        "document_sha256": item["document_sha256"],
        "page_1based": item["page_1based"],
        "page_evidence_sha256": item["page_evidence_sha256"],
        "structural_evidence": item["structural_evidence"],
    }
    return {
        "schema": "harness.electronics-local-model-result.v1",
        "provider": "local",
        "model": DETERMINISTIC_PARAMETRIC_MODEL,
        "capability": "parametrics",
        "request_sha256": hashlib.sha256(canonical_json(request)).hexdigest(),
        "response_sha256": hashlib.sha256(canonical_json(parsed)).hexdigest(),
        "image_sha256": None,
        "latency_ms": 0.0,
        "usage": None,
        "result": parsed,
    }


def _attempt(
    *,
    item: dict[str, Any],
    model: str,
    stage: str,
    status: str,
    image_sha256: str | None,
    reason: str | None = None,
    result: dict[str, Any] | None = None,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "schema": "harness.electronics-local-pillar-attempt.v1",
        "work_id": item["work_id"],
        "provider": "local",
        "model": model,
        "stage": stage,
        "status": status,
        "capability": item["capability"],
        "document_sha256": item["document_sha256"],
        "page_1based": item["page_1based"],
        "page_evidence_sha256": item["page_evidence_sha256"],
        "image_sha256": image_sha256,
        "reason": reason,
        "request_sha256": result.get("request_sha256") if result else None,
        "response_sha256": result.get("response_sha256") if result else None,
        "verification": verification,
    }
    value["receipt_sha256"] = hashlib.sha256(canonical_json(value)).hexdigest()
    return value


def _frontier_candidate(
    *,
    item: dict[str, Any],
    page: dict[str, Any],
    model: str,
    image_path: Path,
    image_sha256: str,
    attempts: list[dict[str, Any]],
) -> FrontierCandidate:
    local_attempts = []
    for attempt in attempts:
        if attempt["status"] not in COMPLETED_LOCAL_STATUSES:
            continue
        local_attempts.append(
            LocalAttempt(
                provider="local",
                model=attempt["model"],
                status=attempt["status"],
                receipt_sha256=attempt["receipt_sha256"],
                output_sha256=attempt.get("response_sha256"),
                reason=(
                    attempt["reason"]
                    or "local output passed the source-evidence gate"
                ),
            )
        )
    if not local_attempts:
        raise ValueError("frontier candidate has no terminal local attempt")
    bboxes = [
        region["bbox"]
        for region in item["structural_evidence"]["regions"]
    ]
    evidence_bbox = (
        min(float(bbox[0]) for bbox in bboxes),
        min(float(bbox[1]) for bbox in bboxes),
        max(float(bbox[2]) for bbox in bboxes),
        max(float(bbox[3]) for bbox in bboxes),
    )
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
                bbox=evidence_bbox,
            ),
        ),
        local_attempts=tuple(local_attempts),
        estimated_input_tokens=len(prompt) // 4 + 1600,
        max_output_tokens=8192,
    )
    return draft.model_copy(
        update={
            "candidate_id": candidate_id(candidate_identity_payload(draft))
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-queue", type=Path, required=True)
    parser.add_argument("--page-evidence", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--render-dpi", type=int, default=180)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument(
        "--vision-policy",
        choices=("fallback", "always"),
        default="fallback",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.offset < 0 or not 1 <= args.limit <= 1000:
        raise ValueError("offset must be non-negative and limit within 1..1000")
    queue_path = args.structural_queue.expanduser().resolve(strict=True)
    queue = _load_json(queue_path)
    if queue.get("schema") != "harness.electronics-structural-local-work.v1":
        raise ValueError("structural queue schema is not supported")
    if hashlib.sha256(canonical_json(_verified_core(queue))).hexdigest() != queue.get(
        "evidence_sha256"
    ):
        raise ValueError("structural queue evidence digest is invalid")
    work = queue["work"][args.offset : args.offset + args.limit]
    if not work:
        raise ValueError("structural extraction selected no work")
    if any(
        item["structural_evidence"]["mode"]
        not in {
            "focused_ordering_evidence",
            "focused_parametric_table",
            "focused_structural_table",
            "focused_summary_evidence",
        }
        for item in work
    ):
        raise ValueError(
            "non-table work requires a separately gated extraction lane"
        )

    evidence_root = args.page_evidence.expanduser().resolve(strict=True)
    evidence_manifest = _load_json(evidence_root / "manifest.json")
    page_path = evidence_root / "page-evidence.jsonl"
    page_receipt = evidence_manifest["artifacts"]["page-evidence.jsonl"]
    if (
        sha256_file(page_path) != page_receipt["sha256"]
        or page_path.stat().st_size != page_receipt["bytes"]
    ):
        raise ValueError("page evidence differs from its manifest")
    wanted = {
        (item["document_sha256"], int(item["page_1based"])) for item in work
    }
    pages = {}
    with page_path.open(encoding="utf-8") as handle:
        for line in handle:
            page = json.loads(line)
            key = (page["document_sha256"], int(page["page_1based"]))
            if key in wanted:
                pages[key] = page
    if set(pages) != wanted:
        raise ValueError("structural work references missing page evidence")

    output = args.output_directory.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    images = temporary / "images"
    images.mkdir()
    implementation = temporary / "implementation"
    implementation.mkdir()
    repository = Path(__file__).resolve().parents[1]
    implementation_receipts = {}
    for relative in (
        "pyproject.toml",
        "uv.lock",
        "harness/electronics/claims.py",
        "harness/electronics/corpus.py",
        "harness/electronics/frontier_batch.py",
        "harness/electronics/local_model.py",
        "harness/electronics/local_verification.py",
        "harness/electronics/locator.py",
        "harness/electronics/models.py",
        "harness/electronics/regions.py",
        "harness/electronics/table_extractors.py",
        "scripts/build_datasheet_structural_work_queue.py",
        "scripts/run_datasheet_structural_extraction.py",
    ):
        source = repository / relative
        name = relative.replace("/", "__")
        destination = implementation / name
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o444)
        implementation_receipts[relative] = {
            "path": f"implementation/{name}",
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        }
    client = LocalExtractionClient(
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout_seconds,
    )
    attempts: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    candidates: list[FrontierCandidate] = []
    pillar_evidence: list[dict[str, Any]] = []
    render_receipts: dict[str, Any] = {}
    counts: Counter[str] = Counter()
    verified_documents: set[str] = set()
    try:
        for index, item in enumerate(work, 1):
            source_path = Path(item["source_path"]).expanduser().resolve(strict=True)
            if item["document_sha256"] not in verified_documents:
                if sha256_file(source_path) != item["document_sha256"]:
                    raise ValueError("source PDF changed after corpus seal")
                verified_documents.add(item["document_sha256"])
            key = (item["document_sha256"], int(item["page_1based"]))
            page = {
                **pages[key],
                "structural_evidence": item["structural_evidence"],
                "digital_text": {
                    "available": False,
                    "error": "not_run: deterministic PyMuPDF facts passed first",
                    "extractor": "pdftotext-layout",
                    "sha256": hashlib.sha256(b"").hexdigest(),
                    "text": "",
                    "truncated": False,
                },
            }
            deterministic_result: dict[str, Any] | None = None
            deterministic_verdict = None
            if item["capability"] == "parametrics":
                deterministic_result = _deterministic_parametric_result(item, page)
                deterministic_verdict = _verify(
                    item["capability"],
                    deterministic_result,
                    page,
                )
            if deterministic_verdict is None or not deterministic_verdict.passed:
                page["digital_text"] = pdftotext_layout_page(
                    source_path,
                    int(item["page_1based"]),
                )
            pillar_evidence.append(
                {
                    "schema": "harness.electronics-local-pillar-evidence.v1",
                    "work_id": item["work_id"],
                    "page": page,
                }
            )
            image_name = (
                f"{item['document_sha256'][:16]}-"
                f"p{int(item['page_1based']):05d}-"
                f"{item['work_id'][-8:]}-focused.png"
            )
            temporary_image = images / image_name
            package_scope = item["structural_evidence"].get("package_scope")
            if item["capability"] in {"pin_or_ball", "pin_semantics"}:
                # An exact package identity and expected pin count are always
                # required. The column header may be absent only for the
                # single-package borderless-table locator path, where there is
                # no per-package column to project; the extracted_n ==
                # package_n gate still applies at verification.
                if (
                    not package_scope
                    or not package_scope.get("package")
                    or package_scope.get("expected_package_pins") is None
                ):
                    raise ValueError(
                        "pin vision requires an exact package and count"
                    )
                rendering = render_full_page(
                    source_path,
                    int(item["page_1based"]),
                    temporary_image,
                    dpi=120,
                )
            else:
                rendering = render_focused_regions(
                    source_path,
                    int(item["page_1based"]),
                    item["structural_evidence"]["regions"],
                    temporary_image,
                    dpi=args.render_dpi,
                )
            image_sha = sha256_file(temporary_image)
            render_receipts[image_name] = {
                **rendering,
                "sha256": image_sha,
                "bytes": temporary_image.stat().st_size,
            }

            item_attempts: list[dict[str, Any]] = []
            deterministic_attempt: dict[str, Any] | None = None
            text_result: dict[str, Any] | None = None
            text_verdict = None
            text_attempt: dict[str, Any] | None = None
            if deterministic_result is not None:
                if deterministic_verdict is None:
                    raise ValueError("deterministic preflight has no verdict")
                deterministic_attempt = _attempt(
                    item=item,
                    model=deterministic_result["model"],
                    stage="deterministic_structure",
                    status=(
                        "passed_evidence_gate"
                        if deterministic_verdict.passed
                        else deterministic_verdict.terminal_status
                    ),
                    image_sha256=None,
                    reason=deterministic_verdict.reason,
                    result=deterministic_result,
                    verification={
                        "checks": list(deterministic_verdict.checks),
                        "metrics": deterministic_verdict.metrics,
                    },
                )
                attempts.append(deterministic_attempt)
                item_attempts.append(deterministic_attempt)
                counts[
                    f"deterministic:{deterministic_attempt['status']}"
                ] += 1
                if deterministic_verdict.passed:
                    text_result = deterministic_result
                    text_verdict = deterministic_verdict
                    text_attempt = deterministic_attempt

            pin_vision_required = item["capability"] in {
                "pin_or_ball",
                "pin_semantics",
            }
            if text_attempt is None and (
                args.vision_policy == "always" or pin_vision_required
            ):
                counts["text_generation_skipped_for_mandatory_vision"] += 1
            if (
                text_attempt is None
                and args.vision_policy == "fallback"
                and not pin_vision_required
            ):
                try:
                    text_result = client.extract(
                        capability=item["capability"],
                        page_evidence=page,
                    )
                    text_verdict = _verify(
                        item["capability"],
                        text_result,
                        page,
                    )
                    text_attempt = _attempt(
                        item=item,
                        model=args.model,
                        stage="local_text",
                        status=(
                            "passed_evidence_gate"
                            if text_verdict.passed
                            else text_verdict.terminal_status
                        ),
                        image_sha256=None,
                        reason=text_verdict.reason,
                        result=text_result,
                        verification={
                            "checks": list(text_verdict.checks),
                            "metrics": text_verdict.metrics,
                        },
                    )
                except (httpx.HTTPError, OSError, TimeoutError) as exc:
                    text_attempt = _attempt(
                        item=item,
                        model=args.model,
                        stage="local_text",
                        status="infrastructure_failed",
                        image_sha256=None,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                except (ValueError, json.JSONDecodeError) as exc:
                    text_attempt = _attempt(
                        item=item,
                        model=args.model,
                        stage="local_text",
                        status=_terminal_status(exc),
                        image_sha256=None,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                attempts.append(text_attempt)
                item_attempts.append(text_attempt)
                counts[f"text:{text_attempt['status']}"] += 1

            final_result = None
            final_stage = None
            vision_attempt = None
            if (
                text_verdict is not None
                and text_verdict.passed
                and args.vision_policy == "fallback"
            ):
                final_result = text_result
                final_stage = text_attempt["stage"]
                counts[f"vision_avoided_by_grounded_{final_stage}"] += 1
            else:
                try:
                    vision_result = client.extract(
                        capability=item["capability"],
                        page_evidence=page,
                        image_path=temporary_image,
                    )
                    vision_verdict = _verify(
                        item["capability"],
                        vision_result,
                        page,
                    )
                    vision_attempt = _attempt(
                        item=item,
                        model=args.model,
                        stage="focused_local_vision",
                        status=(
                            "passed_evidence_gate"
                            if vision_verdict.passed
                            else vision_verdict.terminal_status
                        ),
                        image_sha256=image_sha,
                        reason=vision_verdict.reason,
                        result=vision_result,
                        verification={
                            "checks": list(vision_verdict.checks),
                            "metrics": vision_verdict.metrics,
                        },
                    )
                    final_result = vision_result
                    final_stage = "focused_local_vision"
                except (httpx.HTTPError, OSError, TimeoutError) as exc:
                    vision_attempt = _attempt(
                        item=item,
                        model=args.model,
                        stage="focused_local_vision",
                        status="infrastructure_failed",
                        image_sha256=image_sha,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                except (ValueError, json.JSONDecodeError) as exc:
                    vision_attempt = _attempt(
                        item=item,
                        model=args.model,
                        stage="focused_local_vision",
                        status=_terminal_status(exc),
                        image_sha256=image_sha,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                attempts.append(vision_attempt)
                item_attempts.append(vision_attempt)
                counts[f"vision:{vision_attempt['status']}"] += 1
                if final_result is None and text_result is not None:
                    final_result = text_result
                    final_stage = "local_text_fallback"
                elif final_result is None and deterministic_result is not None:
                    final_result = deterministic_result
                    final_stage = "deterministic_structure_fallback"

            if final_result is not None:
                final_result = {
                    **final_result,
                    "work_id": item["work_id"],
                    "partition": item["partition"],
                    "page_evidence_sha256": item["page_evidence_sha256"],
                    "evidence_image_sha256": image_sha,
                    "local_pillar_stage": final_stage,
                    "attempt_receipts": [
                        attempt["receipt_sha256"]
                        for attempt in item_attempts
                    ],
                }
                results.append(final_result)
                counts[f"result:{final_stage}"] += 1
                final_image_path = output / "images" / image_name
                candidates.append(
                    _frontier_candidate(
                        item=item,
                        page=page,
                        model=args.model,
                        image_path=final_image_path,
                        image_sha256=image_sha,
                        attempts=item_attempts,
                    )
                )
                counts[
                    "frontier_candidates_with_parseable_local_result"
                ] += 1
            elif any(
                attempt["status"] in {"no_answer", "schema_failed"}
                for attempt in item_attempts
            ):
                final_image_path = output / "images" / image_name
                candidates.append(
                    _frontier_candidate(
                        item=item,
                        page=page,
                        model=args.model,
                        image_path=final_image_path,
                        image_sha256=image_sha,
                        attempts=item_attempts,
                    )
                )
                counts["frontier_candidates_without_parseable_local_result"] += 1
            else:
                counts["infrastructure_retry_required"] += 1
            print(f"structural extraction {index}/{len(work)}", flush=True)

        payloads = {
            "attempts.jsonl": b"".join(
                canonical_json(value) + b"\n" for value in attempts
            ),
            "local-results.jsonl": b"".join(
                canonical_json(value) + b"\n" for value in results
            ),
            "frontier-candidates.jsonl": b"".join(
                canonical_json(value.model_dump(mode="json", by_alias=True))
                + b"\n"
                for value in candidates
            ),
            "pillar-evidence.jsonl": b"".join(
                canonical_json(value) + b"\n"
                for value in pillar_evidence
            ),
        }
        artifacts = {}
        for name, payload in payloads.items():
            path = temporary / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(path, 0o444)
            artifacts[name] = {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        core = {
            "schema": SCHEMA,
            "policy": {
                "ordering": [
                    "pymupdf_blocks_and_tables",
                    "deterministic_structure",
                    "pdftotext_layout_if_deterministic_incomplete",
                    "local_text_when_focused_vision_is_fallback",
                    "focused_local_vision",
                    "anthropic_batch_teacher_for_all_bootstrap_outputs",
                ],
                "full_page_vision": False,
                "focused_vision_policy": args.vision_policy,
                "redundant_text_generation_before_mandatory_vision": False,
                "unsupported_claims_admitted": False,
            },
            "model": {
                "provider": "local",
                "model": args.model,
                "base_url": args.base_url,
            },
            "selection": {
                "offset": args.offset,
                "limit": args.limit,
                "work_items": len(work),
            },
            "sources": {
                "structural_queue_sha256": sha256_file(queue_path),
                "structural_queue_evidence_sha256": queue[
                    "evidence_sha256"
                ],
                "page_evidence_sha256": evidence_manifest[
                    "evidence_sha256"
                ],
            },
            "counts": dict(sorted(counts.items())),
            "artifacts": artifacts,
            "images": render_receipts,
            "implementation": implementation_receipts,
        }
        core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **core,
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("xb") as handle:
            handle.write(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        for path in images.glob("*.png"):
            os.chmod(path, 0o444)
        os.chmod(manifest_path, 0o444)
        os.chmod(images, 0o555)
        os.chmod(implementation, 0o555)
        os.chmod(temporary, 0o555)
        os.rename(temporary, output)
    except BaseException:
        try:
            os.chmod(images, 0o755)
            os.chmod(implementation, 0o755)
            os.chmod(temporary, 0o755)
        except OSError:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
