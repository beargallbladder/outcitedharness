#!/usr/bin/env python3
"""Run bounded local extraction shards and prepare only terminal escalations."""

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
from harness.electronics.models import PairCapability


BUNDLE_SCHEMA = "harness.electronics-local-extraction-bundle.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic-bundle", type=Path, required=True)
    parser.add_argument("--priority-queue", type=Path)
    parser.add_argument("--page-evidence", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--partition",
        choices=("factory_candidate", "frozen_evaluation"),
        default="factory_candidate",
    )
    parser.add_argument("--capability", choices=tuple(RESPONSE_SCHEMAS))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--render-dpi", type=int, default=180)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def _load_work(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = args.deterministic_bundle.expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / "local-model-queue.jsonl"
    receipt = manifest["artifacts"]["local-model-queue.jsonl"]
    if sha256_file(path) != receipt["sha256"] or path.stat().st_size != receipt["bytes"]:
        raise ValueError("local work queue differs from its manifest")
    source_rows: Any
    if args.priority_queue is not None:
        priority_path = args.priority_queue.expanduser().resolve(strict=True)
        priority = json.loads(priority_path.read_text())
        if priority.get("schema") != "harness.electronics-prioritized-local-work.v1":
            raise ValueError("priority queue schema is not supported")
        expected = priority.get("evidence_sha256")
        core = {
            key: value
            for key, value in priority.items()
            if key != "evidence_sha256"
        }
        if hashlib.sha256(canonical_json(core)).hexdigest() != expected:
            raise ValueError("priority queue evidence digest is invalid")
        source_rows = priority["work"]
    else:
        source_rows = (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    selected: list[dict[str, Any]] = []
    skipped = 0
    for row in source_rows:
        if row["partition"] != args.partition:
            continue
        if args.capability and row["capability"] != args.capability:
            continue
        if skipped < args.offset:
            skipped += 1
            continue
        selected.append(row)
        if len(selected) == args.limit:
            break
    if not selected:
        raise ValueError("local extraction shard selected no work")
    return manifest, selected


def _load_page_evidence(
    root: Path,
    work: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / "page-evidence.jsonl"
    receipt = manifest["artifacts"]["page-evidence.jsonl"]
    if sha256_file(path) != receipt["sha256"] or path.stat().st_size != receipt["bytes"]:
        raise ValueError("page evidence differs from its manifest")
    wanted = {
        (row["document_sha256"], int(row["page_1based"])) for row in work
    }
    pages: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            key = (value["document_sha256"], int(value["page_1based"]))
            if key in wanted:
                pages[key] = value
    if set(pages) != wanted:
        raise ValueError("local work references missing page evidence")
    return manifest, pages


def _render(
    *,
    source_path: Path,
    document_sha256: str,
    page_1based: int,
    dpi: int,
    destination: Path,
) -> str:
    if sha256_file(source_path) != document_sha256:
        raise ValueError("source PDF changed after corpus seal")
    import pymupdf

    with pymupdf.open(source_path) as document:
        if not 1 <= page_1based <= document.page_count:
            raise ValueError("render page is outside document bounds")
        pixmap = document[page_1based - 1].get_pixmap(dpi=dpi, alpha=False)
        pixmap.save(destination)
    return sha256_file(destination)


def main() -> int:
    args = _parser().parse_args()
    if args.offset < 0 or not 1 <= args.limit <= 1000:
        raise ValueError("offset must be non-negative and limit within 1..1000")
    if not 72 <= args.render_dpi <= 300:
        raise ValueError("render DPI must be within 72..300")
    deterministic_manifest, work = _load_work(args)
    evidence_root = args.page_evidence.expanduser().resolve(strict=True)
    evidence_manifest, pages = _load_page_evidence(evidence_root, work)
    output = args.output_directory.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    image_root = temporary / "images"
    image_root.mkdir()
    attempts_path = temporary / "attempts.jsonl"
    results_path = temporary / "local-results.jsonl"
    candidates_path = temporary / "frontier-candidates.jsonl"
    client = LocalExtractionClient(
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout_seconds,
    )
    statuses: Counter[str] = Counter()
    try:
        with attempts_path.open("xb") as attempts_handle, results_path.open(
            "xb"
        ) as results_handle, candidates_path.open("xb") as candidates_handle:
            for index, item in enumerate(work, 1):
                key = (item["document_sha256"], int(item["page_1based"]))
                page = pages[key]
                image_name = (
                    f"{item['document_sha256'][:16]}-"
                    f"p{int(item['page_1based']):05d}.png"
                )
                image_path = image_root / image_name
                image_sha = _render(
                    source_path=Path(item["source_path"]),
                    document_sha256=item["document_sha256"],
                    page_1based=int(item["page_1based"]),
                    dpi=args.render_dpi,
                    destination=image_path,
                )
                attempt_core = {
                    "schema": "harness.electronics-local-attempt.v1",
                    "work_id": item["work_id"],
                    "provider": "local",
                    "model": args.model,
                    "capability": item["capability"],
                    "document_sha256": item["document_sha256"],
                    "page_1based": item["page_1based"],
                    "page_evidence_sha256": item["page_evidence_sha256"],
                    "image_sha256": image_sha,
                }
                try:
                    result = client.extract(
                        capability=item["capability"],
                        page_evidence=page,
                        image_path=image_path,
                    )
                except (httpx.HTTPError, OSError, TimeoutError) as exc:
                    attempt = {
                        **attempt_core,
                        "status": "infrastructure_failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "frontier_batch_eligible": False,
                    }
                    statuses["infrastructure_failed"] += 1
                except (ValueError, json.JSONDecodeError) as exc:
                    terminal_status = (
                        "no_answer"
                        if "no JSON object" in str(exc)
                        else "schema_failed"
                    )
                    attempt = {
                        **attempt_core,
                        "status": terminal_status,
                        "reason": f"{type(exc).__name__}: {exc}",
                        "frontier_batch_eligible": (
                            item["partition"] == "factory_candidate"
                        ),
                    }
                    statuses[terminal_status] += 1
                    if item["partition"] == "factory_candidate":
                        attempt_sha = hashlib.sha256(
                            canonical_json(attempt)
                        ).hexdigest()
                        local_attempt = LocalAttempt(
                            provider="local",
                            model=args.model,
                            status=terminal_status,
                            receipt_sha256=attempt_sha,
                            reason=attempt["reason"],
                        )
                        final_image_path = output / "images" / image_name
                        draft = FrontierCandidate(
                            candidate_id="candidate-" + ("0" * 32),
                            capability=PairCapability(item["capability"]),
                            document_sha256=item["document_sha256"],
                            entity_hint=item["work_id"],
                            prompt=local_prompt(item["capability"], page),
                            response_schema=RESPONSE_SCHEMAS[item["capability"]],
                            evidence=(
                                FrontierEvidence(
                                    path=final_image_path,
                                    sha256=image_sha,
                                    media_type="image/png",
                                    page_1based=item["page_1based"],
                                ),
                            ),
                            local_attempts=(local_attempt,),
                            estimated_input_tokens=(
                                len(local_prompt(item["capability"], page)) // 4
                                + 1600
                            ),
                        )
                        candidate = draft.model_copy(
                            update={
                                "candidate_id": candidate_id(
                                    candidate_identity_payload(draft)
                                )
                            }
                        )
                        candidates_handle.write(
                            canonical_json(
                                candidate.model_dump(
                                    mode="json",
                                    by_alias=True,
                                )
                            )
                            + b"\n"
                        )
                        statuses["frontier_candidates"] += 1
                else:
                    attempt = {
                        **attempt_core,
                        "status": "succeeded_pending_verification",
                        "request_sha256": result["request_sha256"],
                        "response_sha256": result["response_sha256"],
                        "frontier_batch_eligible": False,
                    }
                    result["work_id"] = item["work_id"]
                    result["partition"] = item["partition"]
                    result["page_evidence_sha256"] = item[
                        "page_evidence_sha256"
                    ]
                    results_handle.write(canonical_json(result) + b"\n")
                    statuses["succeeded_pending_verification"] += 1
                attempt["receipt_sha256"] = hashlib.sha256(
                    canonical_json(attempt)
                ).hexdigest()
                attempts_handle.write(canonical_json(attempt) + b"\n")
                print(f"local extraction {index}/{len(work)}", flush=True)
            for handle in (attempts_handle, results_handle, candidates_handle):
                handle.flush()
                os.fsync(handle.fileno())
        artifacts = {}
        for path in (attempts_path, results_path, candidates_path):
            artifacts[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        image_receipts = {
            path.name: {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(image_root.glob("*.png"))
        }
        core = {
            "schema": BUNDLE_SCHEMA,
            "policy": {
                "partition": args.partition,
                "holdout_outputs_admitted_to_training": False,
                "transient_infrastructure_failure_frontier_eligible": False,
                "terminal_local_failure_frontier_eligible": True,
                "temperature": 0,
                "thinking": False,
            },
            "model": {
                "provider": "local",
                "model": args.model,
                "base_url": args.base_url,
            },
            "selection": {
                "offset": args.offset,
                "limit": args.limit,
                "capability": args.capability,
                "work_items": len(work),
            },
            "sources": {
                "deterministic_evidence_sha256": deterministic_manifest[
                    "evidence_sha256"
                ],
                "page_evidence_sha256": evidence_manifest["evidence_sha256"],
            },
            "artifacts": artifacts,
            "images": image_receipts,
            "counts": dict(sorted(statuses.items())),
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
        for path in (
            attempts_path,
            results_path,
            candidates_path,
            manifest_path,
            *image_root.glob("*.png"),
        ):
            os.chmod(path, 0o444)
        os.chmod(image_root, 0o555)
        os.chmod(temporary, 0o555)
        os.rename(temporary, output)
    except BaseException:
        try:
            os.chmod(image_root, 0o755)
            os.chmod(temporary, 0o755)
        except OSError:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
