#!/usr/bin/env python3
"""Compare local datasheet vision with an independent frontier vision model."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from harness.config import ModelConfig, load_config
from harness.storage.db import Store
from harness.training.ledger import (
    ArtifactPayload,
    LearningLedger,
    VerificationPayload,
)
from harness.training.models import LearningEvent, SourceKind

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evaluate_datasheet_modalities import (  # noqa: E402
    RESULT_SCHEMA,
    SYSTEM_PROMPT,
    _canonical,
    _pin_name,
    _pin_number,
    _request_content,
    extract_json,
    load_fixture,
    prediction_pins,
    sha256_file,
    write_new_json,
)


SCHEMA = "harness.datasheet-frontier-comparison.v1"
POLICY_VERSION = "datasheet-independent-three-way-consensus-v3"


def _load_json_file(path: Path, kind: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{kind} must be a regular file")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{kind} must contain an object")
    return value


def load_local_evaluation(
    path: Path,
    *,
    fixture: Path,
) -> dict[str, Any]:
    value = _load_json_file(path, "local evaluation")
    if value.get("schema") != RESULT_SCHEMA:
        raise ValueError("unsupported local evaluation schema")
    if value.get("fixture_sha256") != sha256_file(fixture.resolve()):
        raise ValueError("local evaluation is bound to another fixture")
    identity = value.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("local evaluation has no identity")
    core = {key: item for key, item in value.items() if key != "identity"}
    if identity.get("core_sha256") != hashlib.sha256(_canonical(core)).hexdigest():
        raise ValueError("local evaluation core digest mismatch")
    cases = value.get("cases")
    if not isinstance(cases, list):
        raise ValueError("local evaluation cases are malformed")
    identifiers = [
        row.get("id") for row in cases if isinstance(row, dict)
    ]
    if (
        len(identifiers) != len(cases)
        or any(not isinstance(identifier, str) or not identifier for identifier in identifiers)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise ValueError("local evaluation case identities are malformed")
    return value


def _anthropic_url(model: ModelConfig) -> str:
    return (
        f"{model.base_url}/messages"
        if model.base_url.endswith("/v1")
        else f"{model.base_url}/v1/messages"
    )


def frontier_chat(
    client: httpx.Client,
    *,
    model: ModelConfig,
    instruction: str,
    image: bytes,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    if model.provider != "anthropic" or not model.capabilities.vision:
        raise ValueError("frontier model must be an Anthropic vision model")
    if model.missing_key:
        raise ValueError(f"missing ${model.api_key_env}")
    payload = {
        "model": model.model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(image).decode(),
                        },
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ],
    }
    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = client.post(
                _anthropic_url(model),
                headers={
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                    "x-api-key": str(model.api_key),
                },
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict) or body.get("model") != model.model:
                raise ValueError("frontier response model identity mismatch")
            blocks = body.get("content") if isinstance(body, dict) else None
            if not isinstance(blocks, list):
                raise ValueError("frontier response has invalid content")
            text = "".join(
                str(block.get("text") or "")
                for block in blocks
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if not text:
                raise ValueError("frontier response has no text")
            return text, {
                "attempts": attempt,
                "elapsed_seconds": time.monotonic() - started,
                "model": body.get("model"),
                "request_id": response.headers.get("request-id"),
                "response_id": body.get("id"),
                "stop_reason": body.get("stop_reason"),
                "usage": body.get("usage"),
            }
        except (httpx.HTTPError, ValueError) as error:
            last_error = error
            if attempt == 3:
                break
            time.sleep(attempt)
    raise RuntimeError(f"frontier request failed after retries: {last_error}")


def _pairs(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (_pin_number(row.get("pin_no")), _pin_name(row.get("name")))
        for row in rows
    }


def compare_page(
    *,
    local_pins: list[dict[str, Any]],
    table_pins: list[dict[str, Any]],
    frontier_pins: list[dict[str, Any]],
    frontier_parse_error: str | None,
    truth_pins: list[dict[str, Any]],
    split_role: str,
) -> dict[str, Any]:
    local_pairs = _pairs(local_pins)
    table_pairs = _pairs(table_pins)
    frontier_pairs = _pairs(frontier_pins)
    truth_pairs = _pairs(truth_pins)
    three_way_consensus = (
        frontier_parse_error is None
        and bool(local_pairs)
        and len(local_pairs) == len(local_pins)
        and len(table_pairs) == len(table_pins)
        and len(frontier_pairs) == len(frontier_pins)
        and local_pairs == table_pairs == frontier_pairs
    )
    return {
        "row_counts": {
            "local": len(local_pins),
            "table": len(table_pins),
            "frontier": len(frontier_pins),
            "truth": len(truth_pins),
        },
        "frontier_parse_error": frontier_parse_error,
        "local_equals_table": local_pairs == table_pairs,
        "local_equals_frontier": local_pairs == frontier_pairs,
        "table_equals_frontier": table_pairs == frontier_pairs,
        "frontier_equals_existing_truth": frontier_pairs == truth_pairs,
        "independent_three_way_consensus": three_way_consensus,
        "training_eligible": split_role == "candidate" and three_way_consensus,
    }


def _cost(
    usage: Any,
    *,
    input_per_million: float | None,
    output_per_million: float | None,
) -> float | None:
    if (
        not isinstance(usage, dict)
        or input_per_million is None
        or output_per_million is None
    ):
        return None
    try:
        return (
            int(usage["input_tokens"]) * input_per_million
            + int(usage["output_tokens"]) * output_per_million
        ) / 1_000_000
    except (KeyError, TypeError, ValueError):
        return None


def _page_rows(mode: Any, page_number: int) -> dict[str, Any]:
    if not isinstance(mode, dict) or not isinstance(mode.get("pages"), list):
        raise ValueError("local evaluation is missing page evidence")
    matches = [
        row
        for row in mode["pages"]
        if isinstance(row, dict) and row.get("page") == page_number
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("pins"), list):
        raise ValueError("local evaluation page evidence is ambiguous")
    return matches[0]


def frozen_lineage_overlaps(
    cases: list[dict[str, Any]],
    frozen_pdf_digests: set[str],
) -> list[str]:
    return sorted(
        str(case["id"])
        for case in cases
        if str(case["pdf_sha256"]) in frozen_pdf_digests
    )


def _capture_page(
    *,
    ledger: LearningLedger,
    run_id: str,
    case: dict[str, Any],
    page_number: int,
    image: bytes,
    instruction: str,
    local_pins: list[dict[str, Any]],
    table_pins: list[dict[str, Any]],
    frontier_pins: list[dict[str, Any]],
    frontier_evidence: dict[str, Any],
    comparison: dict[str, Any],
    estimated_cost: float | None,
    split_role: str,
    admit_consensus: bool,
    local_identity: dict[str, Any],
    frontier_model: str,
) -> tuple[str, str | None]:
    pdf_sha256 = str(case["pdf_sha256"])
    consensus = bool(comparison["independent_three_way_consensus"])
    eligible = bool(comparison["training_eligible"])
    page_identity = hashlib.sha256(
        f"{pdf_sha256}\n{case['requested_package']}\n{page_number}".encode()
    ).hexdigest()[:20]
    event_id = f"datasheet-frontier-{run_id}-{page_identity}"
    event = LearningEvent(
        event_id=event_id,
        event_type="datasheet_frontier_vision_comparison",
        source_kind=SourceKind.OTHER,
        source_uri=f"datasheet://sha256/{pdf_sha256}/page/{page_number}",
        source_revision=pdf_sha256,
        lineage_id=(
            f"datasheet:{pdf_sha256}:{case['requested_package']}:page:{page_number}"
        ),
        authorization_scope="public_datasheet_frontier_comparison_and_distillation",
        created_at=datetime.now(timezone.utc),
        estimated_cost=estimated_cost,
        metadata={
            "data_use": "training" if eligible else "quarantine",
            "disposition": "verified" if eligible else "quarantine",
            "task_class": "electronics_pinout_extraction",
            "case_id": case["id"],
            "page": page_number,
            "package": case["requested_package"],
            "split_role": split_role,
            "training_eligible": eligible,
            "input_sha256": hashlib.sha256(image).hexdigest(),
            "local_model": local_identity["model"],
            "local_evaluation_sha256": local_identity[
                "local_evaluation_sha256"
            ],
            "local_model_manifest_sha256": local_identity[
                "model_manifest_sha256"
            ],
            "local_runtime_image_id": local_identity["runtime_image_id"],
            "local_runtime_version": local_identity["runtime_version"],
            "frontier_model": frontier_model,
            "verification_policy": POLICY_VERSION,
        },
    )
    comparison_json = json.dumps(comparison, ensure_ascii=False, sort_keys=True)
    capture = ledger.capture(
        event,
        [
            ArtifactPayload(
                kind="rendered_page_rows",
                content=image,
                media_type="image/png",
                redact=False,
            ),
            ArtifactPayload(kind="training_prompt", content=instruction),
            ArtifactPayload(
                kind="local_prediction",
                content=json.dumps({"pins": local_pins}, sort_keys=True),
                media_type="application/json",
            ),
            ArtifactPayload(
                kind="deterministic_table_prediction",
                content=json.dumps({"pins": table_pins}, sort_keys=True),
                media_type="application/json",
            ),
            ArtifactPayload(
                kind="frontier_prediction",
                content=json.dumps(
                    {"pins": frontier_pins, "evidence": frontier_evidence},
                    sort_keys=True,
                ),
                media_type="application/json",
            ),
            ArtifactPayload(
                kind="comparison",
                content=comparison_json,
                media_type="application/json",
            ),
        ],
        [
            VerificationPayload(
                kind="independent_three_way_page_consensus",
                status="pass" if consensus else "fail",
                verifier=POLICY_VERSION,
                output_kind="comparison",
                metadata={
                    "proof_scope": "rendered_page_physical_rows",
                    "split_role": split_role,
                },
            )
        ],
    )
    ledger.verify_event(event_id)
    admission_id = None
    if eligible and admit_consensus:
        admission = ledger.admit_verified_event(
            event_id,
            capture.verifications[0].verification_id,
            policy_version=POLICY_VERSION,
            reason=(
                "independent local vision, deterministic table extraction, and "
                "frontier vision agree on every visible physical pin row"
            ),
        )
        admission_id = admission.admission_id
    return event_id, admission_id


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture_manifest, cases = load_fixture(args.fixture)
    local = load_local_evaluation(args.local_evaluation, fixture=args.fixture)
    local_by_id = {row["id"]: row for row in local["cases"]}
    selected_ids = set(args.cases or [case["id"] for case in cases])
    unknown = selected_ids - {case["id"] for case in cases}
    if unknown:
        raise ValueError(f"unknown requested cases: {sorted(unknown)}")

    frozen_pdf_digests: set[str] = set()
    for path in args.frozen_fixtures:
        frozen, _ = load_fixture(path)
        frozen_pdf_digests.update(
            str(row["pdf_sha256"]) for row in frozen["cases"]
        )
    if args.admit_consensus and not args.capture_ledger:
        raise ValueError("--admit-consensus requires --capture-ledger")
    if args.admit_consensus and args.split_role != "candidate":
        raise ValueError("only candidate data can be admitted")
    if args.admit_consensus and not args.frozen_fixtures:
        raise ValueError("admission requires at least one frozen fixture")

    selected_cases = [
        case for case in cases if case["id"] in selected_ids
    ]
    if args.admit_consensus:
        overlaps = frozen_lineage_overlaps(
            selected_cases,
            frozen_pdf_digests,
        )
        if overlaps:
            raise ValueError(
                f"candidate lineages overlap a frozen fixture: {overlaps}"
            )

    config = load_config()
    model = config.models[args.frontier_model_key]
    pricing = config.pricing.get(args.frontier_model_key)
    ledger = (
        LearningLedger(Store(config.settings.db_path), config.settings.learning_artifact_root)
        if args.capture_ledger
        else None
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    page_results: list[dict[str, Any]] = []
    pages_sent = 0

    import pymupdf as fitz

    with httpx.Client(timeout=args.timeout) as client:
        for case in selected_cases:
            local_case = local_by_id.get(case["id"])
            if not isinstance(local_case, dict):
                raise ValueError(f"{case['id']}: missing local evaluation")
            modalities = local_case.get("modalities")
            if not isinstance(modalities, dict):
                raise ValueError(f"{case['id']}: missing local modalities")
            located = local_case.get("locator")
            pages = located.get("pages_1based") if isinstance(located, dict) else None
            column = located.get("column_header") if isinstance(located, dict) else None
            if not isinstance(pages, list) or not pages or not isinstance(column, str):
                raise ValueError(f"{case['id']}: local locator did not send")

            with fitz.open(case["pdf_path"]) as document:
                for raw_page_number in pages:
                    if args.max_pages and pages_sent >= args.max_pages:
                        break
                    page_number = int(raw_page_number)
                    local_page = _page_rows(
                        modalities.get("image_rows"),
                        page_number,
                    )
                    table_page = _page_rows(
                        modalities.get("table"),
                        page_number,
                    )
                    if local_page.get("parse_error") is not None:
                        raise ValueError(
                            f"{case['id']} page {page_number}: local parse failed"
                        )
                    if table_page.get("parse_error") is not None:
                        raise ValueError(
                            f"{case['id']} page {page_number}: table parse failed"
                        )
                    content, input_evidence = _request_content(
                        mode="image_rows",
                        page=document[page_number - 1],
                        page_number=page_number,
                        document_id=str(case["id"]),
                        package=str(case["requested_package"]),
                        column_header=column,
                        dpi=args.dpi,
                    )
                    data_url = content[1]["image_url"]["url"]
                    image = base64.b64decode(data_url.split(",", 1)[1])
                    if input_evidence["input_sha256"] != local_page.get(
                        "input_sha256"
                    ):
                        raise ValueError(
                            f"{case['id']} page {page_number}: rendered input changed"
                        )
                    response, frontier_evidence = frontier_chat(
                        client,
                        model=model,
                        instruction=str(content[0]["text"]),
                        image=image,
                        max_tokens=args.max_tokens,
                    )
                    frontier_pins, parse_error = prediction_pins(
                        extract_json(response),
                        maximum_rows=max(
                            32,
                            int(case["expected_package_pins"]) * 2,
                        ),
                    )
                    table_numbers = {
                        _pin_number(row.get("pin_no"))
                        for row in table_page["pins"]
                    }
                    truth_pins = [
                        row
                        for row in case["truth"]["pins"]
                        if _pin_number(row.get("pin_no")) in table_numbers
                    ]
                    comparison = compare_page(
                        local_pins=local_page["pins"],
                        table_pins=table_page["pins"],
                        frontier_pins=frontier_pins,
                        frontier_parse_error=parse_error,
                        truth_pins=truth_pins,
                        split_role=args.split_role,
                    )
                    estimated_cost = _cost(
                        frontier_evidence.get("usage"),
                        input_per_million=(
                            pricing.input_per_million if pricing else None
                        ),
                        output_per_million=(
                            pricing.output_per_million if pricing else None
                        ),
                    )
                    record = {
                        "case_id": case["id"],
                        "page": page_number,
                        "package": case["requested_package"],
                        "pdf_sha256": case["pdf_sha256"],
                        "input_sha256": input_evidence["input_sha256"],
                        "local_model": local["model"],
                        "frontier_model": model.model,
                        "local_pins": local_page["pins"],
                        "table_pins": table_page["pins"],
                        "frontier_pins": frontier_pins,
                        "existing_truth_pins": truth_pins,
                        "frontier_response": response,
                        "frontier_evidence": frontier_evidence,
                        "estimated_cost": estimated_cost,
                        "comparison": comparison,
                    }
                    if ledger is not None:
                        event_id, admission_id = _capture_page(
                            ledger=ledger,
                            run_id=run_id,
                            case=case,
                            page_number=page_number,
                            image=image,
                            instruction=str(content[0]["text"]),
                            local_pins=local_page["pins"],
                            table_pins=table_page["pins"],
                            frontier_pins=frontier_pins,
                            frontier_evidence=frontier_evidence,
                            comparison=comparison,
                            estimated_cost=estimated_cost,
                            split_role=args.split_role,
                            admit_consensus=args.admit_consensus,
                            local_identity={
                                "model": local["model"],
                                "local_evaluation_sha256": sha256_file(
                                    args.local_evaluation.resolve()
                                ),
                                "model_manifest_sha256": local[
                                    "model_manifest_sha256"
                                ],
                                "runtime_image_id": local["runtime_image_id"],
                                "runtime_version": local["runtime_version"],
                            },
                            frontier_model=model.model,
                        )
                        record["learning_event_id"] = event_id
                        record["admission_id"] = admission_id
                    page_results.append(record)
                    pages_sent += 1
                if args.max_pages and pages_sent >= args.max_pages:
                    break

    total_cost = sum(
        float(row["estimated_cost"])
        for row in page_results
        if row["estimated_cost"] is not None
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "split_role": args.split_role,
        "fixture_sha256": sha256_file(args.fixture.resolve()),
        "local_evaluation_sha256": sha256_file(args.local_evaluation.resolve()),
        "source_gold_set_sha256": fixture_manifest["source_gold_set_sha256"],
        "frontier_model": model.model,
        "pages": page_results,
        "summary": {
            "pages": len(page_results),
            "consensus_pages": sum(
                row["comparison"]["independent_three_way_consensus"]
                for row in page_results
            ),
            "training_eligible_pages": sum(
                row["comparison"]["training_eligible"] for row in page_results
            ),
            "admitted_pages": sum(
                bool(row.get("admission_id")) for row in page_results
            ),
            "estimated_cost": total_cost,
        },
    }
    result["result_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--local-evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frontier-model-key", default="frontier")
    parser.add_argument("--case", action="append", dest="cases", default=[])
    parser.add_argument(
        "--split-role",
        required=True,
        choices=("candidate", "frozen_holdout"),
    )
    parser.add_argument(
        "--frozen-fixture",
        action="append",
        dest="frozen_fixtures",
        default=[],
        type=Path,
    )
    parser.add_argument("--capture-ledger", action="store_true")
    parser.add_argument("--admit-consensus", action="store_true")
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if (
        args.max_pages < 0
        or not 72 <= args.dpi <= 300
        or not 256 <= args.max_tokens <= 16384
        or args.timeout < 10
    ):
        parser.error("invalid frontier comparison limits")
    return args


def main() -> int:
    args = parse_args()
    result = run(args)
    write_new_json(args.output, result)
    print(json.dumps(result["summary"], sort_keys=True))
    print("DATASHEET_FRONTIER_COMPARISON_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
