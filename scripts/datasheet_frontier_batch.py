#!/usr/bin/env python3
"""Prepare, submit, retrieve, and reconcile Anthropic teacher batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from harness.electronics.claims import canonical_json
from harness.electronics.frontier_batch import (
    AnthropicBatchClient,
    FrontierCandidate,
    FrontierTeacherVerification,
    build_preference_training_pairs,
    finalize_training_pairs,
    load_prepared_bundle,
    prepare_batch_bundle,
    reconcile_results,
    request_chunks,
)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"immutable output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_bytes(path: Path, value: bytes) -> None:
    destination = path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"immutable output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(destination, 0o444)


def _client(args: argparse.Namespace) -> AnthropicBatchClient:
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"missing ${args.api_key_env}")
    return AnthropicBatchClient(
        api_key=api_key,
        base_url=args.base_url,
        timeout_s=args.timeout_seconds,
    )


def _add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-key-env", default="ANTHROPIC_API_KEY")
    parser.add_argument("--base-url", default="https://api.anthropic.com")
    parser.add_argument("--timeout-seconds", type=float, default=900)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument(
        "--candidates",
        type=Path,
        action="append",
        required=True,
    )
    prepare.add_argument("--allowed-root", type=Path, action="append", required=True)
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--input-price-per-million", type=float, required=True)
    prepare.add_argument("--output-price-per-million", type=float, required=True)
    prepare.add_argument("--batch-discount", type=float, default=0.5)
    prepare.add_argument("--spend-cap-usd", type=float, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    submit = commands.add_parser("submit")
    submit.add_argument("--bundle", type=Path, required=True)
    submit.add_argument("--state-directory", type=Path, required=True)
    submit.add_argument("--approved-spend-cap-usd", type=float, required=True)
    submit.add_argument("--maximum-batch-bytes", type=int, default=240_000_000)
    submit.add_argument("--maximum-batch-requests", type=int, default=100_000)
    submit.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse verified submitted chunk receipts and continue only chunks "
            "that have never reached the API."
        ),
    )
    _add_api_arguments(submit)

    status = commands.add_parser("status")
    status.add_argument("--state-directory", type=Path, required=True)
    status.add_argument("--output", type=Path, required=True)
    _add_api_arguments(status)

    retrieve = commands.add_parser("retrieve")
    retrieve.add_argument("--state-directory", type=Path, required=True)
    retrieve.add_argument("--output-directory", type=Path, required=True)
    _add_api_arguments(retrieve)

    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("--bundle", type=Path, required=True)
    reconcile.add_argument("--results-directory", type=Path, required=True)
    reconcile.add_argument("--input-price-per-million", type=float, required=True)
    reconcile.add_argument("--output-price-per-million", type=float, required=True)
    reconcile.add_argument("--batch-discount", type=float, default=0.5)
    reconcile.add_argument("--output", type=Path, required=True)

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--bundle", type=Path, required=True)
    finalize.add_argument("--reconciliation", type=Path, required=True)
    finalize.add_argument("--verifications", type=Path, required=True)
    finalize.add_argument(
        "--local-results",
        type=Path,
        action="append",
        default=[],
    )
    finalize.add_argument("--output-directory", type=Path, required=True)
    return parser


def _candidates(paths: list[Path]):
    for path in paths:
        with path.expanduser().resolve(strict=True).open(
            encoding="utf-8"
        ) as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    yield FrontierCandidate.model_validate_json(line)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid candidate at {path}:{line_number}"
                    ) from exc


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    return prepare_batch_bundle(
        args.output,
        _candidates(args.candidates),
        model=args.model,
        allowed_roots=args.allowed_root,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
        batch_discount=args.batch_discount,
        spend_cap_usd=args.spend_cap_usd,
        created_at=datetime.now(timezone.utc),
    )


def _submission_receipts(state: Path) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for path in sorted(state.glob("chunk-*.submitted.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        receipts.append(value)
    return receipts


def _submit(args: argparse.Namespace) -> dict[str, Any]:
    manifest, requests = load_prepared_bundle(args.bundle)
    configured_cap = float(manifest["pricing"]["spend_cap_usd"])
    estimated = float(manifest["pricing"]["estimated_maximum_usd"])
    if args.approved_spend_cap_usd != configured_cap:
        raise ValueError(
            "--approved-spend-cap-usd must exactly match the sealed bundle cap"
        )
    if estimated > args.approved_spend_cap_usd:
        raise ValueError("estimated cost exceeds approved spend cap")
    chunks = request_chunks(
        requests,
        maximum_bytes=args.maximum_batch_bytes,
        maximum_requests=args.maximum_batch_requests,
    )
    state = args.state_directory.expanduser().resolve()
    if state.is_symlink():
        raise ValueError(f"submission state is unsafe: {state}")
    intended = {
        "schema": "harness.electronics-frontier-submit-intent.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "local_training_pair_generation",
        "prepared_evidence_sha256": manifest["evidence_sha256"],
        "approved_spend_cap_usd": args.approved_spend_cap_usd,
        "chunks": [
            {
                "index": index,
                "request_count": len(chunk),
                "request_sha256": hashlib.sha256(
                    canonical_json({"requests": chunk})
                ).hexdigest(),
                "custom_ids": [request["custom_id"] for request in chunk],
            }
            for index, chunk in enumerate(chunks, 1)
        ],
    }
    intent_path = state / "submission-intent.json"
    resumed = state.exists()
    if resumed:
        if not args.resume or not state.is_dir():
            raise ValueError(f"submission state already exists: {state}")
        if intent_path.is_symlink() or not intent_path.is_file():
            raise ValueError("existing submission state has no safe intent")
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        comparable = {
            key: value
            for key, value in intent.items()
            if key != "created_at"
        }
        intended_comparable = {
            key: value
            for key, value in intended.items()
            if key != "created_at"
        }
        if comparable != intended_comparable:
            raise ValueError(
                "existing submission intent differs from prepared bundle"
            )
    else:
        state.mkdir(parents=True, mode=0o700)
        _write_new_json(intent_path, intended)
        intent = intended
    client = _client(args)
    submitted: list[dict[str, Any]] = []
    network_submissions = 0
    for index, chunk in enumerate(chunks, 1):
        chunk_intent = intent["chunks"][index - 1]
        attempted_path = state / f"chunk-{index:04d}.attempted.json"
        submitted_path = state / f"chunk-{index:04d}.submitted.json"
        if submitted_path.exists():
            if submitted_path.is_symlink() or not submitted_path.is_file():
                raise ValueError(f"unsafe submission receipt: {submitted_path}")
            receipt = json.loads(submitted_path.read_text(encoding="utf-8"))
            expected_receipt = {
                "prepared_evidence_sha256": manifest["evidence_sha256"],
                **chunk_intent,
            }
            if any(
                receipt.get(key) != value
                for key, value in expected_receipt.items()
            ) or not isinstance(receipt.get("batch_id"), str):
                raise ValueError(
                    f"submission receipt differs from intent: {submitted_path}"
                )
            submitted.append(receipt)
            continue
        if attempted_path.exists():
            raise ValueError(
                "submission outcome is ambiguous; attempted receipt exists "
                f"without a submitted receipt: {attempted_path}"
            )
        _write_new_json(
            attempted_path,
            {
                **chunk_intent,
                "attempted_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        response = client.submit(chunk)
        network_submissions += 1
        receipt = {
            "schema": "harness.electronics-frontier-submission.v1",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "local_training_pair_generation",
            "prepared_evidence_sha256": manifest["evidence_sha256"],
            **chunk_intent,
            "batch_id": response["id"],
            "api_response": response,
        }
        _write_new_json(submitted_path, receipt)
        submitted.append(receipt)
    return {
        "status": "resumed" if resumed else "submitted",
        "state_directory": str(state),
        "batches": [receipt["batch_id"] for receipt in submitted],
        "requests": len(requests),
        "estimated_maximum_usd": estimated,
        "approved_spend_cap_usd": args.approved_spend_cap_usd,
        "network_submissions": network_submissions,
    }


def _status(args: argparse.Namespace) -> dict[str, Any]:
    state = args.state_directory.expanduser().resolve(strict=True)
    receipts = _submission_receipts(state)
    if not receipts:
        raise ValueError("submission state has no completed batch receipts")
    client = _client(args)
    statuses = [client.status(receipt["batch_id"]) for receipt in receipts]
    core = {
        "schema": "harness.electronics-frontier-status.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "local_training_pair_generation",
        "batches": statuses,
    }
    core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
    _write_new_json(args.output, core)
    return core


def _retrieve(args: argparse.Namespace) -> dict[str, Any]:
    state = args.state_directory.expanduser().resolve(strict=True)
    receipts = _submission_receipts(state)
    if not receipts:
        raise ValueError("submission state has no completed batch receipts")
    output = args.output_directory.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"results output already exists: {output}")
    output.mkdir(parents=True, mode=0o700)
    client = _client(args)
    batch_receipts: list[dict[str, Any]] = []
    try:
        for receipt in receipts:
            batch_id = receipt["batch_id"]
            status = client.status(batch_id)
            if status.get("processing_status") != "ended":
                raise ValueError(f"batch has not ended: {batch_id}")
            results_url = status.get("results_url")
            if not isinstance(results_url, str) or not results_url:
                raise ValueError(f"ended batch has no results URL: {batch_id}")
            payload = client.results(results_url)
            result_path = output / f"{batch_id}.jsonl"
            _write_new_bytes(result_path, payload)
            batch_receipts.append(
                {
                    "batch_id": batch_id,
                    "result_file": result_path.name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "api_status": status,
                }
            )
        core = {
            "schema": "harness.electronics-frontier-results.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "local_training_pair_generation",
            "batches": batch_receipts,
        }
        core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
        _write_new_json(output / "manifest.json", core)
        os.chmod(output, 0o555)
        return core
    except BaseException:
        print(
            f"partial retrieval retained for reconciliation: {output}",
            file=sys.stderr,
        )
        raise


def _reconcile(args: argparse.Namespace) -> dict[str, Any]:
    results = args.results_directory.expanduser().resolve(strict=True)
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    raw: list[tuple[str, bytes]] = []
    for batch in manifest.get("batches") or []:
        path = results / batch["result_file"]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != batch["sha256"]:
            raise ValueError(f"result hash mismatch: {path}")
        raw.append((batch["batch_id"], payload))
    report = reconcile_results(
        args.bundle,
        raw,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
        batch_discount=args.batch_discount,
    )
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    _write_new_json(args.output, report)
    return report


def _finalize(args: argparse.Namespace) -> dict[str, Any]:
    reconciliation_path = args.reconciliation.expanduser().resolve(strict=True)
    reconciliation = json.loads(
        reconciliation_path.read_text(encoding="utf-8")
    )
    verifications: list[FrontierTeacherVerification] = []
    with args.verifications.expanduser().resolve(strict=True).open(
        encoding="utf-8"
    ) as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                verifications.append(
                    FrontierTeacherVerification.model_validate_json(line)
                )
            except ValueError as exc:
                raise ValueError(
                    f"invalid teacher verification at line {line_number}"
                ) from exc
    report, pairs = finalize_training_pairs(
        args.bundle,
        reconciliation,
        verifications,
    )
    local_results: list[dict[str, Any]] = []
    for local_path in args.local_results:
        with local_path.expanduser().resolve(strict=True).open(
            encoding="utf-8"
        ) as handle:
            for line_number, line in enumerate(handle, 1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"invalid local result at line {line_number}: "
                        f"{local_path}"
                    )
                local_results.append(value)
    preference_report, preference_pairs = build_preference_training_pairs(
        args.bundle,
        reconciliation,
        verifications,
        pairs,
        local_results,
    )
    output = args.output_directory.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"finalization output already exists: {output}")
    output.mkdir(parents=True, mode=0o700)
    pairs_payload = b"".join(
        canonical_json(pair.model_dump(mode="json", by_alias=True)) + b"\n"
        for pair in pairs
    )
    preference_payload = b"".join(
        canonical_json(pair.model_dump(mode="json", by_alias=True)) + b"\n"
        for pair in preference_pairs
    )
    _write_new_bytes(output / "training-pairs.jsonl", pairs_payload)
    _write_new_bytes(
        output / "preference-training-pairs.jsonl",
        preference_payload,
    )
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["preference_training"] = preference_report
    report["artifacts"] = {
        "training-pairs.jsonl": {
            "sha256": hashlib.sha256(pairs_payload).hexdigest(),
            "bytes": len(pairs_payload),
        },
        "preference-training-pairs.jsonl": {
            "sha256": hashlib.sha256(preference_payload).hexdigest(),
            "bytes": len(preference_payload),
        },
        "reconciliation": {
            "path": str(reconciliation_path),
            "sha256": hashlib.sha256(
                reconciliation_path.read_bytes()
            ).hexdigest(),
        },
    }
    core = {
        key: value
        for key, value in report.items()
        if key not in {"created_at", "evidence_sha256"}
    }
    report["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
    _write_new_json(output / "manifest.json", report)
    os.chmod(output, 0o555)
    return report


def main() -> int:
    args = _parser().parse_args()
    handlers = {
        "prepare": _prepare,
        "submit": _submit,
        "status": _status,
        "retrieve": _retrieve,
        "reconcile": _reconcile,
        "finalize": _finalize,
    }
    value = handlers[args.command](args)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
