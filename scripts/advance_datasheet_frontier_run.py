#!/usr/bin/env python3
"""Idempotently advance one submitted teacher batch through pair finalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.electronics.claims import canonical_json
from harness.electronics.factory_control import ElectronicsFactoryState
from harness.electronics.frontier_batch import (
    AnthropicBatchClient,
    load_prepared_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prepared-bundle", type=Path, required=True)
    parser.add_argument("--submission-state", type=Path, required=True)
    parser.add_argument("--lifecycle-root", type=Path, required=True)
    parser.add_argument(
        "--work-queue",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--page-evidence", type=Path, required=True)
    parser.add_argument("--pillar-evidence", type=Path)
    parser.add_argument("--corpus-registry", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument(
        "--local-results",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--input-price-per-million", type=float, required=True)
    parser.add_argument("--output-price-per-million", type=float, required=True)
    parser.add_argument("--batch-discount", type=float, default=0.5)
    parser.add_argument("--api-key-env", default="ANTHROPIC_API_KEY")
    parser.add_argument("--base-url", default="https://api.anthropic.com")
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--minimum-sft-pairs", type=int, default=1)
    parser.add_argument("--minimum-dpo-pairs", type=int, default=1)
    parser.add_argument("--training-ready-output", type=Path)
    return parser


def _submission_receipts(state: Path) -> list[dict[str, Any]]:
    receipts = []
    for path in sorted(state.glob("chunk-*.submitted.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe submission receipt: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value.get("batch_id"), str):
            raise ValueError(f"submission receipt has no batch ID: {path}")
        receipts.append(value)
    if not receipts:
        raise ValueError("submission state has no completed batch receipts")
    return receipts


def _run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"stage failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr[-4000:]}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError:
        value = {"stdout": completed.stdout.strip()}
    return value


def _regular(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if path.expanduser().is_symlink() or not resolved.is_file():
        raise ValueError(f"expected regular file: {path}")
    return resolved


def _line_count(path: Path) -> int:
    return sum(
        1
        for line in _regular(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _write_ready_receipt(
    path: Path,
    *,
    run_id: str,
    finalization: Path,
    sft_pairs: int,
    dpo_pairs: int,
) -> dict[str, Any]:
    destination = path.expanduser().resolve()
    if destination.exists():
        value = json.loads(destination.read_text(encoding="utf-8"))
        if (
            value.get("run_id") != run_id
            or value.get("counts")
            != {"sft": sft_pairs, "dpo": dpo_pairs}
        ):
            raise ValueError("training-ready receipt conflicts with current run")
        return value
    manifest_path = _regular(finalization / "manifest.json")
    core = {
        "schema": "harness.electronics-training-ready.v1",
        "run_id": run_id,
        "finalization": {
            "path": str(finalization),
            "manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
        },
        "counts": {"sft": sft_pairs, "dpo": dpo_pairs},
        "decision": "eligible_for_dataset_assembly",
    }
    value = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **core,
        "evidence_sha256": hashlib.sha256(canonical_json(core)).hexdigest(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def main() -> int:
    args = _parser().parse_args()
    if args.minimum_sft_pairs < 1 or args.minimum_dpo_pairs < 1:
        raise ValueError("training pair thresholds must be positive")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"missing ${args.api_key_env}")

    prepared = args.prepared_bundle.expanduser().resolve(strict=True)
    submission = args.submission_state.expanduser().resolve(strict=True)
    lifecycle = args.lifecycle_root.expanduser().resolve()
    lifecycle.mkdir(parents=True, exist_ok=True, mode=0o700)
    if lifecycle.is_symlink():
        raise ValueError("lifecycle root cannot be a symlink")
    os.chmod(lifecycle, 0o700)

    factory = ElectronicsFactoryState(args.state_root)
    factory.register_frontier_run(
        run_id=args.run_id,
        prepared_bundle=prepared,
        submission_state=submission,
        lifecycle_root=lifecycle,
    )
    manifest, _requests = load_prepared_bundle(prepared)
    receipts = _submission_receipts(submission)
    client = AnthropicBatchClient(
        api_key=api_key,
        base_url=args.base_url,
        timeout_s=args.timeout_seconds,
    )
    statuses = [client.status(receipt["batch_id"]) for receipt in receipts]
    status_payload = {
        "schema": "harness.electronics-frontier-status.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "local_training_pair_generation",
        "batches": statuses,
    }
    status_payload["evidence_sha256"] = hashlib.sha256(
        canonical_json(status_payload)
    ).hexdigest()
    ended = bool(statuses) and all(
        status.get("processing_status") == "ended" for status in statuses
    )
    factory.record_frontier_status(
        args.run_id,
        status_payload,
        stage="ended" if ended else "processing",
    )
    if not ended:
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "status": "processing",
                    "batches": statuses,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    batch_script = Path(__file__).with_name("datasheet_frontier_batch.py")
    results = lifecycle / "results"
    if not (results / "manifest.json").is_file():
        _run(
            [
                sys.executable,
                str(batch_script),
                "retrieve",
                "--state-directory",
                str(submission),
                "--output-directory",
                str(results),
                "--api-key-env",
                args.api_key_env,
                "--base-url",
                args.base_url,
                "--timeout-seconds",
                str(args.timeout_seconds),
            ]
        )
    factory.record_frontier_status(
        args.run_id,
        status_payload,
        stage="retrieved",
    )

    reconciliation = lifecycle / "reconciliation.json"
    if not reconciliation.is_file():
        _run(
            [
                sys.executable,
                str(batch_script),
                "reconcile",
                "--bundle",
                str(prepared),
                "--results-directory",
                str(results),
                "--input-price-per-million",
                str(args.input_price_per_million),
                "--output-price-per-million",
                str(args.output_price_per_million),
                "--batch-discount",
                str(args.batch_discount),
                "--output",
                str(reconciliation),
            ]
        )
    factory.record_frontier_status(
        args.run_id,
        status_payload,
        stage="reconciled",
    )

    verification_root = lifecycle / "verification"
    verifications = verification_root / "verifications.jsonl"
    claims = verification_root / "claims"
    if not verifications.is_file() or not (claims / "manifest.json").is_file():
        if verification_root.exists():
            raise ValueError("partial verification output requires operator review")
        temporary = Path(
            tempfile.mkdtemp(prefix=".verification.", dir=lifecycle)
        )
        try:
            command = [
                sys.executable,
                str(Path(__file__).with_name("verify_datasheet_frontier_teachers.py")),
                "--bundle",
                str(prepared),
                "--reconciliation",
                str(reconciliation),
            ]
            for queue in args.work_queue:
                command.extend(["--work-queue", str(queue)])
            command.extend(
                [
                    "--page-evidence",
                    str(args.page_evidence),
                    "--corpus-registry",
                    str(args.corpus_registry),
                    "--ground-truth-root",
                    str(args.ground_truth_root),
                    "--verifications-output",
                    str(temporary / "verifications.jsonl"),
                    "--claims-output",
                    str(temporary / "claims"),
                ]
            )
            if args.pillar_evidence is not None:
                command.extend(
                    ["--pillar-evidence", str(args.pillar_evidence)]
                )
            _run(command)
            os.replace(temporary, verification_root)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    factory.record_frontier_status(
        args.run_id,
        status_payload,
        stage="verified",
    )

    finalization = lifecycle / "finalization"
    if not (finalization / "manifest.json").is_file():
        command = [
            sys.executable,
            str(batch_script),
            "finalize",
            "--bundle",
            str(prepared),
            "--reconciliation",
            str(reconciliation),
            "--verifications",
            str(verifications),
        ]
        for local_results in args.local_results:
            command.extend(["--local-results", str(local_results)])
        command.extend(["--output-directory", str(finalization)])
        _run(command)
    factory.record_frontier_status(
        args.run_id,
        status_payload,
        stage="finalized",
    )

    sft_pairs = _line_count(finalization / "training-pairs.jsonl")
    dpo_pairs = _line_count(
        finalization / "preference-training-pairs.jsonl"
    )
    ready = (
        sft_pairs >= args.minimum_sft_pairs
        and dpo_pairs >= args.minimum_dpo_pairs
    )
    receipt = None
    if ready and args.training_ready_output is not None:
        receipt = _write_ready_receipt(
            args.training_ready_output,
            run_id=args.run_id,
            finalization=finalization,
            sft_pairs=sft_pairs,
            dpo_pairs=dpo_pairs,
        )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": "finalized",
                "prepared_evidence_sha256": manifest["evidence_sha256"],
                "counts": {"sft": sft_pairs, "dpo": dpo_pairs},
                "training_ready": ready,
                "training_ready_receipt": receipt,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ready else 3


if __name__ == "__main__":
    raise SystemExit(main())
