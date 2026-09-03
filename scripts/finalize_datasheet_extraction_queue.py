#!/usr/bin/env python3
"""Wait for leased extraction chunks, then seal and submit all teacher work."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from harness.electronics.claims import canonical_json
from harness.electronics.corpus import sha256_file
from harness.electronics.factory_control import ElectronicsFactoryState


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--work-queue", type=Path, required=True)
    parser.add_argument("--manual-bundle", action="append", type=Path, default=[])
    parser.add_argument("--expected-managed-items", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--page-evidence", type=Path, required=True)
    parser.add_argument("--corpus-registry", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--frontier-model", required=True)
    parser.add_argument("--input-price-per-million", type=float, required=True)
    parser.add_argument("--output-price-per-million", type=float, required=True)
    parser.add_argument("--batch-discount", type=float, default=0.5)
    parser.add_argument("--spend-cap-usd", type=float, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--wait-timeout-seconds", type=int, default=43_200)
    parser.add_argument("--api-key-env", default="ANTHROPIC_API_KEY")
    return parser


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
            f"finalization stage failed ({completed.returncode}): "
            f"{' '.join(command)}\n{completed.stderr[-4000:]}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"stdout": completed.stdout[-4000:]}


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _wait_for_chunks(
    state: ElectronicsFactoryState,
    queue_sha: str,
    *,
    expected_items: int,
    poll_seconds: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    if poll_seconds < 5:
        raise ValueError("poll_seconds must be at least 5")
    deadline = time.monotonic() + timeout_seconds
    while True:
        summary = state.queue_chunk_summary(queue_sha)
        if summary["registered_items"] == 0:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"extraction queue registration timed out: {summary}"
                )
            time.sleep(poll_seconds)
            continue
        if summary["registered_items"] != expected_items:
            raise RuntimeError(
                "managed extraction item count differs from expectation: "
                f"{summary}"
            )
        if summary["chunks"].get("failed", 0):
            raise RuntimeError(f"one or more extraction chunks failed: {summary}")
        completed = summary["items"].get("completed", 0)
        if completed == expected_items:
            return summary
        if time.monotonic() >= deadline:
            raise TimeoutError(f"extraction wait timed out: {summary}")
        time.sleep(poll_seconds)


def _verify_manual_bundles(
    paths: list[Path],
    queue_sha: str,
) -> list[Path]:
    bundles = []
    for raw in paths:
        path = raw.expanduser().resolve(strict=True)
        if raw.expanduser().is_symlink() or not path.is_dir():
            raise ValueError(f"manual bundle is unsafe: {raw}")
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema")
            != "harness.electronics-structural-local-extraction.v1"
            or manifest.get("sources", {}).get("structural_queue_sha256")
            != queue_sha
        ):
            raise ValueError(f"manual bundle used a different queue: {path}")
        bundles.append(path)
    return bundles


def main() -> int:
    args = _parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    scripts = repository / "scripts"
    queue = args.work_queue.expanduser().resolve(strict=True)
    queue_sha = sha256_file(queue)
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output.is_symlink():
        raise ValueError("output root cannot be a symlink")
    state = ElectronicsFactoryState(args.state_root)
    chunk_summary = _wait_for_chunks(
        state,
        queue_sha,
        expected_items=args.expected_managed_items,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.wait_timeout_seconds,
    )
    bundles = [
        *_verify_manual_bundles(args.manual_bundle, queue_sha),
        *state.completed_bundle_paths(queue_sha),
    ]

    candidates = output / "frontier-candidates"
    if not (candidates / "manifest.json").is_file():
        command = [
            sys.executable,
            str(scripts / "build_datasheet_frontier_candidates.py"),
            "--work-queue",
            str(queue),
        ]
        for bundle in bundles:
            command.extend(["--local-bundle", str(bundle)])
        command.extend(
            ["--require-complete", "--output-directory", str(candidates)]
        )
        _run(command)

    prepared = output / "frontier-prepared"
    if not (prepared / "manifest.json").is_file():
        command = [
            sys.executable,
            str(scripts / "datasheet_frontier_batch.py"),
            "prepare",
            "--candidates",
            str(candidates / "candidates.jsonl"),
        ]
        for bundle in bundles:
            command.extend(["--allowed-root", str(bundle)])
        command.extend(
            [
                "--model",
                args.frontier_model,
                "--input-price-per-million",
                str(args.input_price_per_million),
                "--output-price-per-million",
                str(args.output_price_per_million),
                "--batch-discount",
                str(args.batch_discount),
                "--spend-cap-usd",
                str(args.spend_cap_usd),
                "--output",
                str(prepared),
            ]
        )
        _run(command)

    submission = output / "frontier-submission"
    submission_result = _run(
        [
            sys.executable,
            str(scripts / "datasheet_frontier_batch.py"),
            "submit",
            "--bundle",
            str(prepared),
            "--state-directory",
            str(submission),
            "--approved-spend-cap-usd",
            str(args.spend_cap_usd),
            "--api-key-env",
            args.api_key_env,
            "--resume",
        ]
    )

    run_core = {
        "schema": "harness.electronics-frontier-run-config.v1",
        "run": {
            "run_id": args.run_id,
            "prepared_bundle": str(prepared),
            "submission_state": str(submission),
            "lifecycle_root": str(output / "frontier-lifecycle"),
            "work_queues": [str(queue)],
            "page_evidence": str(
                args.page_evidence.expanduser().resolve(strict=True)
            ),
            "pillar_evidence": str(candidates / "pillar-evidence.jsonl"),
            "corpus_registry": str(
                args.corpus_registry.expanduser().resolve(strict=True)
            ),
            "ground_truth_root": str(
                args.ground_truth_root.expanduser().resolve(strict=True)
            ),
            "local_results": [str(candidates / "local-results.jsonl")],
            "input_price_per_million": args.input_price_per_million,
            "output_price_per_million": args.output_price_per_million,
            "batch_discount": args.batch_discount,
            "minimum_sft_pairs": 1,
            "minimum_dpo_pairs": 1,
            "api_key_env": args.api_key_env,
            "timeout_seconds": 900,
        },
    }
    run_config = {
        **run_core,
        "evidence_sha256": hashlib.sha256(canonical_json(run_core)).hexdigest(),
    }
    config_path = state.root / "frontier-run-configs" / f"{args.run_id}.json"
    if not config_path.exists():
        _write_new(config_path, run_config)
    receipt = {
        "schema": "harness.electronics-extraction-queue-finalization.v1",
        "run_id": args.run_id,
        "queue_sha256": queue_sha,
        "chunk_summary": chunk_summary,
        "manual_bundles": len(args.manual_bundle),
        "managed_bundles": len(
            state.completed_bundle_paths(queue_sha)
        ),
        "frontier_prepared_evidence_sha256": json.loads(
            (prepared / "manifest.json").read_text(encoding="utf-8")
        )["evidence_sha256"],
        "submission": submission_result,
        "run_config": str(config_path),
    }
    receipt["evidence_sha256"] = hashlib.sha256(
        canonical_json(receipt)
    ).hexdigest()
    receipt_path = output / "queue-finalization.json"
    if not receipt_path.exists():
        _write_new(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
