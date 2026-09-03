#!/usr/bin/env python3
"""Process one sealed download cohort through local vision and frontier submit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from harness.electronics.claims import canonical_json
from harness.electronics.corpus import sha256_file
from harness.electronics.factory_control import ElectronicsFactoryState


def _worker(value: str) -> str:
    name, separator, url = value.partition("=")
    if (
        not separator
        or not name
        or not url.startswith(("http://", "https://"))
    ):
        raise argparse.ArgumentTypeError("worker must be NAME=BASE_URL")
    return f"{name}={url.rstrip('/')}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--pdf-root", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--validated-root", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--worker", action="append", type=_worker, default=[])
    parser.add_argument("--local-model", required=True)
    parser.add_argument("--frontier-model", required=True)
    parser.add_argument("--input-price-per-million", type=float, required=True)
    parser.add_argument("--output-price-per-million", type=float, required=True)
    parser.add_argument("--batch-discount", type=float, default=0.5)
    parser.add_argument("--spend-cap-usd", type=float, required=True)
    parser.add_argument("--index-workers", type=int, default=6)
    parser.add_argument("--evidence-workers", type=int, default=6)
    parser.add_argument("--maximum-pages-per-lane", type=int, default=24)
    parser.add_argument("--maximum-work", type=int, default=5000)
    parser.add_argument(
        "--pages-per-document-capability",
        type=int,
        default=24,
    )
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--render-dpi", type=int, default=220)
    parser.add_argument("--request-timeout-seconds", type=float, default=900)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--submit", action="store_true")
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
            f"cohort stage failed ({completed.returncode}): "
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


def _stage(
    marker: Path,
    command: list[str],
) -> dict[str, Any]:
    if marker.exists():
        return {"status": "already_complete", "marker": str(marker)}
    return _run(command)


def main() -> int:
    args = _parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    root = args.cohort_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink():
        raise ValueError("cohort root cannot be a symlink")
    state = ElectronicsFactoryState(args.state_root)
    scripts = repository / "scripts"
    stages: dict[str, Any] = {}

    registry = root / "corpus-registry.json"
    stages["corpus"] = _stage(
        registry,
        [
            sys.executable,
            str(scripts / "build_incremental_datasheet_corpus.py"),
            "--source-snapshot",
            str(args.source_snapshot),
            "--pdf-root",
            str(args.pdf_root),
            "--ground-truth-root",
            str(args.ground_truth_root),
            "--validated-root",
            str(args.validated_root),
            "--output",
            str(registry),
        ],
    )

    page_index = root / "page-index"
    stages["page_index"] = _stage(
        page_index / "manifest.json",
        [
            sys.executable,
            str(scripts / "index_datasheet_pages.py"),
            "--corpus-registry",
            str(registry),
            "--output-directory",
            str(page_index),
            "--workers",
            str(args.index_workers),
        ],
    )

    page_evidence = root / "page-evidence"
    stages["page_evidence"] = _stage(
        page_evidence / "manifest.json",
        [
            sys.executable,
            str(scripts / "extract_datasheet_page_evidence.py"),
            "--page-index",
            str(page_index),
            "--output-directory",
            str(page_evidence),
            "--maximum-pages-per-lane",
            str(args.maximum_pages_per_lane),
            "--workers",
            str(args.evidence_workers),
        ],
    )

    deterministic = root / "deterministic"
    stages["deterministic"] = _stage(
        deterministic / "manifest.json",
        [
            sys.executable,
            str(scripts / "parse_datasheet_page_evidence.py"),
            "--page-evidence",
            str(page_evidence),
            "--holdout",
            str(args.holdout),
            "--shadow-all-model-lanes",
            "--output-directory",
            str(deterministic),
        ],
    )

    priority = root / "priority-queue.json"
    stages["priority"] = _stage(
        priority,
        [
            sys.executable,
            str(scripts / "prioritize_datasheet_local_work.py"),
            "--deterministic-bundle",
            str(deterministic),
            "--corpus-registry",
            str(registry),
            "--maximum-work",
            str(args.maximum_work),
            "--pages-per-document-capability",
            str(args.pages_per_document_capability),
            "--capability",
            "pin_or_ball",
            "--capability",
            "pin_semantics",
            "--capability",
            "parametrics",
            "--capability",
            "series_summary",
            "--capability",
            "opn_decoder",
            "--output",
            str(priority),
        ],
    )

    structural = root / "structural-queue.json"
    stages["structural"] = _stage(
        structural,
        [
            sys.executable,
            str(scripts / "build_datasheet_structural_work_queue.py"),
            "--priority-queue",
            str(priority),
            "--page-evidence",
            str(page_evidence),
            "--page-index",
            str(page_index),
            "--maximum-work",
            str(args.maximum_work),
            "--capability",
            "pin_or_ball",
            "--capability",
            "pin_semantics",
            "--capability",
            "parametrics",
            "--capability",
            "series_summary",
            "--capability",
            "opn_decoder",
            "--package-scope-policy",
            "require",
            "--output",
            str(structural),
        ],
    )
    queue = json.loads(structural.read_text(encoding="utf-8"))
    work_count = len(queue.get("work") or [])
    if work_count == 0:
        print(
            json.dumps(
                {
                    "status": "complete_without_model_work",
                    "cohort_id": args.cohort_id,
                    "stages": stages,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "ready_for_local_extraction",
                    "cohort_id": args.cohort_id,
                    "structural_queue": str(structural),
                    "work_items": work_count,
                    "stages": stages,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    local_output = root / "local-chunks"
    pool_command = [
        sys.executable,
        str(scripts / "run_datasheet_factory_pool.py"),
        "--state-root",
        str(args.state_root),
        "--structural-queue",
        str(structural),
        "--page-evidence",
        str(page_evidence),
        "--output-root",
        str(local_output),
        "--model",
        args.local_model,
        "--chunk-size",
        str(args.chunk_size),
        "--render-dpi",
        str(args.render_dpi),
        "--request-timeout-seconds",
        str(args.request_timeout_seconds),
        "--vision-policy",
        "always",
    ]
    for worker in args.worker:
        pool_command.extend(["--worker", worker])
    if not (root / "local-complete.json").exists():
        result = _run(pool_command)
        _write_new(root / "local-complete.json", result)
        stages["local_extraction"] = result
    else:
        stages["local_extraction"] = {"status": "already_complete"}

    queue_sha = sha256_file(structural)
    bundles = state.completed_bundle_paths(queue_sha)
    if not bundles:
        raise RuntimeError("no completed local bundles were registered")
    candidates = root / "frontier-candidates"
    if not (candidates / "manifest.json").exists():
        command = [
            sys.executable,
            str(scripts / "build_datasheet_frontier_candidates.py"),
            "--work-queue",
            str(structural),
        ]
        for bundle in bundles:
            command.extend(["--local-bundle", str(bundle)])
        command.extend(
            [
                "--require-complete",
                "--output-directory",
                str(candidates),
            ]
        )
        stages["frontier_candidates"] = _run(command)
    else:
        stages["frontier_candidates"] = {"status": "already_complete"}

    prepared = root / "frontier-prepared"
    if not (prepared / "manifest.json").exists():
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
        stages["frontier_prepare"] = _run(command)
    else:
        stages["frontier_prepare"] = {"status": "already_complete"}

    submission = root / "frontier-submission"
    if args.submit:
        stages["frontier_submit"] = _run(
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
                "--resume",
            ]
        )
    else:
        stages["frontier_submit"] = {"status": "plan_only"}

    run_config = None
    if args.submit:
        run_core = {
            "schema": "harness.electronics-frontier-run-config.v1",
            "run": {
                "run_id": args.cohort_id,
                "prepared_bundle": str(prepared),
                "submission_state": str(submission),
                "lifecycle_root": str(root / "frontier-lifecycle"),
                "work_queues": [str(structural)],
                "page_evidence": str(page_evidence),
                "pillar_evidence": str(
                    candidates / "pillar-evidence.jsonl"
                ),
                "corpus_registry": str(registry),
                "ground_truth_root": str(
                    args.ground_truth_root.expanduser().resolve(strict=True)
                ),
                "local_results": [
                    str(candidates / "local-results.jsonl")
                ],
                "input_price_per_million": args.input_price_per_million,
                "output_price_per_million": args.output_price_per_million,
                "batch_discount": args.batch_discount,
                "minimum_sft_pairs": 1,
                "minimum_dpo_pairs": 1,
                "api_key_env": "ANTHROPIC_API_KEY",
                "timeout_seconds": args.request_timeout_seconds,
            },
        }
        run_config = {
            **run_core,
            "evidence_sha256": hashlib.sha256(
                canonical_json(run_core)
            ).hexdigest(),
        }
        config_path = (
            state.root
            / "frontier-run-configs"
            / f"{args.cohort_id}.json"
        )
        if not config_path.exists():
            _write_new(config_path, run_config)

    print(
        json.dumps(
            {
                "status": "submitted" if args.submit else "ready_to_submit",
                "cohort_id": args.cohort_id,
                "work_items": work_count,
                "local_bundles": len(bundles),
                "frontier_run": run_config,
                "stages": stages,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
