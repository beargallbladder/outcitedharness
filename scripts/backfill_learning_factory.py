#!/usr/bin/env python3
"""Backfill approved sources into the immutable learning ledger."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from harness.config import load_config
from harness.storage.db import Store
from harness.training.backfill import (
    BackfillReport,
    backfill_ci_history,
    backfill_cursor_transcript,
    backfill_designwins,
    backfill_greenfield_history,
    backfill_git_repository,
    backfill_harness_pass_history,
    inventory_harness_learning_gaps,
)
from harness.training.ledger import LearningLedger


def _write_report(path: Path, reports: list[BackfillReport]) -> None:
    payload = {
        "schema": "harness.learning-backfill.v1",
        "reports": [report.as_dict() for report in reports],
        "totals": {
            "captured": sum(report.captured for report in reports),
            "duplicates": sum(report.duplicates for report in reports),
            "rejected": sum(report.rejected for report in reports),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--designwins", action="append", default=[], type=Path)
    parser.add_argument("--designwins-source-snapshot", type=Path)
    parser.add_argument("--admit-audited-designwins", action="store_true")
    parser.add_argument("--git-repo", action="append", default=[], type=Path)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--cursor-transcript", action="append", default=[], type=Path)
    parser.add_argument("--ci-history", action="append", default=[], type=Path)
    parser.add_argument("--harness-cases-root", type=Path)
    parser.add_argument("--harness-answer-root", type=Path)
    parser.add_argument("--greenfield-runs-root", type=Path)
    parser.add_argument("--inventory-harness", action="store_true")
    parser.add_argument("--inventory-ci", action="store_true")
    parser.add_argument("--max-commits", type=int, default=500)
    args = parser.parse_args()
    if not (
        args.designwins
        or args.git_repo
        or args.cursor_transcript
        or args.ci_history
        or args.harness_cases_root
        or args.greenfield_runs_root
        or args.inventory_harness
        or args.inventory_ci
    ):
        parser.error("at least one approved source is required")
    if args.max_commits < 1:
        parser.error("--max-commits must be positive")
    if args.designwins and args.designwins_source_snapshot is None:
        parser.error("--designwins requires --designwins-source-snapshot")
    return args


def main() -> int:
    args = parse_args()
    store = Store(args.database)
    ledger = LearningLedger(store, args.artifact_root)
    reports: list[BackfillReport] = []
    for source in args.designwins:
        reports.append(
            backfill_designwins(
                source,
                ledger,
                source_snapshot=args.designwins_source_snapshot,
                admit_verified=args.admit_audited_designwins,
            )
        )
    approved_repositories: list[str] = []
    if args.git_repo:
        approved_repositories = load_config(
            args.config_root
        ).settings.code_index_repos
    for repository in args.git_repo:
        reports.append(
            backfill_git_repository(
                repository,
                ledger,
                approved_repositories=approved_repositories,
                max_commits=args.max_commits,
            )
        )
    if args.harness_cases_root is not None:
        reports.append(
            backfill_harness_pass_history(
                args.database,
                args.harness_cases_root,
                ledger,
                answer_root=args.harness_answer_root,
            )
        )
    if args.greenfield_runs_root is not None:
        reports.append(
            backfill_greenfield_history(
                args.database,
                ledger,
                runs_root=args.greenfield_runs_root,
            )
        )
    for transcript in args.cursor_transcript:
        reports.append(backfill_cursor_transcript(transcript, ledger))
    for history in args.ci_history:
        reports.append(backfill_ci_history(history, ledger))
    if args.inventory_harness:
        reports.append(
            inventory_harness_learning_gaps(
                store,
                include_pass_answers=args.harness_cases_root is None,
            )
        )
    if args.inventory_ci:
        missing_ci = BackfillReport(source="ci-history")
        missing_ci.reject(
            "no CI history source configured",
            record_id="ci:configuration",
        )
        reports.append(missing_ci)
    _write_report(args.output, reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
