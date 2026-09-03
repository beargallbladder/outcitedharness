from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from harness.training.hygiene import deduplicate_pairs
from harness.training.models import (
    DataUse,
    GitCandidate,
    SourceKind,
    SourceProvenance,
    TestEvidence,
    TextPair,
)
from harness.training.security import assert_no_secrets, redact_text


class AdapterValidationError(ValueError):
    pass


def _sha256(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _readonly_db(path: Path) -> sqlite3.Connection:
    path = path.resolve(strict=True)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _write_models(path: Path, rows: Iterable[TextPair | GitCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row.model_dump(mode="json", exclude_none=True),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_harness_pass_candidates(
    database: Path,
    cases_root: Path,
    *,
    approved_model_keys: frozenset[str] = frozenset(),
    answer_root: Path | None = None,
    destination: Path | None = None,
    strict: bool = True,
    rejections: list[dict[str, Any]] | None = None,
) -> list[TextPair]:
    """Join real Harness PASS rows to prompts, quarantining unapproved models."""

    database = database.resolve(strict=True)
    cases_root = cases_root.resolve(strict=True)
    answer_root = (
        answer_root.resolve(strict=True)
        if answer_root is not None
        else database.parent.resolve(strict=True)
    )
    cases: dict[str, tuple[str, Path]] = {}
    for spec_path in sorted(cases_root.rglob("case.yaml")):
        raw = yaml.safe_load(spec_path.read_text()) or {}
        case_id = str(raw.get("id") or "").strip()
        prompt_path = spec_path.parent / "prompt.md"
        if not case_id or not prompt_path.is_file():
            continue
        if case_id in cases:
            raise AdapterValidationError(f"duplicate Harness case id: {case_id}")
        cases[case_id] = (prompt_path.read_text(), spec_path)

    with _readonly_db(database) as connection:
        rows = connection.execute(
            """
            SELECT id, run_id, case_id, model_key, provider, model, started_at,
                   answer_path, evaluator
            FROM model_results
            WHERE verdict = 'PASS'
              AND error IS NULL
              AND evaluator IS NOT NULL
              AND answer_path IS NOT NULL
            ORDER BY started_at DESC, id DESC
            """
        ).fetchall()

    output: list[TextPair] = []
    seen_case_model: set[tuple[str, str]] = set()
    for row in rows:
        case_id = str(row["case_id"])
        model_key = str(row["model_key"])
        dedupe_key = (case_id, model_key)
        if dedupe_key in seen_case_model:
            if rejections is not None:
                rejections.append(
                    {
                        "record_id": int(row["id"]),
                        "case_id": case_id,
                        "model_key": model_key,
                        "reason": "duplicate case/model PASS row",
                        "duplicate": True,
                    }
                )
            continue
        try:
            if case_id not in cases:
                raise AdapterValidationError(f"missing Harness case: {case_id}")
            prompt, spec_path = cases[case_id]
            answer_path = Path(str(row["answer_path"])).expanduser().resolve(strict=True)
            if answer_path.is_symlink() or not answer_path.is_file():
                raise AdapterValidationError("PASS answer is not a regular file")
            if not answer_path.is_relative_to(answer_root):
                raise AdapterValidationError("PASS answer escapes the approved result root")
            prompt = redact_text(prompt.strip())
            response = redact_text(answer_path.read_text().strip())
            if not prompt or not response:
                raise AdapterValidationError("Harness pair contains empty text")
            assert_no_secrets(prompt, field="Harness prompt")
            assert_no_secrets(response, field="Harness response")
            approved = model_key in approved_model_keys
            data_use = DataUse.TRAINING if approved else DataUse.QUARANTINE
            record_id = f"{row['run_id']}:{case_id}:{model_key}"
            provenance = SourceProvenance(
                source_kind=SourceKind.HARNESS,
                source_uri=spec_path.resolve().as_uri(),
                source_record_id=record_id,
                collected_at=_timestamp(str(row["started_at"])),
                content_sha256=_sha256(prompt, response),
                lineage_id=f"harness:{case_id}",
                license=(
                    "internal-harness-eval-approved"
                    if approved
                    else "model-output-license-review-required"
                ),
                data_use=data_use,
            )
            metadata: dict[str, Any] = {
                "case_id": case_id,
                "run_id": str(row["run_id"]),
                "model_key": model_key,
                "provider": str(row["provider"] or ""),
                "model": str(row["model"] or ""),
                "evaluator": str(row["evaluator"]),
                "verdict": "PASS",
            }
            if not approved:
                metadata["quarantine_reason"] = (
                    "model key was not explicitly approved for output reuse"
                )
            output.append(
                TextPair(
                    pair_id=f"harness-{_sha256(record_id, response)[:24]}",
                    prompt=prompt,
                    response=response,
                    provenance=provenance,
                    data_use=data_use,
                    metadata=metadata,
                )
            )
            seen_case_model.add(dedupe_key)
        except (OSError, ValueError) as error:
            if strict:
                raise
            if rejections is not None:
                rejections.append(
                    {
                        "record_id": int(row["id"]),
                        "case_id": case_id,
                        "model_key": model_key,
                        "reason": str(error),
                    }
                )

    pairs = deduplicate_pairs(output)
    if destination is not None:
        _write_models(destination, pairs)
    return pairs


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise AdapterValidationError(result.stderr.strip()[-1000:] or "git command failed")
    return result.stdout


def load_greenfield_git_candidates(
    database: Path,
    *,
    runs_root: Path,
    destination: Path | None = None,
    strict: bool = True,
    rejections: list[dict[str, Any]] | None = None,
) -> list[GitCandidate]:
    """Extract complete greenfield milestones as review-only git candidates."""

    database = database.resolve(strict=True)
    runs_root = runs_root.expanduser().resolve(strict=True)
    with _readonly_db(database) as connection:
        rows = connection.execute(
            """
            SELECT m.run_id, m.milestone_id, m.objective, m.acceptance_json,
                   m.starting_commit, m.verified_state_hash, m.commit_sha,
                   m.updated_at, r.workspace_root
            FROM greenfield_milestones m
            JOIN greenfield_runs r ON r.run_id = m.run_id
            WHERE m.state = 'complete'
              AND m.commit_sha IS NOT NULL
              AND m.verified_state_hash IS NOT NULL
              AND r.workspace_root IS NOT NULL
            ORDER BY m.run_id, m.ordinal
            """
        ).fetchall()

    output: list[GitCandidate] = []
    for row in rows:
        source_record_id = f"{row['run_id']}:{row['milestone_id']}"
        try:
            repo = Path(str(row["workspace_root"])).expanduser().resolve(strict=True)
            if not repo.is_relative_to(runs_root):
                raise AdapterValidationError("greenfield repository escapes runs root")
            revision = str(row["commit_sha"])
            verified_revision = _git(repo, "rev-parse", "--verify", f"{revision}^{{commit}}")
            if verified_revision.strip() != revision:
                raise AdapterValidationError("greenfield commit is not the recorded revision")
            starting_commit = row["starting_commit"]
            if starting_commit:
                patch = _git(
                    repo,
                    "diff",
                    "--binary",
                    "--no-ext-diff",
                    str(starting_commit),
                    revision,
                ).strip()
            else:
                patch = _git(
                    repo,
                    "show",
                    "--format=",
                    "--binary",
                    "--no-ext-diff",
                    revision,
                ).strip()
            if not patch.startswith("diff --git "):
                raise AdapterValidationError("greenfield milestone has no unified diff")
            problem = str(row["objective"]).strip()
            assert_no_secrets(problem, field="greenfield objective")
            assert_no_secrets(patch, field="greenfield patch")
            acceptance = json.loads(str(row["acceptance_json"]))
            commands = acceptance.get("acceptance_commands")
            if not isinstance(commands, list) or not commands:
                raise AdapterValidationError("milestone has no acceptance commands")
            tests = tuple(
                TestEvidence(command=str(command), status="unknown")
                for command in commands
            )
            provenance = SourceProvenance(
                source_kind=SourceKind.GIT,
                source_uri=repo.as_uri(),
                source_record_id=source_record_id,
                collected_at=_timestamp(str(row["updated_at"])),
                content_sha256=_sha256(problem, patch),
                lineage_id=f"greenfield:{row['run_id']}",
                license="internal-quarantine",
                revision=revision,
                data_use=DataUse.QUARANTINE,
            )
            output.append(
                GitCandidate(
                    candidate_id=f"git-{_sha256(source_record_id, revision)[:24]}",
                    problem=problem,
                    patch=patch,
                    tests=tests,
                    provenance=provenance,
                    quarantine_reason=(
                        "acceptance commands are recorded but command-output digests "
                        "must be independently regenerated before promotion"
                    ),
                )
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            if strict:
                raise
            if rejections is not None:
                rejections.append(
                    {
                        "source_record_id": source_record_id,
                        "reason": str(error),
                    }
                )

    if destination is not None:
        _write_models(destination, output)
    return output
