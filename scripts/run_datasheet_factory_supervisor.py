#!/usr/bin/env python3
"""Continuously advance sealed teacher runs into a qualified training handoff."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from harness.electronics.claims import canonical_json
from harness.electronics.factory_control import ElectronicsFactoryState


SCHEMA = "harness.electronics-factory-supervisor.v1"
_STOP = False


def _stop(_signum: int, _frame: object) -> None:
    global _STOP
    _STOP = True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--stop-when-ready", action="store_true")
    return parser


def _load_config(path: Path) -> tuple[dict[str, Any], Path]:
    config_path = path.expanduser().resolve(strict=True)
    if path.expanduser().is_symlink() or not config_path.is_file():
        raise ValueError("supervisor config must be a regular non-symlink file")
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported factory supervisor config")
    expected = value.get("evidence_sha256")
    core = {
        key: item
        for key, item in value.items()
        if key != "evidence_sha256"
    }
    if hashlib.sha256(canonical_json(core)).hexdigest() != expected:
        raise ValueError("factory supervisor config evidence is invalid")
    if not isinstance(value.get("runs"), list):
        raise ValueError("factory supervisor runs must be a list")
    dynamic_directories = [
        *(
            [value["dynamic_run_directory"]]
            if value.get("dynamic_run_directory")
            else []
        ),
        *(value.get("dynamic_run_directories") or []),
    ]
    if not value["runs"] and not dynamic_directories:
        raise ValueError("factory supervisor config has no frontier runs")
    return value, config_path.parents[1]


def _path(repository: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repository / path).resolve()


def _dynamic_runs(
    config: Mapping[str, Any],
    repository: Path,
) -> list[dict[str, Any]]:
    raw_directories = [
        *(
            [config["dynamic_run_directory"]]
            if config.get("dynamic_run_directory")
            else []
        ),
        *(config.get("dynamic_run_directories") or []),
    ]
    if not raw_directories:
        return []
    runs: list[dict[str, Any]] = []
    seen_directories: set[Path] = set()
    for raw_directory in raw_directories:
        directory = _path(repository, str(raw_directory))
        if directory in seen_directories:
            continue
        seen_directories.add(directory)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory.is_symlink():
            raise ValueError(
                "dynamic frontier run directory cannot be a symlink"
            )
        for path in sorted(directory.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"unsafe dynamic run config: {path}")
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                value.get("schema")
                != "harness.electronics-frontier-run-config.v1"
            ):
                raise ValueError(f"unsupported dynamic run config: {path}")
            core = {
                key: item
                for key, item in value.items()
                if key != "evidence_sha256"
            }
            if hashlib.sha256(canonical_json(core)).hexdigest() != value.get(
                "evidence_sha256"
            ):
                raise ValueError(
                    f"dynamic run config evidence is invalid: {path}"
                )
            if not isinstance(value.get("run"), dict):
                raise ValueError(f"dynamic run config has no run object: {path}")
            runs.append(value["run"])
    return runs


def _run(command: list[str]) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    if completed.returncode not in {0, 3}:
        payload["error"] = completed.stderr[-4000:]
    return completed.returncode, payload


def _advance_command(
    repository: Path,
    state_root: Path,
    run: Mapping[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        str(repository / "scripts" / "advance_datasheet_frontier_run.py"),
        "--state-root",
        str(state_root),
        "--run-id",
        str(run["run_id"]),
        "--prepared-bundle",
        str(_path(repository, str(run["prepared_bundle"]))),
        "--submission-state",
        str(_path(repository, str(run["submission_state"]))),
        "--lifecycle-root",
        str(_path(repository, str(run["lifecycle_root"]))),
        "--page-evidence",
        str(_path(repository, str(run["page_evidence"]))),
        "--corpus-registry",
        str(_path(repository, str(run["corpus_registry"]))),
        "--ground-truth-root",
        str(_path(repository, str(run["ground_truth_root"]))),
        "--input-price-per-million",
        str(run["input_price_per_million"]),
        "--output-price-per-million",
        str(run["output_price_per_million"]),
        "--batch-discount",
        str(run.get("batch_discount", 0.5)),
        "--minimum-sft-pairs",
        str(run.get("minimum_sft_pairs", 1)),
        "--minimum-dpo-pairs",
        str(run.get("minimum_dpo_pairs", 1)),
    ]
    for queue in run["work_queues"]:
        command.extend(["--work-queue", str(_path(repository, str(queue)))])
    for local in run.get("local_results", []):
        command.extend(["--local-results", str(_path(repository, str(local)))])
    if run.get("pillar_evidence"):
        command.extend(
            [
                "--pillar-evidence",
                str(_path(repository, str(run["pillar_evidence"]))),
            ]
        )
    if run.get("training_ready_output"):
        command.extend(
            [
                "--training-ready-output",
                str(_path(repository, str(run["training_ready_output"]))),
            ]
        )
    if run.get("api_key_env"):
        command.extend(["--api-key-env", str(run["api_key_env"])])
    if run.get("base_url"):
        command.extend(["--base-url", str(run["base_url"])])
    if run.get("timeout_seconds"):
        command.extend(["--timeout-seconds", str(run["timeout_seconds"])])
    return command


def _handoff_command(
    repository: Path,
    handoff: Mapping[str, Any],
    run_finalizations: list[Path],
) -> list[str]:
    command = [
        sys.executable,
        str(
            repository
            / "scripts"
            / "prepare_electronics_30b_training_handoff.py"
        ),
    ]
    for bundle in [
        *(
            _path(repository, str(value))
            for value in handoff.get("static_bundles", [])
        ),
        *run_finalizations,
    ]:
        command.extend(["--bundle", str(bundle)])
    for cohort in handoff["frozen_cohorts"]:
        command.extend(
            ["--frozen-cohort", str(_path(repository, str(cohort)))]
        )
    command.extend(
        [
            "--dataset-directory",
            str(_path(repository, str(handoff["dataset_directory"]))),
            "--handoff-directory",
            str(_path(repository, str(handoff["handoff_directory"]))),
            "--candidate-id",
            str(handoff["candidate_id"]),
            "--validation-fraction",
            str(handoff.get("validation_fraction", 0.2)),
            "--split-seed",
            str(handoff.get("split_seed", "electronics-teacher-v2")),
            "--minimum-sft-pairs",
            str(handoff.get("minimum_sft_pairs", 256)),
            "--minimum-dpo-pairs",
            str(handoff.get("minimum_dpo_pairs", 192)),
            "--minimum-lineages",
            str(handoff.get("minimum_lineages", 100)),
            "--sft-epochs",
            str(handoff.get("sft_epochs", 3)),
            "--dpo-epochs",
            str(handoff.get("dpo_epochs", 2)),
        ]
    )
    for capability, count in sorted(
        handoff.get("minimum_sft_capabilities", {}).items()
    ):
        command.extend(
            ["--minimum-sft-capability", f"{capability}={count}"]
        )
    for capability, count in sorted(
        handoff.get("minimum_dpo_capabilities", {}).items()
    ):
        command.extend(
            ["--minimum-dpo-capability", f"{capability}={count}"]
        )
    return command


def _write_status(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _tick(
    config: Mapping[str, Any],
    repository: Path,
    state_root: Path,
) -> dict[str, Any]:
    run_results = []
    finalizations: dict[str, Path] = {}
    runs = [*config["runs"], *_dynamic_runs(config, repository)]
    run_ids = [str(run["run_id"]) for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("factory supervisor has duplicate run IDs")
    for run in runs:
        code, payload = _run(_advance_command(repository, state_root, run))
        lifecycle = _path(repository, str(run["lifecycle_root"]))
        finalization = lifecycle / "finalization"
        finalized = (finalization / "manifest.json").is_file()
        if finalized:
            finalizations[str(run["run_id"])] = finalization
        run_results.append(
            {
                "run_id": run["run_id"],
                "return_code": code,
                "finalized": finalized,
                "result": payload,
            }
        )

    handoff_result: dict[str, Any] | None = None
    ready = False
    handoff = config.get("handoff")
    handoff_run_ids = (
        [
            str(value)
            for value in handoff.get(
                "run_ids",
                [run["run_id"] for run in config["runs"]],
            )
        ]
        if isinstance(handoff, Mapping)
        else []
    )
    handoff_runs_finalized = bool(handoff_run_ids) and all(
        run_id in finalizations for run_id in handoff_run_ids
    )
    if handoff_runs_finalized and isinstance(handoff, Mapping):
        code, payload = _run(
            _handoff_command(
                repository,
                handoff,
                [finalizations[run_id] for run_id in handoff_run_ids],
            )
        )
        ready = code == 0 and payload.get("status") == "ready_to_stage"
        handoff_result = {"return_code": code, "result": payload}
    factory = ElectronicsFactoryState(state_root).status()
    return {
        "schema": "harness.electronics-factory-supervisor-status.v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "ready_to_stage": ready,
        "runs": run_results,
        "handoff": handoff_result,
        "factory": factory,
    }


def main() -> int:
    args = _parser().parse_args()
    config, repository = _load_config(args.config)
    state_root = _path(repository, str(config["state_root"]))
    state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = state_root / "supervisor.lock"
    lock_descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT,
        0o600,
    )
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another factory supervisor is active") from exc
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
        interval = int(config.get("poll_seconds", 300))
        if not 30 <= interval <= 86_400:
            raise ValueError("poll_seconds must be within 30..86400")
        status_path = state_root / "supervisor-status.json"
        while not _STOP:
            status = _tick(config, repository, state_root)
            _write_status(status_path, status)
            print(json.dumps(status, ensure_ascii=False, sort_keys=True), flush=True)
            if (
                args.once
                or (args.stop_when_ready and status["ready_to_stage"])
            ):
                return 0
            deadline = time.monotonic() + interval
            while not _STOP and time.monotonic() < deadline:
                time.sleep(min(1.0, deadline - time.monotonic()))
        return 0
    finally:
        os.close(lock_descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
