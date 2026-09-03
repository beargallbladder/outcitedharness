#!/usr/bin/env python3
"""Run the one allowlisted DesignWins v4 training job for a durable worker."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_JOB_KIND = "electronics_designwins_v4"
EXPECTED_DATASET_VERSION = "designwins-text-v4-20260831"
EXPECTED_IMAGE = (
    "sha256:728b352622d569b1343cb384dd8aa394917203d02631322b02d302d068e38139"
)
ALLOWED_CONFIG = {
    "recipe_relative": "configs/llamafactory_designwins_text_v4_full.yaml",
    "sequence_audit_relative": (
        "manifests/designwins-v4-20260831.sequence-audit.json"
    ),
    "dataset_artifact_manifest_relative": (
        "manifests/designwins-v4-20260831.artifact.sha256.json"
    ),
    "dataset_relative": "datasets/designwins-v4-20260831",
    "output_relative": (
        "checkpoints/designwins-text-qwen3-8b-v4-full-20260831"
    ),
    "image": EXPECTED_IMAGE,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _owned_path(root: Path, relative: str) -> Path:
    if (
        not relative
        or relative.startswith("/")
        or ".." in Path(relative).parts
        or not re.fullmatch(r"[A-Za-z0-9._/-]+", relative)
    ):
        raise ValueError("job contains an unsafe relative path")
    path = (root / relative).resolve(strict=False)
    if not path.is_relative_to(root):
        raise ValueError("job path escapes the training root")
    return path


def _yaml_scalar(path: Path, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}\s*:\s*(.*?)\s*$")
    values = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        match = pattern.fullmatch(line)
        if match:
            values.append(match.group(1).strip().strip("\"'"))
    if len(values) != 1:
        raise ValueError(f"recipe requires exactly one {key}")
    return values[0]


def _write_once(path: Path, value: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(value, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_once(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _complete_checkpoint(output: Path) -> bool:
    adapter = output / "adapter_model.safetensors"
    state_path = output / "trainer_state.json"
    if not adapter.is_file() or not state_path.is_file():
        return False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    step = state.get("global_step")
    maximum = state.get("max_steps")
    return (
        isinstance(step, int)
        and isinstance(maximum, int)
        and maximum > 0
        and step == maximum
    )


def _resume_recipe(
    root: Path,
    source: Path,
    output: Path,
    *,
    job_id: str,
    attempt: int,
) -> Path:
    checkpoints: list[tuple[int, Path]] = []
    for path in output.glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if all(
            (path / name).is_file()
            for name in (
                "adapter_model.safetensors",
                "optimizer.pt",
                "trainer_state.json",
            )
        ):
            checkpoints.append((step, path))
    if not checkpoints:
        raise ValueError("partial output has no resumable checkpoint")
    step, checkpoint = max(checkpoints)
    state = json.loads(
        (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    if state.get("global_step") != step:
        raise ValueError("resume checkpoint name and state disagree")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", job_id):
        raise ValueError("job ID is unsafe for a retry recipe")
    retry = root / "configs" / f"{job_id}.attempt-{attempt}.yaml"
    value = source.read_text(encoding="utf-8")
    if re.search(r"^resume_from_checkpoint\s*:", value, flags=re.MULTILINE):
        raise ValueError("source recipe unexpectedly contains resume state")
    _write_text_once(
        retry,
        value.rstrip()
        + "\nresume_from_checkpoint: "
        + f"/training/{checkpoint.relative_to(root).as_posix()}\n",
    )
    return retry


def run() -> dict[str, str]:
    root = Path(os.environ.get("HARNESS_TRAINING_ROOT", Path.home() / "harness-training"))
    root = root.expanduser().resolve(strict=True)
    payload_path = Path(os.environ["HARNESS_TRAINING_JOB_PAYLOAD"])
    result_path = Path(os.environ["HARNESS_TRAINING_RESULT"])
    if payload_path.is_symlink() or not payload_path.is_file():
        raise ValueError("worker payload is missing or unsafe")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    job = payload.get("job") if isinstance(payload, dict) else None
    if not isinstance(job, dict):
        raise ValueError("worker payload has no job")
    if job.get("job_kind") != EXPECTED_JOB_KIND:
        raise ValueError("handler received the wrong job kind")
    if job.get("dataset_version_id") != EXPECTED_DATASET_VERSION:
        raise ValueError("handler received the wrong dataset version")
    config = job.get("config")
    if config != ALLOWED_CONFIG:
        raise ValueError("job configuration differs from the allowlisted experiment")

    recipe = _owned_path(root, config["recipe_relative"])
    audit = _owned_path(root, config["sequence_audit_relative"])
    dataset = _owned_path(root, config["dataset_relative"])
    dataset_manifest = _owned_path(
        root, config["dataset_artifact_manifest_relative"]
    )
    output = _owned_path(root, config["output_relative"])
    database = root / "ledger" / "learning.db"
    for path in (recipe, audit, dataset_manifest, database):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required artifact is missing or unsafe: {path}")
    configured_output = _yaml_scalar(recipe, "output_dir")
    expected_container_output = f"/training/{config['output_relative']}"
    if configured_output != expected_container_output:
        raise ValueError("recipe output does not match the allowlisted checkpoint")
    if _yaml_scalar(recipe, "do_train") != "true":
        raise ValueError("allowlisted v4 recipe is not enabled for training")

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "training_manifest.py"),
            "verify",
            str(dataset),
            str(dataset_manifest),
        ],
        check=True,
    )
    complete = output.exists() and _complete_checkpoint(output)
    if not complete:
        attempt = int(os.environ["HARNESS_TRAINING_ATTEMPT"])
        launch_recipe = recipe
        if output.exists():
            if attempt <= 1:
                raise FileExistsError(
                    f"first attempt refuses existing checkpoint: {output}"
                )
            launch_recipe = _resume_recipe(
                root,
                recipe,
                output,
                job_id=str(job["job_id"]),
                attempt=attempt,
            )
        subprocess.run(
            [
                str(root / "scripts" / "training_launch_lora.sh"),
                "--root",
                str(root),
                "--config",
                str(launch_recipe),
                "--image",
                config["image"],
                "--sequence-audit",
                str(audit),
                "--database",
                str(database),
                "--dataset-version-id",
                EXPECTED_DATASET_VERSION,
                "--launch",
            ],
            check=True,
        )
    adapter = output / "adapter_model.safetensors"
    trainer_state = output / "trainer_state.json"
    if (
        not adapter.is_file()
        or not trainer_state.is_file()
        or not _complete_checkpoint(output)
    ):
        raise ValueError("training completed without final checkpoint artifacts")
    checkpoint_manifest = (
        root
        / "manifests"
        / f"{os.environ['HARNESS_TRAINING_JOB_ID']}.checkpoint.sha256.json"
    )
    if not checkpoint_manifest.exists():
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "training_manifest.py"),
                "create",
                str(output),
                str(checkpoint_manifest),
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "training_manifest.py"),
            "verify",
            str(output),
            str(checkpoint_manifest),
        ],
        check=True,
    )
    result = {
        "checkpoint_uri": adapter.resolve(strict=True).as_uri(),
        "checkpoint_sha256": _sha256(adapter),
    }
    _write_once(result_path, result)
    return result


def main() -> int:
    print(json.dumps(run(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
