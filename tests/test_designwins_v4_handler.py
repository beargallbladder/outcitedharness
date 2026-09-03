from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_designwins_v4_training.py"
SPEC = importlib.util.spec_from_file_location("run_designwins_v4_training", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
handler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handler)


def _fixture(tmp_path: Path, monkeypatch, *, drift: bool = False):
    root = tmp_path / "training"
    for relative in (
        "scripts",
        "configs",
        "manifests",
        "datasets/designwins-v4-20260831",
        "ledger",
        "logs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    for name in ("training_manifest.py", "training_launch_lora.sh"):
        path = root / "scripts" / name
        path.write_text("#!/bin/sh\n")
        path.chmod(0o700)
    (root / "configs" / "llamafactory_designwins_text_v4_full.yaml").write_text(
        "do_train: true\n"
        "output_dir: "
        "/training/checkpoints/designwins-text-qwen3-8b-v4-full-20260831\n"
    )
    (root / "manifests" / "designwins-v4-20260831.sequence-audit.json").write_text(
        "{}"
    )
    (
        root / "manifests" / "designwins-v4-20260831.artifact.sha256.json"
    ).write_text("{}")
    (root / "ledger" / "learning.db").write_bytes(b"db")
    config = dict(handler.ALLOWED_CONFIG)
    if drift:
        config["output_relative"] = "checkpoints/unapproved"
    payload = root / "logs" / "payload.json"
    payload.write_text(
        json.dumps(
            {
                "job": {
                    "job_kind": handler.EXPECTED_JOB_KIND,
                    "job_id": "job-v4",
                    "dataset_version_id": handler.EXPECTED_DATASET_VERSION,
                    "config": config,
                }
            }
        )
    )
    result = root / "logs" / "result.json"
    monkeypatch.setenv("HARNESS_TRAINING_ROOT", str(root))
    monkeypatch.setenv("HARNESS_TRAINING_JOB_PAYLOAD", str(payload))
    monkeypatch.setenv("HARNESS_TRAINING_RESULT", str(result))
    monkeypatch.setenv("HARNESS_TRAINING_JOB_ID", "job-v4")
    monkeypatch.setenv("HARNESS_TRAINING_ATTEMPT", "1")
    return root, result


def test_handler_runs_only_allowlisted_experiment_and_writes_result(
    tmp_path: Path,
    monkeypatch,
):
    root, result_path = _fixture(tmp_path, monkeypatch)
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append([str(value) for value in argv])
        if str(argv[0]).endswith("training_launch_lora.sh"):
            output = (
                root
                / "checkpoints"
                / "designwins-text-qwen3-8b-v4-full-20260831"
            )
            output.mkdir(parents=True)
            (output / "adapter_model.safetensors").write_bytes(b"adapter")
            (output / "trainer_state.json").write_text(
                '{"global_step": 1, "max_steps": 1}'
            )
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(handler.subprocess, "run", fake_run)

    result = handler.run()

    assert result_path.is_file()
    assert result == json.loads(result_path.read_text())
    assert result["checkpoint_sha256"]
    assert any("--dataset-version-id" in call for call in calls)
    assert len(calls) == 4


def test_handler_rejects_job_configuration_drift(tmp_path: Path, monkeypatch):
    _fixture(tmp_path, monkeypatch, drift=True)

    with pytest.raises(ValueError, match="allowlisted experiment"):
        handler.run()


def test_handler_resumes_owned_checkpoint_on_retry(tmp_path: Path, monkeypatch):
    root, _result_path = _fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("HARNESS_TRAINING_ATTEMPT", "2")
    output = (
        root / "checkpoints" / "designwins-text-qwen3-8b-v4-full-20260831"
    )
    checkpoint = output / "checkpoint-4"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"partial")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "trainer_state.json").write_text('{"global_step": 4}')
    launch_configs: list[Path] = []

    def fake_run(argv, **_kwargs):
        if str(argv[0]).endswith("training_launch_lora.sh"):
            config_index = argv.index("--config") + 1
            launch_configs.append(Path(argv[config_index]))
            output.mkdir(exist_ok=True)
            (output / "adapter_model.safetensors").write_bytes(b"complete")
            (output / "trainer_state.json").write_text(
                '{"global_step": 5, "max_steps": 5}'
            )
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(handler.subprocess, "run", fake_run)

    handler.run()

    assert len(launch_configs) == 1
    retry = launch_configs[0]
    assert retry.name == "job-v4.attempt-2.yaml"
    assert (
        "resume_from_checkpoint: "
        "/training/checkpoints/designwins-text-qwen3-8b-v4-full-20260831/"
        "checkpoint-4"
    ) in retry.read_text()
