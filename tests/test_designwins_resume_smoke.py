from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "designwins_resume_smoke.py"
SPEC = importlib.util.spec_from_file_location("designwins_resume_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
resume_smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume_smoke)


def _checkpoint(root: Path, step: int, *, optimizer: bool = True) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step})
    )
    if optimizer:
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    return checkpoint


def test_resume_smoke_prepares_one_step_and_verifies_hashes(tmp_path: Path):
    source = tmp_path / "source.yaml"
    source.write_text("output_dir: /old\nnum_train_epochs: 1\n")
    checkpoints = tmp_path / "checkpoints"
    _checkpoint(checkpoints, 65)
    (checkpoints / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoints / "trainer_state.json").write_text(
        json.dumps({"global_step": 65})
    )
    output = tmp_path / "resume"
    config = tmp_path / "resume.yaml"
    summary = tmp_path / "summary.json"

    assert resume_smoke.prepare(source, checkpoints, output, config) == 65
    prepared = yaml.safe_load(config.read_text())
    assert prepared["resume_from_checkpoint"].endswith("checkpoint-65")
    assert prepared["max_steps"] == 66
    assert prepared["output_dir"] == str(output)

    _checkpoint(output, 66, optimizer=False)
    resume_smoke.verify(source, checkpoints, output, config, summary)
    result = json.loads(summary.read_text())
    assert result["passed"] is True
    assert result["source_step"] == 65
    assert result["resumed_step"] == 66
    assert len(result["sha256"]["source_optimizer"]) == 64
    assert (
        result["sha256"]["candidate_adapter"]
        == result["sha256"]["source_adapter"]
    )


def test_resume_smoke_requires_optimizer_state(tmp_path: Path):
    source = tmp_path / "source.yaml"
    source.write_text("output_dir: /old\n")
    checkpoints = tmp_path / "checkpoints"
    _checkpoint(checkpoints, 12, optimizer=False)

    with pytest.raises(ValueError, match="optimizer.pt"):
        resume_smoke.prepare(
            source,
            checkpoints,
            tmp_path / "resume",
            tmp_path / "resume.yaml",
        )
