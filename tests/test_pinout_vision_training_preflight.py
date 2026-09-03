from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_pinout_vision_training_preflight.py"
PILOT = (
    ROOT
    / "deploy"
    / "training"
    / "llamafactory_pinout_rows_qwen3_vl_8b_pilot.yaml"
)
CANDIDATE = (
    ROOT
    / "deploy"
    / "training"
    / "llamafactory_pinout_rows_qwen3_vl_8b_candidate.yaml"
)
MM_CANDIDATE = (
    ROOT
    / "deploy"
    / "training"
    / "llamafactory_pinout_rows_qwen3_vl_8b_mm_candidate.yaml"
)
SPEC = importlib.util.spec_from_file_location("pinout_vision_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def test_pilot_config_is_pinned_to_bounded_recipe(tmp_path: Path) -> None:
    root = tmp_path / "training"
    config_root = root / "configs"
    config_root.mkdir(parents=True)
    config = config_root / "pilot.yaml"
    config.write_text(PILOT.read_text())

    value, output = preflight._config(root, config, "pilot")

    assert value["max_steps"] == 8
    assert value["max_samples"] == 32
    assert output == (
        root / "checkpoints" / "pinout-rows-qwen3-vl-8b-pilot-v1"
    )


def test_pilot_config_rejects_unbounded_or_wrong_dataset(tmp_path: Path) -> None:
    root = tmp_path / "training"
    config_root = root / "configs"
    config_root.mkdir(parents=True)
    value = yaml.safe_load(PILOT.read_text())
    value["max_samples"] = 1000
    value["dataset"] = "unreviewed_data"
    config = config_root / "pilot.yaml"
    config.write_text(yaml.safe_dump(value))

    with pytest.raises(ValueError, match="dataset must be"):
        preflight._config(root, config, "pilot")


def test_candidate_is_bounded_to_capability_floor(tmp_path: Path) -> None:
    root = tmp_path / "training"
    config_root = root / "configs"
    config_root.mkdir(parents=True)
    config = config_root / "candidate.yaml"
    config.write_text(CANDIDATE.read_text())

    value, output = preflight._config(root, config, "candidate")

    assert value["max_samples"] == 1101
    assert value["max_steps"] == 1101
    assert output.name == "pinout-rows-qwen3-vl-8b-candidate-v1"


def test_multimodal_candidate_trains_visual_and_projector_lora(
    tmp_path: Path,
) -> None:
    root = tmp_path / "training"
    config_root = root / "configs"
    config_root.mkdir(parents=True)
    config = config_root / "mm-candidate.yaml"
    config.write_text(MM_CANDIDATE.read_text())

    value, output = preflight._config(root, config, "mm_candidate")

    assert value["freeze_vision_tower"] is False
    assert value["freeze_multi_modal_projector"] is False
    assert value["learning_rate"] == 0.00002
    assert output.name == "pinout-rows-qwen3-vl-8b-mm-candidate-v1"


def test_preflight_receipt_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    preflight.write_new(path, {"passed": True})

    assert json.loads(path.read_text()) == {"passed": True}
    assert path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(ValueError, match="already exists"):
        preflight.write_new(path, {"passed": False})
