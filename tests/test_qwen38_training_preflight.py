from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from verify_qwen38_training_preflight import (  # noqa: E402
    TP_RECEIPT_SCHEMA,
    verify,
)


def _config(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {
                    "ple_layer_ids": [2],
                    "ngram_size": 3,
                    "heads_per_ngram": 8,
                    "ple_embed_dim": 2560,
                    "ngram_vocab_size_base": 20_000_000,
                    "make_ngram_vocab_size_divisible_by": 128,
                    "split_ngram_parts": 128,
                },
            }
        )
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(path: Path, config: Path) -> Path:
    core = {
        "schema": TP_RECEIPT_SCHEMA,
        "model_config_sha256": _sha256(config),
        "world_size": 4,
        "ple_sharded": True,
        "load_passed": True,
        "optimizer_step_passed": True,
        "finite_gradients": True,
        "adapter_save_passed": True,
        "max_peak_memory_gib": 104.0,
    }
    core["evidence_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(core))
    return path


def test_vanilla_zero3_is_blocked_by_exact_ple_geometry(tmp_path: Path):
    result = verify(
        config_path=_config(tmp_path / "config.json"),
        strategy="vanilla-zero3",
        world_size=4,
        node_memory_gib=128,
    )

    assert result["passed"] is False
    assert result["ple"]["rows_per_table"] == [320_001_536]
    assert result["ple"]["row_width"] == 160
    assert result["ple"]["full_bf16_bytes"] == 102_400_491_520
    assert "gathers the full PLE" in result["blockers"][0]


def test_native_tp_load_probe_is_allowed_but_training_needs_receipt(
    tmp_path: Path,
):
    config = _config(tmp_path / "config.json")
    load = verify(
        config_path=config,
        strategy="native-tp-load",
        world_size=4,
        node_memory_gib=128,
    )
    train = verify(
        config_path=config,
        strategy="native-tp-lora",
        world_size=4,
        node_memory_gib=128,
    )

    assert load["passed"] is True
    assert load["scope"] == "load-and-forward-probe-only"
    assert load["ple"]["column_shard_width"] == 40
    assert train["passed"] is False
    assert "compatibility receipt is absent" in " ".join(train["blockers"])


def test_native_tp_training_requires_untampered_mechanical_receipt(
    tmp_path: Path,
):
    config = _config(tmp_path / "config.json")
    receipt = _receipt(tmp_path / "receipt.json", config)

    passed = verify(
        config_path=config,
        strategy="native-tp-lora",
        world_size=4,
        node_memory_gib=128,
        receipt_path=receipt,
    )
    assert passed["passed"] is True

    payload = json.loads(receipt.read_text())
    payload["finite_gradients"] = False
    receipt.write_text(json.dumps(payload))
    rejected = verify(
        config_path=config,
        strategy="native-tp-lora",
        world_size=4,
        node_memory_gib=128,
        receipt_path=receipt,
    )
    assert rejected["passed"] is False
    assert "digest is invalid" in " ".join(rejected["blockers"])
    assert "finite_gradients" in " ".join(rejected["blockers"])


def test_native_tp_probe_is_four_rank_load_only():
    root = Path(__file__).resolve().parents[1]
    probe = (root / "scripts" / "qwen38_tp_load_smoke.py").read_text()
    lora_probe = (root / "scripts" / "qwen38_tp_lora_smoke.py").read_text()
    image = (
        root / "deploy" / "training" / "Dockerfile.qwen38-lora"
    ).read_text()

    assert 'tp_plan="auto"' in probe
    assert 'attn_implementation="sdpa"' in probe
    assert "trust_remote_code=False" in probe
    assert "world_size != 4" in probe
    assert "native TP did not shard the runtime PLE table" in probe
    assert "torch.no_grad()" in probe
    assert "get_peft_model" not in probe
    assert "harness.qwen38-tp-lora-smoke.v1" in lora_probe
    assert "tp_plan=\"auto\"" in lora_probe
    assert "EXPECTED_MODULES = 228" in lora_probe
    assert "EXPECTED_TRAINABLE_ELEMENTS = 13_052_928" in lora_probe
    assert "dist.all_reduce(parameter.grad" in lora_probe
    assert "get_peft_model" not in lora_probe
    assert "deepspeed" not in lora_probe.casefold()
    assert "fla-core==0.5.2" in image
    assert "ple-sharded-lora-receipt-required" in image


def test_native_tp_launcher_defaults_to_safe_plan(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts" / "training_launch_qwen38_tp_load_smoke.sh"
    result = subprocess.run(
        [
            "bash",
            str(launcher),
            "--node-rank",
            "3",
            "--root",
            str(tmp_path / "training"),
            "--master-addr",
            "10.77.0.1",
            "--interface",
            "enp1s0f1np1",
            "--hcas",
            "rocep1s0f1,roceP2p1s0f1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "PLAN only" in result.stdout
    assert "rank=3/4 host=asus3 gid=3" in result.stdout
    source = launcher.read_text()
    assert "--strategy native-tp-load" in source
    assert "--read-only" in source
    assert "deepspeed" not in source.casefold()


def test_lora_launcher_requires_load_receipt_and_one_short_step(
    tmp_path: Path,
):
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts" / "training_launch_qwen38_tp_lora_smoke.sh"
    result = subprocess.run(
        [
            "bash",
            str(launcher),
            "--node-rank",
            "0",
            "--root",
            str(tmp_path / "training"),
            "--master-addr",
            "10.77.0.1",
            "--interface",
            "enP2p1s0f1np1",
            "--hcas",
            "roceP2p1s0f1,rocep1s0f1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "PLAN only" in result.stdout
    assert "rank=0/4 host=dgx2 gid=5 seq=8" in result.stdout
    source = launcher.read_text()
    assert "harness.qwen38-tp-load-smoke.v1" in source
    assert "--sequence-length 8" in source
    assert "--strategy native-tp-load" in source
    assert "curriculum" in source
    assert "deepspeed" not in source.casefold()
