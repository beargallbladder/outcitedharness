from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEPLOY = ROOT / "deploy" / "training"


def run(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_shell_assets_parse():
    for name in (
        "training_prepare_storage.sh",
        "training_link_doctor.sh",
        "training_launch_single.sh",
        "training_launch_two_node.sh",
        "training_launch_lora.sh",
    ):
        result = run("bash", "-n", str(SCRIPTS / name))
        assert result.returncode == 0, result.stderr


def test_storage_plan_is_noop_and_remote_targets_are_restricted(tmp_path: Path):
    target = tmp_path / "training-root"
    plan = run(
        "bash",
        str(SCRIPTS / "training_prepare_storage.sh"),
        "--role",
        "dgx2",
        "--root",
        str(target),
    )
    assert plan.returncode == 0, plan.stderr
    assert "PLAN only" in plan.stdout
    assert not target.exists()

    forbidden = run(
        "bash",
        str(SCRIPTS / "training_prepare_storage.sh"),
        "--role",
        "dgx2",
        "--host",
        "asus2",
        "--root",
        str(target),
        "--apply",
    )
    assert forbidden.returncode == 2
    assert "must name the selected role" in forbidden.stderr
    assert not target.exists()


def test_launch_templates_default_to_plan_without_side_effects(tmp_path: Path):
    single = run(
        "bash",
        str(SCRIPTS / "training_launch_single.sh"),
        "--root",
        str(tmp_path / "store"),
        "--config",
        str(tmp_path / "store" / "configs" / "job.yaml"),
    )
    assert single.returncode == 0, single.stderr
    assert "PLAN only" in single.stdout

    two_node = run(
        "bash",
        str(SCRIPTS / "training_launch_two_node.sh"),
        "--node-rank",
        "1",
        "--store-root",
        str(tmp_path / "store"),
        "--scratch-root",
        str(tmp_path / "scratch"),
        "--config",
        str(tmp_path / "store" / "configs" / "job.yaml"),
        "--master-addr",
        "10.77.0.1",
        "--interface",
        "enp65s0f0",
    )
    assert two_node.returncode == 0, two_node.stderr
    assert "PLAN only" in two_node.stdout

    lora = run(
        "bash",
        str(SCRIPTS / "training_launch_lora.sh"),
        "--root",
        str(tmp_path / "store"),
        "--config",
        str(tmp_path / "store" / "configs" / "lora.yaml"),
    )
    assert lora.returncode == 0, lora.stderr
    assert "PLAN only" in lora.stdout


def test_manifest_is_deterministic_idempotent_and_detects_tampering(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"weights")
    (artifact / "metadata.json").write_text('{"step": 1}\n')
    manifest = tmp_path / "manifest.json"

    create = run(
        sys.executable,
        str(SCRIPTS / "training_manifest.py"),
        "create",
        str(artifact),
        str(manifest),
    )
    assert create.returncode == 0, create.stderr
    first = manifest.read_bytes()
    data = json.loads(first)
    assert data["schema"] == "harness.training.manifest.v1"
    assert {entry["path"] for entry in data["files"]} == {
        "metadata.json",
        "weights.bin",
    }

    again = run(
        sys.executable,
        str(SCRIPTS / "training_manifest.py"),
        "create",
        str(artifact),
        str(manifest),
    )
    assert again.returncode == 0, again.stderr
    assert "unchanged" in again.stdout
    assert manifest.read_bytes() == first

    verify = run(
        sys.executable,
        str(SCRIPTS / "training_manifest.py"),
        "verify",
        str(artifact),
        str(manifest),
    )
    assert verify.returncode == 0, verify.stderr

    (artifact / "weights.bin").write_bytes(b"tampered")
    tampered = run(
        sys.executable,
        str(SCRIPTS / "training_manifest.py"),
        "verify",
        str(artifact),
        str(manifest),
    )
    assert tampered.returncode == 1
    assert "mismatch" in tampered.stderr

    overwrite = run(
        sys.executable,
        str(SCRIPTS / "training_manifest.py"),
        "create",
        str(artifact),
        str(manifest),
    )
    assert overwrite.returncode == 2
    assert "different content" in overwrite.stderr


def test_manifest_rejects_symlinked_artifacts(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    source = tmp_path / "outside.bin"
    source.write_bytes(b"outside")
    (artifact / "linked.bin").symlink_to(source)

    result = run(
        sys.executable,
        str(SCRIPTS / "training_manifest.py"),
        "create",
        str(artifact),
        str(tmp_path / "manifest.json"),
    )
    assert result.returncode == 2
    assert "symlinked file" in result.stderr


def test_templates_pin_caches_floor_and_non_destructive_controls():
    storage = (DEPLOY / "storage.env.example").read_text()
    for variable in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "TORCH_EXTENSIONS_DIR",
        "XDG_CACHE_HOME",
    ):
        assert f"{variable}=" in storage
    assert "DGX2_MIN_CAPACITY_BYTES=4000000000000" in storage
    assert "ASUS1_MIN_FREE_BYTES=268435456000" in storage

    prepare = (SCRIPTS / "training_prepare_storage.sh").read_text()
    scratch = (SCRIPTS / "training_check_scratch.py").read_text()
    doctor = (SCRIPTS / "training_link_doctor.sh").read_text()
    two_node = (SCRIPTS / "training_launch_two_node.sh").read_text()
    lora = (SCRIPTS / "training_launch_lora.sh").read_text()
    combined = "\n".join((prepare, scratch, doctor, two_node, lora))
    assert ".harness-training-owner-v1" in prepare
    assert "refusing to claim non-empty unowned directory" in prepare
    assert "250 * 1024**3" in scratch
    assert "rdma link show" in doctor
    assert "expected_mtu" in doctor
    assert "torch.distributed.is_nccl_available()" in doctor
    assert "NCCL_SOCKET_IFNAME" in two_node
    assert "training_check_scratch.py" in two_node
    assert '"$specforge" train' in two_node
    assert "torchrun" not in two_node
    assert "--network none" in lora
    for forbidden in (
        "systemctl stop",
        "systemctl restart",
        "systemctl disable",
        "rm -rf",
    ):
        assert forbidden not in combined
