from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


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
        "training_configure_link.sh",
        "training_launch_nccl_smoke.sh",
        "training_install_specforge.sh",
        "training_launch_single.sh",
        "training_launch_two_node.sh",
        "training_launch_lora.sh",
        "training_launch_pinout_vision_lora.sh",
        "training_launch_pinout_vision_candidate.sh",
        "training_launch_electronics_30b_six_node.sh",
        "training_launch_electronics_30b_candidate.sh",
        "training_launch_electronics_30b_dpo.sh",
        "training_launch_electronics_30b_handoff.sh",
        "training_launch_bge_cr.sh",
        "run_tapes_offline_repro.sh",
        "run_designwins_evaluation.sh",
        "run_designwins_posttrain_qualification.sh",
        "run_designwins_resume_smoke.sh",
        "run_designwins_v4_chunk_evaluation.sh",
        "dgx3_datasheet_vision.sh",
    ):
        result = run("bash", "-n", str(SCRIPTS / name))
        assert result.returncode == 0, result.stderr


def test_six_node_vision_recipe_trains_all_multimodal_blocks():
    recipe = yaml.safe_load(
        (
            DEPLOY
            / "llamafactory_electronics_teacher_qwen3_vl_30b_six_node_smoke.yaml"
        ).read_text()
    )

    assert recipe["model_name_or_path"].endswith(
        "Qwen3-VL-30B-A3B-Instruct-FP8"
    )
    assert recipe["finetuning_type"] == "lora"
    assert recipe["lora_target"] == "all"
    assert recipe["freeze_vision_tower"] is False
    assert recipe["freeze_multi_modal_projector"] is False
    assert recipe["freeze_language_model"] is False
    assert recipe["gradient_checkpointing_kwargs"] == {
        "use_reentrant": False
    }
    assert recipe["ddp_find_unused_parameters"] is False
    assert recipe["max_steps"] == 2


def test_bf16_vision_recipes_use_training_checkpoint_and_v5_teachers():
    smoke = yaml.safe_load(
        (
            DEPLOY
            / "llamafactory_electronics_teacher_qwen3_vl_30b_bf16_six_node_smoke.yaml"
        ).read_text()
    )
    candidate = yaml.safe_load(
        (
            DEPLOY
            / "llamafactory_electronics_teacher_qwen3_vl_30b_bf16_candidate_v1.yaml"
        ).read_text()
    )
    dpo = yaml.safe_load(
        (
            DEPLOY
            / "llamafactory_electronics_teacher_qwen3_vl_30b_bf16_candidate_v2_dpo.yaml"
        ).read_text()
    )
    base_eval = yaml.safe_load(
        (
            DEPLOY
            / "llamafactory_electronics_teacher_qwen3_vl_30b_bf16_base_eval.yaml"
        ).read_text()
    )
    smoke_eval = yaml.safe_load(
        (
            DEPLOY
            / "llamafactory_electronics_teacher_qwen3_vl_30b_bf16_smoke_eval.yaml"
        ).read_text()
    )
    candidate_eval = yaml.safe_load(
        (
            DEPLOY
            / "llamafactory_electronics_teacher_qwen3_vl_30b_bf16_candidate_eval.yaml"
        ).read_text()
    )
    dpo_eval = yaml.safe_load(
        (
            DEPLOY
            / "llamafactory_electronics_teacher_qwen3_vl_30b_bf16_dpo_eval.yaml"
        ).read_text()
    )

    for recipe in (smoke, candidate):
        assert recipe["model_name_or_path"].endswith(
            "Qwen3-VL-30B-A3B-Instruct-BF16"
        )
        assert recipe["dataset_dir"].endswith(
            "datasheet-electronics-teacher-v5/llamafactory"
        )
        assert recipe["lora_target"] == "all"
        assert recipe["freeze_vision_tower"] is False
        assert recipe["freeze_multi_modal_projector"] is False
        assert recipe["freeze_language_model"] is False
        assert recipe["gradient_checkpointing_kwargs"] == {
            "use_reentrant": False
        }
        assert recipe["ddp_find_unused_parameters"] is False
    assert smoke["max_steps"] == 2
    assert candidate["num_train_epochs"] == 3
    assert candidate["max_samples"] == 227
    assert candidate["eval_dataset"] == "electronics_sft_validation"
    assert dpo["stage"] == "dpo"
    assert dpo["adapter_name_or_path"] == candidate["output_dir"]
    assert dpo["dataset"] == "electronics_dpo_train"
    assert dpo["eval_dataset"] == "electronics_dpo_validation"
    assert dpo["pref_loss"] == "sigmoid"
    assert dpo["pref_ftx"] == 0.1
    assert dpo["max_samples"] == 221
    assert base_eval["model_name_or_path"] == smoke["model_name_or_path"]
    assert base_eval["infer_backend"] == "huggingface"
    assert smoke_eval["model_name_or_path"] == smoke["model_name_or_path"]
    assert smoke_eval["adapter_name_or_path"] == smoke["output_dir"]
    assert candidate_eval["model_name_or_path"] == candidate["model_name_or_path"]
    assert candidate_eval["adapter_name_or_path"] == candidate["output_dir"]
    assert dpo_eval["model_name_or_path"] == dpo["model_name_or_path"]
    assert dpo_eval["adapter_name_or_path"] == dpo["output_dir"]


def test_excluded_categoryrank_and_tapes_launchers_fail_closed():
    for name in ("training_launch_bge_cr.sh", "run_tapes_offline_repro.sh"):
        result = run("bash", str(SCRIPTS / name))
        assert result.returncode == 2
        assert "processing is suspended" in result.stderr


def test_excluded_categoryrank_and_tapes_python_entrypoints_fail_closed():
    for name in (
        "export_categoryrank.py",
        "train_categoryrank_persistence_pilot.py",
        "evaluate_tapes_open_set.py",
    ):
        result = run(sys.executable, str(SCRIPTS / name), "--help")
        assert result.returncode == 2
        assert "processing is suspended" in result.stderr


def test_cr_learning_data_request_excludes_suspended_sources():
    request = yaml.safe_load(
        (DEPLOY / "cr-learning-data-request.yaml").read_text()
    )

    assert request["schema"] == "harness.learning-data-request.v1"
    assert {"CategoryRank", "Tapes"} <= set(request["explicitly_excluded"])
    collections = request["requested_collections"]
    assert set(collections) == {
        "verified_code_repairs",
        "electronics_pinout_resolutions",
    }
    assert {
        "parent_commit_sha",
        "repaired_commit_sha",
        "passing_test_output_sha256",
    } <= set(collections["verified_code_repairs"]["required_fields"])
    assert {
        "source_document_sha256",
        "canonical_pinout_json",
        "verification_output_sha256",
    } <= set(collections["electronics_pinout_resolutions"]["required_fields"])
    requested_text = json.dumps(collections).casefold()
    assert "categoryrank" not in requested_text
    assert "tapes" not in requested_text


def test_truncating_v3_recipes_are_permanently_disabled():
    for name in (
        "llamafactory_designwins_text_full.yaml",
        "llamafactory_designwins_text_pilot.yaml",
    ):
        recipe = (DEPLOY / name).read_text()
        assert "### REJECTED:" in recipe
        assert "do_train: false" in recipe
        assert "do_train: true" not in recipe


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
        "enp1s0f1np1",
        "--hcas",
        "rocep1s0f1,roceP2p1s0f1",
    )
    assert two_node.returncode == 0, two_node.stderr
    assert "PLAN only" in two_node.stdout

    link = run(
        "bash",
        str(SCRIPTS / "training_configure_link.sh"),
        "--role",
        "dgx2",
        "--primary-interface",
        "enp1s0f1np1",
        "--primary-cidr",
        "10.77.0.1/24",
        "--secondary-interface",
        "enP2p1s0f1np1",
        "--secondary-cidr",
        "10.77.1.1/24",
    )
    assert link.returncode == 0, link.stderr
    assert "PLAN only" in link.stdout

    nccl = run(
        "bash",
        str(SCRIPTS / "training_launch_nccl_smoke.sh"),
        "--node-rank",
        "0",
        "--root",
        str(tmp_path / "store"),
        "--master-addr",
        "10.77.0.1",
        "--interface",
        "enp1s0f1np1",
        "--hcas",
        "rocep1s0f1,roceP2p1s0f1",
    )
    assert nccl.returncode == 0, nccl.stderr
    assert "PLAN only" in nccl.stdout
    assert "gid=5" in nccl.stdout

    four_node_nccl = run(
        "bash",
        str(SCRIPTS / "training_launch_nccl_smoke.sh"),
        "--node-rank",
        "3",
        "--world-size",
        "4",
        "--root",
        str(tmp_path / "store"),
        "--master-addr",
        "10.77.0.1",
        "--interface",
        "enp1s0f1np1",
        "--hcas",
        "rocep1s0f1,roceP2p1s0f1",
    )
    assert four_node_nccl.returncode == 0, four_node_nccl.stderr
    assert "rank=3/4 host=asus3" in four_node_nccl.stdout
    assert "gid=3" in four_node_nccl.stdout

    six_node_nccl = run(
        "bash",
        str(SCRIPTS / "training_launch_nccl_smoke.sh"),
        "--node-rank",
        "5",
        "--world-size",
        "6",
        "--root",
        str(tmp_path / "store"),
        "--master-addr",
        "10.77.0.1",
        "--interface",
        "enp1s0f1np1",
        "--hcas",
        "rocep1s0f1,roceP2p1s0f1",
    )
    assert six_node_nccl.returncode == 0, six_node_nccl.stderr
    assert "rank=5/6 host=asus4" in six_node_nccl.stdout
    assert "gid=5" in six_node_nccl.stdout

    electronics_30b = run(
        "bash",
        str(SCRIPTS / "training_launch_electronics_30b_six_node.sh"),
        "--node-rank",
        "4",
        "--root",
        str(tmp_path / "store"),
        "--config",
        str(tmp_path / "store" / "configs" / "electronics-30b.yaml"),
        "--model-manifest",
        str(tmp_path / "store" / "manifests" / "qwen3-vl-30b.json"),
        "--master-addr",
        "10.77.0.1",
        "--interface",
        "enp1s0f1np1",
        "--hcas",
        "rocep1s0f1,roceP2p1s0f1",
    )
    assert electronics_30b.returncode == 0, electronics_30b.stderr
    assert "rank=4/6 host=asus2 gid=5" in electronics_30b.stdout
    assert "PLAN only" in electronics_30b.stdout

    electronics_candidate = run(
        "bash",
        str(SCRIPTS / "training_launch_electronics_30b_candidate.sh"),
        "--node-rank",
        "5",
        "--root",
        str(tmp_path / "store"),
        "--config",
        str(tmp_path / "store" / "configs" / "electronics-candidate.yaml"),
        "--model-manifest",
        str(tmp_path / "store" / "manifests" / "qwen3-vl-30b.json"),
        "--master-addr",
        "10.77.0.1",
        "--interface",
        "enp1s0f1np1",
        "--hcas",
        "rocep1s0f1,roceP2p1s0f1",
    )
    assert electronics_candidate.returncode == 0, electronics_candidate.stderr
    assert "rank=5/6 host=asus4 gid=5" in electronics_candidate.stdout
    assert "PLAN only" in electronics_candidate.stdout

    electronics_dpo = run(
        "bash",
        str(SCRIPTS / "training_launch_electronics_30b_dpo.sh"),
        "--node-rank",
        "3",
        "--root",
        str(tmp_path / "store"),
        "--config",
        str(tmp_path / "store" / "configs" / "electronics-dpo.yaml"),
        "--model-manifest",
        str(tmp_path / "store" / "manifests" / "qwen3-vl-30b.json"),
        "--master-addr",
        "10.77.0.1",
        "--interface",
        "enp1s0f1np1",
        "--hcas",
        "rocep1s0f1,roceP2p1s0f1",
    )
    assert electronics_dpo.returncode == 0, electronics_dpo.stderr
    assert "rank=3/6 host=asus3 gid=3" in electronics_dpo.stdout
    assert "PLAN only" in electronics_dpo.stdout

    handoff = tmp_path / "store" / "handoffs" / "electronics-v6"
    handoff.mkdir(parents=True)
    (handoff / "manifest.json").write_text(
        json.dumps({"candidate_id": "electronics-v6-candidate"})
    )
    electronics_handoff = run(
        "bash",
        str(SCRIPTS / "training_launch_electronics_30b_handoff.sh"),
        "--stage",
        "sft",
        "--node-rank",
        "2",
        "--root",
        str(tmp_path / "store"),
        "--handoff",
        str(handoff),
        "--model-manifest",
        str(tmp_path / "store" / "manifests" / "qwen3-vl-30b.json"),
        "--master-addr",
        "10.77.0.1",
        "--interface",
        "enp1s0f1np1",
        "--hcas",
        "rocep1s0f1,roceP2p1s0f1",
    )
    assert electronics_handoff.returncode == 0, electronics_handoff.stderr
    assert "candidate=electronics-v6-candidate stage=sft" in (
        electronics_handoff.stdout
    )
    assert "rank=2/6 host=dgx3 gid=3" in electronics_handoff.stdout
    assert "PLAN only" in electronics_handoff.stdout

    specforge = run(
        "bash",
        str(SCRIPTS / "training_install_specforge.sh"),
        "--role",
        "dgx2",
        "--root",
        str(tmp_path / "store"),
        "--build",
        "--install-wrapper",
    )
    assert specforge.returncode == 0, specforge.stderr
    assert "PLAN only" in specforge.stdout

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

    pinout_lora = run(
        "bash",
        str(SCRIPTS / "training_launch_pinout_vision_lora.sh"),
        "--root",
        str(tmp_path / "store"),
        "--config",
        str(tmp_path / "store" / "configs" / "pinout.yaml"),
        "--mode",
        "pilot",
        "--receipt-output",
        str(tmp_path / "store" / "runs" / "pinout-preflight.json"),
    )
    assert pinout_lora.returncode == 0, pinout_lora.stderr
    assert "PLAN only" in pinout_lora.stdout

    pinout_candidate = run(
        "bash",
        str(SCRIPTS / "training_launch_pinout_vision_candidate.sh"),
        "--root",
        str(tmp_path / "store"),
        "--config",
        str(tmp_path / "store" / "configs" / "candidate.yaml"),
        "--receipt-output",
        str(tmp_path / "store" / "runs" / "candidate-preflight.json"),
    )
    assert pinout_candidate.returncode == 0, pinout_candidate.stderr
    assert "PLAN only" in pinout_candidate.stdout


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


def test_manifest_can_exclude_download_metadata(tmp_path: Path):
    artifact = tmp_path / "model"
    (artifact / ".cache").mkdir(parents=True)
    (artifact / ".cache" / "download.metadata").write_text("transient")
    (artifact / ".gitattributes").write_text("filter=lfs")
    (artifact / "config.json").write_text("{}")
    manifest = tmp_path / "runtime-manifest.json"

    result = run(
        sys.executable,
        str(SCRIPTS / "training_manifest.py"),
        "create",
        str(artifact),
        str(manifest),
        "--exclude-hidden",
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(manifest.read_text())
    assert [entry["path"] for entry in data["files"]] == ["config.json"]


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
    configure_link = (SCRIPTS / "training_configure_link.sh").read_text()
    nccl_smoke = (SCRIPTS / "training_launch_nccl_smoke.sh").read_text()
    nccl_probe = (SCRIPTS / "training_nccl_smoke.py").read_text()
    tapes_repro = (SCRIPTS / "run_tapes_offline_repro.sh").read_text()
    cr_bge = (SCRIPTS / "training_launch_bge_cr.sh").read_text()
    specforge_install = (SCRIPTS / "training_install_specforge.sh").read_text()
    specforge_image = (DEPLOY / "SpecForge.GB10.Dockerfile").read_text()
    two_node = (SCRIPTS / "training_launch_two_node.sh").read_text()
    lora = (SCRIPTS / "training_launch_lora.sh").read_text()
    designwins_preflight = (
        SCRIPTS / "verify_designwins_training_preflight.py"
    ).read_text()
    designwins_eval = (SCRIPTS / "run_designwins_evaluation.sh").read_text()
    designwins_qualification = (
        SCRIPTS / "run_designwins_posttrain_qualification.sh"
    ).read_text()
    designwins_resume = (
        SCRIPTS / "run_designwins_resume_smoke.sh"
    ).read_text()
    designwins_v4_evaluation = (
        SCRIPTS / "run_designwins_v4_chunk_evaluation.sh"
    ).read_text()
    combined = "\n".join(
        (
            prepare,
            scratch,
            doctor,
            configure_link,
            nccl_smoke,
            two_node,
            lora,
            designwins_preflight,
            cr_bge,
            specforge_install,
            designwins_eval,
            designwins_qualification,
            designwins_resume,
            designwins_v4_evaluation,
        )
    )
    assert ".harness-training-owner-v1" in prepare
    assert "refusing to claim non-empty unowned directory" in prepare
    assert "250 * 1024**3" in scratch
    assert "rdma link show" in doctor
    assert "expected_mtu" in doctor
    assert "torch.distributed.is_nccl_available()" in doctor
    assert "nvcr.io/nvidia/pytorch@sha256:" in doctor
    assert "ConnectTimeout=10" in doctor
    assert "nmcli connection modify" in configure_link
    assert "ipv4.never-default yes" in configure_link
    assert "ip address replace" in configure_link
    assert "default-route interface" in configure_link
    assert "nvcr.io/nvidia/pytorch@sha256:" in nccl_smoke
    assert "--device /dev/infiniband" in nccl_smoke
    assert "--world-size" in nccl_smoke
    assert "--gid-index" in nccl_smoke
    assert "--expected-world-size" in nccl_smoke
    assert "dist.all_reduce" in nccl_probe
    assert "expected_sum" in nccl_probe
    assert "incorrect all-reduce result" in nccl_probe
    assert "--network none" in tapes_repro
    assert "127.0.0.1:18881" in tapes_repro
    assert ":8800" not in tapes_repro
    assert "harness/bge-repro-gb10:prod-20260829" in tapes_repro
    assert "BGE_MODEL_RELATIVE" in tapes_repro
    assert "TAPES_REPRO_OUTPUT_RELATIVE" in tapes_repro
    assert "processing is suspended" in tapes_repro
    assert "d822f07c7a0458424daa3cc18b88bb6b936f091acb6bc16cfa9c13c8ab66e61d" in cr_bge
    assert "--network none" in cr_bge
    assert '--user "$(id -u):$(id -g)"' in cr_bge
    assert "category_mentions_v2" not in cr_bge
    assert "processing is suspended" in cr_bge
    assert "c439546983863facd8126f505c2d291d0ab31faf" in specforge_image
    assert "nvcr.io/nvidia/pytorch@sha256:" in specforge_image
    assert "torch.cuda.is_available()" in specforge_install
    assert "--device /dev/infiniband" in specforge_install
    assert "NCCL_SOCKET_IFNAME" in two_node
    assert "NCCL_IB_HCA" in two_node
    assert "NCCL_IB_GID_INDEX" in two_node
    assert "training_check_scratch.py" in two_node
    assert '"$specforge" train' in two_node
    assert "torchrun" not in two_node
    assert "--network none" in lora
    assert '--user "$(id -u):$(id -g)"' in lora
    assert "--env HOME=/tmp" in lora
    assert "--sequence-audit is required for launch" in lora
    assert "--dataset-version-id is required for launch" in lora
    assert "verify_designwins_training_preflight.py" in lora
    assert "truncated records" in designwins_preflight
    assert "--network none" in designwins_eval
    assert "--max-samples 141" in designwins_eval
    assert "1788159600" in designwins_qualification
    assert "candidate-repeat-full-141.json" in designwins_qualification
    assert "designwins-resume-smoke-20260830.json" in designwins_resume
    assert "--network none" in designwins_resume
    assert "--network none" in designwins_v4_evaluation
    assert "chunk_samples=676" in designwins_v4_evaluation
    assert "chunk_samples=19" in designwins_v4_evaluation
    assert "aggregate_designwins_chunk_evaluation.py" in designwins_v4_evaluation
    for forbidden in (
        "systemctl stop",
        "systemctl restart",
        "systemctl disable",
        "rm -rf",
    ):
        assert forbidden not in combined
