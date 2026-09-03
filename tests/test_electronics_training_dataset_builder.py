from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from harness.electronics.training_handoff import (
    seal_training_handoff,
    verify_training_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_electronics_training_dataset.py"
SPEC = importlib.util.spec_from_file_location(
    "build_electronics_training_dataset", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _model(provider: str, suffix: str, *, batch: bool = False) -> dict:
    return {
        "provider": provider,
        "model": f"model-{suffix}",
        "revision": None,
        "request_sha256": suffix * 64,
        "response_id": None,
        "batch_id": f"batch-{suffix}" if batch else None,
    }


def _pair(
    image: Path,
    *,
    lineage: str,
    suffix: str,
) -> tuple[dict, dict]:
    image_sha = _sha(image)
    prompt = (
        "Extract capability=pin_or_ball. Return JSON only."
        "\n\nPyMuPDF evidence:\n{\"secret\":\"text shortcut\"}"
    )
    common = {
        "purpose": "local_training_pair_generation",
        "capability": "pin_or_ball",
        "prompt": prompt,
        "source_claim_ids": [f"claim-{suffix * 32}"],
        "lineage_ids": [lineage],
        "image_uris": [image.as_uri()],
        "image_sha256": [image_sha],
        "teacher": _model("anthropic", suffix, batch=True),
        "disposition": "admitted",
    }
    response = json.dumps(
        {"pins": [{"pin_no": 1, "name": f"PIN_{suffix}"}]}
    )
    sft = {
        "schema": "harness.electronics-training-pair.v1",
        "pair_id": f"pair-{suffix * 32}",
        "modality": "vision",
        "response": response,
        "quarantine_reason": None,
        **common,
    }
    dpo = {
        "schema": "harness.electronics-preference-training-pair.v1",
        "pair_id": f"pair-{'f' * 31}{suffix}",
        "training_format": "vision_dpo",
        "chosen_response": response,
        "rejected_response": json.dumps({"pins": []}),
        "chosen_source_sha256": "c" * 64,
        "rejected_source_sha256": "d" * 64,
        "local_model": _model("local", "e"),
        **common,
    }
    return sft, dpo


def _source_bundle(tmp_path: Path) -> tuple[Path, set[str]]:
    source = tmp_path / "source"
    source.mkdir()
    images = []
    pairs = []
    for index, suffix in enumerate(("a", "b"), 1):
        image = tmp_path / f"image-{index}.png"
        image.write_bytes(f"image-{index}".encode())
        images.append(image)
        pairs.append(
            _pair(
                image,
                lineage=suffix * 64,
                suffix=suffix,
            )
        )
    sft_path = source / "training-pairs.jsonl"
    dpo_path = source / "preference-training-pairs.jsonl"
    _write_jsonl(sft_path, [pair[0] for pair in pairs])
    _write_jsonl(dpo_path, [pair[1] for pair in pairs])
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "harness.electronics-frontier-finalization.v1",
                "evidence_sha256": "9" * 64,
                "artifacts": {
                    sft_path.name: {"sha256": _sha(sft_path)},
                    dpo_path.name: {"sha256": _sha(dpo_path)},
                },
            }
        )
    )
    return source, {"a" * 64, "b" * 64}


def _cohort(tmp_path: Path, document: str = "8" * 64) -> Path:
    cohort = tmp_path / f"cohort-{document[0]}"
    cohort.mkdir()
    (cohort / "work-queue.json").write_text(
        json.dumps(
            {
                "policy": {"evaluation_only": True},
                "work": [
                    {
                        "document_sha256": document,
                        "partition": "frozen_evaluation",
                    }
                ],
            }
        )
    )
    (cohort / "manifest.json").write_text(
        json.dumps({"schema": "test-frozen-cohort"})
    )
    return cohort


def test_builder_seals_portable_lineage_split_sft_and_dpo(
    tmp_path: Path,
) -> None:
    source, _ = _source_bundle(tmp_path)
    destination = tmp_path / "dataset"
    manifest = builder.build_dataset(
        [source],
        _cohort(tmp_path),
        destination,
        validation_fraction=0.2,
        split_seed="test",
    )

    assert manifest["counts"]["sft"] == 2
    assert manifest["counts"]["dpo"] == 2
    assert manifest["counts"]["images"] == 2
    assert manifest["counts"]["splits"] in (
        {
            "train": {"sft": 1, "dpo": 1},
            "validation": {"sft": 1, "dpo": 1},
        },
    )
    info = json.loads(
        (destination / "llamafactory" / "dataset_info.json").read_text()
    )
    assert info["electronics_dpo_train"]["ranking"] is True
    lf_rows = []
    for split in ("train", "validation"):
        lf_rows.extend(
            json.loads(
                (
                    destination
                    / "llamafactory"
                    / f"electronics_sft_{split}.json"
                ).read_text()
            )
        )
    assert len(lf_rows) == 2
    assert all(
        row["messages"][0]["content"].count("<image>") == 1
        for row in lf_rows
    )
    assert all(
        "PyMuPDF evidence" not in row["messages"][0]["content"]
        for row in lf_rows
    )
    assert len(list((destination / "images").iterdir())) == 2

    proof = verify_training_dataset(
        destination,
        minimum_sft_pairs=2,
        minimum_dpo_pairs=2,
        minimum_lineages=2,
        minimum_sft_capabilities={"pin_or_ball": 2},
        minimum_dpo_capabilities={"pin_or_ball": 2},
    )
    handoff = tmp_path / "handoff"
    receipt = seal_training_handoff(
        destination,
        handoff,
        candidate_id="electronics-v6-test",
        proof=proof,
    )
    sft = yaml.safe_load((handoff / "sft.yaml").read_text())
    dpo = yaml.safe_load((handoff / "dpo.yaml").read_text())

    assert receipt["decision"] == "ready_to_stage"
    assert receipt["dataset"]["counts"]["lineages"] == 2
    assert sft["dataset_dir"].endswith("/dataset/llamafactory")
    assert sft["freeze_vision_tower"] is False
    assert dpo["adapter_name_or_path"].endswith("electronics-v6-test-sft")
    assert dpo["stage"] == "dpo"


def test_handoff_verifier_rejects_threshold_shortfall(
    tmp_path: Path,
) -> None:
    source, _ = _source_bundle(tmp_path)
    destination = tmp_path / "dataset"
    builder.build_dataset(
        [source],
        _cohort(tmp_path),
        destination,
        validation_fraction=0.2,
        split_seed="test",
    )

    with pytest.raises(ValueError, match="SFT pairs has 2"):
        verify_training_dataset(
            destination,
            minimum_sft_pairs=3,
            minimum_dpo_pairs=2,
            minimum_lineages=2,
        )


def test_handoff_verifier_rejects_tampered_dataset_artifact(
    tmp_path: Path,
) -> None:
    source, _ = _source_bundle(tmp_path)
    destination = tmp_path / "dataset"
    builder.build_dataset(
        [source],
        _cohort(tmp_path),
        destination,
        validation_fraction=0.2,
        split_seed="test",
    )
    (
        destination / "llamafactory" / "electronics_sft_train.json"
    ).write_text("[]")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_training_dataset(
            destination,
            minimum_sft_pairs=1,
            minimum_dpo_pairs=1,
            minimum_lineages=2,
        )


def test_builder_rejects_frozen_document_leakage(tmp_path: Path) -> None:
    source, lineages = _source_bundle(tmp_path)
    with pytest.raises(ValueError, match="training/holdout lineage overlap"):
        builder.build_dataset(
            [source],
            _cohort(tmp_path, sorted(lineages)[0]),
            tmp_path / "leaking-dataset",
        )


def test_builder_unions_multiple_frozen_cohorts(tmp_path: Path) -> None:
    source, lineages = _source_bundle(tmp_path)
    leaking_lineage = sorted(lineages)[0]
    with pytest.raises(ValueError, match="training/holdout lineage overlap"):
        builder.build_dataset(
            [source],
            [
                _cohort(tmp_path, "8" * 64),
                _cohort(tmp_path, leaking_lineage),
            ],
            tmp_path / "multi-cohort-leaking-dataset",
        )
