from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from harness.training.capability import CapabilityEvidence, qualify_target


ROOT = Path(__file__).resolve().parents[1]
LADDER = yaml.safe_load(
    (ROOT / "deploy" / "training" / "capability-ladder.yaml").read_text()
)


def _evidence(target: str, checkpoint_format: str, examples: int):
    gates = frozenset(LADDER["targets"][target]["required_gates"])
    return CapabilityEvidence(
        target=target,
        verified_examples=examples,
        checkpoint_format=checkpoint_format,
        completed_gates=gates,
        evidence_sha256={gate: "a" * 64 for gate in gates},
    )


def test_electronics_target_qualifies_only_with_every_gate():
    evidence = _evidence(
        "electronics_qwen3_8b",
        "bf16_trainable",
        1101,
    )

    decision = qualify_target(LADDER, evidence)

    assert decision.qualified is True
    assert decision.missing_gates == ()


def test_vision_electronics_target_requires_alignment_and_pdf_split():
    evidence = _evidence(
        "electronics_qwen3_vl_8b",
        "bf16_lora",
        1101,
    )

    decision = qualify_target(LADDER, evidence)

    assert decision.qualified is True
    assert {
        "corpus_authorization",
        "exact_page_package_alignment",
        "pdf_lineage_split",
        "resume",
        "frozen_holdout",
    } <= evidence.completed_gates


def test_datasheet_factory_requires_cost_gain_and_import_receipts():
    evidence = _evidence(
        "electronics_datasheet_factory_v1",
        "immutable_claim_training_pairs",
        1101,
    )

    decision = qualify_target(LADDER, evidence)

    assert decision.qualified is True
    assert {
        "authoritative_corpus_join",
        "anthropic_message_batch_only",
        "claim_level_verification",
        "local_capability_gain",
        "paid_call_replacement",
        "immutable_cr_import_bundle",
    } <= evidence.completed_gates


def test_80b_serving_artifact_never_counts_as_trainable():
    evidence = _evidence(
        "qwen3_coder_next_80b",
        "nvfp4_serving",
        500,
    )

    decision = qualify_target(LADDER, evidence)

    assert decision.qualified is False
    assert "not trainable" in " ".join(decision.reasons)


def test_qwen38_requires_ple_sharded_bf16_and_all_distributed_gates():
    unsafe = _evidence(
        "qwen38_flash_next",
        "bf16_trainable",
        500,
    )
    safe = _evidence(
        "qwen38_flash_next",
        "bf16_ple_sharded",
        500,
    )

    rejected = qualify_target(LADDER, unsafe)
    qualified = qualify_target(LADDER, safe)

    assert rejected.qualified is False
    assert "does not match 'bf16_ple_sharded'" in " ".join(rejected.reasons)
    assert qualified.qualified is True


def test_coding_target_blocks_until_verified_backlog_exists():
    evidence = _evidence(
        "coding_qwen_30_35b",
        "supported_4bit",
        499,
    )

    decision = qualify_target(LADDER, evidence)

    assert decision.qualified is False
    assert "requires 500" in " ".join(decision.reasons)


def test_completed_gate_requires_evidence_digest():
    with pytest.raises(ValidationError, match="requires a lowercase SHA-256"):
        CapabilityEvidence(
            target="electronics_qwen3_8b",
            verified_examples=1101,
            checkpoint_format="bf16_trainable",
            completed_gates=frozenset({"model_load"}),
        )


def test_doctor_fails_closed_for_rejected_electronics_evidence():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "training_capability_doctor.py"),
            "--ladder",
            str(ROOT / "deploy" / "training" / "capability-ladder.yaml"),
            "--evidence",
            str(
                ROOT
                / "tests"
                / "fixtures"
                / "electronics_rejected_capability_evidence.json"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    decision = json.loads(result.stdout)
    assert decision["qualified"] is False
    assert set(decision["missing_gates"]) == {
        "frozen_holdout",
        "family_non_regression",
        "deterministic_reproduction",
    }
