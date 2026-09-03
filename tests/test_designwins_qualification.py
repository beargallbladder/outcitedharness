from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_designwins_qualification.py"
SPEC = importlib.util.spec_from_file_location(
    "compare_designwins_qualification", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def _evaluation(
    overall_f1: float,
    valid_json: float,
    microchip_f1: float,
    *,
    generated_tokens: int = 20,
    adapter_sha256: str | None = None,
    hit_generation_limit: bool = False,
) -> dict:
    other_f1 = (overall_f1 * 10 - microchip_f1 * 5) / 5
    details = []
    for index in range(10):
        family = "microchip" if index < 5 else "other"
        score = microchip_f1 if family == "microchip" else other_f1
        details.append(
            {
                "index": index,
                "part": f"{family}_part_{index}",
                "family": family,
                "score": {
                    "valid_json": index < int(valid_json * 10),
                    "exact": False,
                    "leaf_precision": score,
                    "leaf_recall": score,
                    "leaf_f1": score,
                },
                "generated_tokens": generated_tokens,
                "generation_budget": 1024,
                "hit_generation_limit": hit_generation_limit,
            }
        )
    evaluation = {
        "model": "/training/model",
        "adapter": "/training/adapter" if adapter_sha256 else None,
        "dataset": "/training/test.json",
        "summary": qualification._summary(details),
        "details": details,
    }
    core_sha256 = qualification.hashlib.sha256(
        qualification._canonical(evaluation)
    ).hexdigest()
    evaluation["identity"] = {
        "schema": "harness.designwins-evaluation-identity.v1",
        "core_sha256": core_sha256,
        "dataset_sha256": "a" * 64,
        "model_manifest_sha256": "b" * 64,
        "scorer_sha256": "c" * 64,
        "adapter_manifest_sha256": adapter_sha256,
        "runtime_image_id": "sha256:" + "e" * 64,
        "max_samples": 10,
        "cutoff_len": 4096,
        "max_new_tokens": 8192,
        "batch_size": 8,
        "generation_slack_tokens": 256,
    }
    return evaluation


def test_qualification_passes_material_reproducible_gain() -> None:
    baseline = _evaluation(0.40, 0.9, 0.50)
    candidate = _evaluation(0.46, 0.9, 0.49, adapter_sha256="d" * 64)
    repeat = _evaluation(0.46, 0.9, 0.49, adapter_sha256="d" * 64)

    result = qualification.compare(
        baseline,
        candidate,
        repeat,
        minimum_f1_gain=0.05,
        maximum_family_regression=0.02,
        minimum_family_samples=5,
        expected_samples=10,
    )

    assert result["passed"] is True
    assert result["checks"]["deterministic_reproduction"] is True


def test_qualification_rejects_family_regression() -> None:
    baseline = _evaluation(0.40, 0.9, 0.50)
    candidate = _evaluation(0.47, 0.9, 0.47, adapter_sha256="d" * 64)

    result = qualification.compare(
        baseline,
        candidate,
        candidate,
        minimum_f1_gain=0.05,
        maximum_family_regression=0.02,
        minimum_family_samples=5,
        expected_samples=10,
    )

    assert result["passed"] is False
    assert result["regressed_families"] == ["microchip"]


def test_qualification_rejects_non_reproducible_candidate() -> None:
    baseline = _evaluation(0.40, 0.9, 0.50)
    candidate = _evaluation(0.46, 0.9, 0.50, adapter_sha256="d" * 64)
    repeat = _evaluation(
        0.46,
        0.9,
        0.50,
        generated_tokens=21,
        adapter_sha256="d" * 64,
    )

    result = qualification.compare(
        baseline,
        candidate,
        repeat,
        minimum_f1_gain=0.05,
        maximum_family_regression=0.02,
        minimum_family_samples=5,
        expected_samples=10,
    )

    assert result["passed"] is False
    assert result["checks"]["deterministic_reproduction"] is False


def test_qualification_identity_binds_all_evaluation_artifacts(
    tmp_path: Path,
) -> None:
    baseline = _evaluation(0.40, 0.9, 0.50)
    candidate = _evaluation(0.46, 0.9, 0.49, adapter_sha256="d" * 64)
    paths = {
        "baseline": tmp_path / "baseline.json",
        "candidate": tmp_path / "candidate.json",
        "repeat": tmp_path / "repeat.json",
    }
    paths["baseline"].write_text(json.dumps(baseline))
    paths["candidate"].write_text(json.dumps(candidate))
    paths["repeat"].write_text(json.dumps(candidate))
    result = qualification.compare(
        baseline,
        candidate,
        candidate,
        minimum_f1_gain=0.05,
        maximum_family_regression=0.02,
        minimum_family_samples=5,
        expected_samples=10,
    )

    bound = qualification.bind_identity(
        result,
        baseline=paths["baseline"],
        candidate=paths["candidate"],
        candidate_repeat=paths["repeat"],
    )

    assert bound["identity"]["schema"].endswith("qualification-identity.v1")
    assert bound["identity"]["core_sha256"] == qualification.hashlib.sha256(
        qualification._canonical(result)
    ).hexdigest()
    assert (
        bound["identity"]["candidate_evaluation_sha256"]
        == bound["identity"]["candidate_repeat_evaluation_sha256"]
    )


def test_baseline_budget_exhaustion_is_measured_but_does_not_block_gain():
    baseline = _evaluation(
        0.10,
        0.1,
        0.10,
        hit_generation_limit=True,
    )
    candidate = _evaluation(0.50, 1.0, 0.50, adapter_sha256="d" * 64)

    result = qualification.compare(
        baseline,
        candidate,
        candidate,
        minimum_f1_gain=0.05,
        maximum_family_regression=0.02,
        minimum_family_samples=5,
        expected_samples=10,
    )

    assert result["passed"] is True
    assert result["metrics"]["baseline_generation_limit_hits"] == 10
    assert result["checks"]["candidate_generation_complete"] is True
