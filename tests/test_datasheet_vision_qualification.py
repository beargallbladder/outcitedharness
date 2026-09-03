from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_datasheet_vision_qualification.py"
SPEC = importlib.util.spec_from_file_location("datasheet_qualification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def _result(path: Path, *, model: str, scores: dict[str, tuple[float, bool]]):
    core = {
        "schema": qualification.RESULT_SCHEMA,
        "model": model,
        "fixture_sha256": "a" * 64,
        "configuration": {"modes": ["image_rows"]},
        "cases": [
            {
                "id": case_id,
                "modalities": {
                    "image_rows": {
                        "score": {"pair_f1": score},
                        "contract_valid": valid,
                        "physical_identity_error": None if valid else "invalid",
                    }
                },
            }
            for case_id, (score, valid) in scores.items()
        ],
    }
    core["identity"] = {
        "source_gold_set_sha256": "b" * 64,
        "core_sha256": hashlib.sha256(
            qualification._canonical(core)
        ).hexdigest(),
    }
    path.write_text(json.dumps(core))


def test_frozen_holdout_requires_gain_and_no_case_regression(tmp_path: Path):
    baseline = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    _result(
        baseline,
        model="base",
        scores={"part-a": (0.8, True), "part-b": (0.7, True)},
    )
    _result(
        candidate,
        model="candidate",
        scores={"part-a": (0.82, True), "part-b": (0.71, True)},
    )

    result = qualification.compare(
        baseline_path=baseline,
        candidate_path=candidate,
        minimum_mean_delta=0.0,
        maximum_case_regression=0.02,
    )

    assert result["passed"] is True
    assert result["metrics"]["mean_pair_f1_delta"] == pytest.approx(0.015)


def test_contract_failure_rejects_candidate(tmp_path: Path):
    baseline = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    _result(baseline, model="base", scores={"part-a": (0.8, True)})
    _result(candidate, model="candidate", scores={"part-a": (0.9, False)})

    result = qualification.compare(
        baseline_path=baseline,
        candidate_path=candidate,
        minimum_mean_delta=0.0,
        maximum_case_regression=0.02,
    )

    assert result["passed"] is False


def test_evaluation_tampering_is_rejected(tmp_path: Path):
    baseline = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    _result(baseline, model="base", scores={"part-a": (0.8, True)})
    _result(candidate, model="candidate", scores={"part-a": (0.9, True)})
    value = json.loads(candidate.read_text())
    value["cases"][0]["modalities"]["image_rows"]["score"]["pair_f1"] = 1.0
    candidate.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="digest mismatch"):
        qualification.compare(
            baseline_path=baseline,
            candidate_path=candidate,
            minimum_mean_delta=0.0,
            maximum_case_regression=0.02,
        )
