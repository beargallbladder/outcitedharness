from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_owned_code_curriculum_qualification.py"
SPEC = importlib.util.spec_from_file_location("curriculum_qualification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def _evaluation(
    model: str,
    outcomes: list[bool],
    *,
    patch_rate: float = 1.0,
    latency: float = 100.0,
) -> dict:
    core = {
        "schema": "harness.owned-code-curriculum-evaluation.v1",
        "model": model,
        "model_endpoint_sha256": hashlib.sha256(model.encode()).hexdigest(),
        "generation_config": {
            "temperature": 0,
            "seed": 0,
            "max_completion_tokens": 2048,
            "thinking": False,
        },
        "sample_count": len(outcomes),
        "passed": sum(outcomes),
        "patch_applied": round(len(outcomes) * patch_rate),
        "verified_success_rate": sum(outcomes) / len(outcomes),
        "patch_application_rate": patch_rate,
        "median_latency_ms": latency,
        "p95_latency_ms": latency,
        "cases": [
            {"case_id": f"case-{index}", "passed": passed}
            for index, passed in enumerate(outcomes)
        ],
        "dataset_manifest_sha256": "a" * 64,
        "split": "test",
    }
    core["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return core


def test_retains_only_gain_without_regressions() -> None:
    result = qualification.compare(
        baseline=_evaluation("base", [True, False, False]),
        candidate=_evaluation("adapter", [True, True, False]),
        minimum_gain=0.02,
        maximum_latency_regression=0.25,
        minimum_cases=3,
    )

    assert result["passed"] is True
    assert result["action"] == "retain_for_shadow"
    assert result["improvements"] == ["case-1"]
    assert result["regressions"] == []


def test_rejects_candidate_with_any_case_regression() -> None:
    result = qualification.compare(
        baseline=_evaluation("base", [True, False, False]),
        candidate=_evaluation("adapter", [False, True, True]),
        minimum_gain=0.02,
        maximum_latency_regression=0.25,
        minimum_cases=3,
    )

    assert result["passed"] is False
    assert result["action"] == "reject"
    assert result["regressions"] == ["case-0"]


def test_rejects_tampered_evaluation(tmp_path: Path) -> None:
    report = _evaluation("base", [True])
    report["verified_success_rate"] = 0.0
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="digest mismatch"):
        qualification.load(path)
