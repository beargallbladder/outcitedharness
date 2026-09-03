from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_pinout_vision_evaluations.py"
SPEC = importlib.util.spec_from_file_location("pinout_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def _evaluation(
    path: Path,
    *,
    adapter: bool,
    identity_f1: float,
    type_accuracy: float,
) -> None:
    aggregate = {
        "examples": 6,
        "json_valid": 6,
        "identity_exact": 4,
        "identity": {"f1": identity_f1},
        "rich": {
            "type_accuracy": type_accuracy,
            "direction_accuracy": 0.5,
            "functions_exact_rate": 0.1,
        },
    }
    core = {
        "schema": comparison.EVALUATION_SCHEMA,
        "cohort": {
            "sha256": "a" * 64,
            "evidence_sha256": "b" * 64,
            "limited": False,
        },
        "model": {
            "config_sha256": "c" * 64,
            "adapter": (
                {"sha256": "d" * 64, "bytes": 100, "path": "/adapter"}
                if adapter
                else None
            ),
        },
        "generation": {"do_sample": False, "maximum_new_tokens": 512},
        "aggregate": aggregate,
        "by_vendor": {
            "vendor": {
                "examples": 6,
                "identity": {"f1": identity_f1},
            }
        },
        "results": [],
    }
    core["evidence_sha256"] = hashlib.sha256(
        comparison._canonical(core)
    ).hexdigest()
    path.write_text(
        json.dumps({"created_at": "2026-09-01T00:00:00Z", **core})
    )


def test_comparison_retains_only_material_nonregressing_candidate(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    _evaluation(base, adapter=False, identity_f1=0.90, type_accuracy=0.10)
    _evaluation(candidate, adapter=True, identity_f1=0.91, type_accuracy=0.40)

    decision = comparison.compare(base_path=base, candidate_path=candidate)

    assert decision["passed"] is True
    assert decision["decision"] == "retain_for_further_qualification"
    assert decision["promotion_authorized"] is False


def test_comparison_rejects_identity_regression(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    _evaluation(base, adapter=False, identity_f1=0.90, type_accuracy=0.10)
    _evaluation(candidate, adapter=True, identity_f1=0.70, type_accuracy=0.80)

    decision = comparison.compare(base_path=base, candidate_path=candidate)

    assert decision["passed"] is False
    assert decision["decision"] == "reject"
    assert any("identity" in reason for reason in decision["reasons"])
