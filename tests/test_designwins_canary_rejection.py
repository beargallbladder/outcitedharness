from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "record_designwins_canary_rejection.py"
SPEC = importlib.util.spec_from_file_location("canary_rejection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
recorder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recorder)


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def test_failed_canary_is_persisted_as_rejection(tmp_path: Path, monkeypatch):
    canary = _write(
        tmp_path / "canary.json",
        {
            "schema": "harness.designwins.rejection-canary.v1",
            "passed": False,
            "metrics": {
                "candidate_leaf_f1": 0.20,
                "baseline_leaf_f1": 0.80,
                "candidate_exact_rate": 0.10,
                "baseline_exact_rate": 0.70,
                "minimum_required_gain": 0.01,
            },
            "checks": {
                "deterministic_reproduction": True,
                "semantic_gain": False,
            },
        },
    )
    training = _write(
        tmp_path / "training.json",
        {
            "schema": "harness.designwins-teacher-forced-qualification.v1",
            "passed": True,
        },
    )
    resume = _write(tmp_path / "resume.json", {"passed": True})
    captured: dict = {}

    def fake_record_evaluation(_store, **kwargs):
        captured.update(kwargs)
        return "b" * 64

    monkeypatch.setattr(recorder, "record_evaluation", fake_record_evaluation)
    args = argparse.Namespace(
        database=tmp_path / "harness.db",
        canary_qualification=canary,
        training_qualification=training,
        resume_summary=resume,
        job_id="job-1",
        dataset_version_id="dataset-v1",
        candidate_sha256="a" * 64,
        evaluation_id="evaluation-1",
    )

    decision = recorder.record_rejection(args)

    assert decision.action == "reject"
    assert decision.passed is False
    assert captured["stage"] == "offline"
    assert captured["evaluation"].critical_regressions == 1
    assert captured["evaluation"].metadata["training_signal_passed"] is True


def test_recorder_refuses_a_passing_canary(tmp_path: Path):
    canary = _write(tmp_path / "canary.json", {"passed": True})
    other = _write(tmp_path / "other.json", {})
    args = argparse.Namespace(
        database=tmp_path / "harness.db",
        canary_qualification=canary,
        training_qualification=other,
        resume_summary=other,
        job_id="job-1",
        dataset_version_id="dataset-v1",
        candidate_sha256="a" * 64,
        evaluation_id="evaluation-1",
    )

    try:
        recorder.record_rejection(args)
    except ValueError as exc:
        assert "only accepts a failed" in str(exc)
    else:
        raise AssertionError("passing canary was accepted by rejection recorder")
