from __future__ import annotations

from harness.electronics.qualification import FactoryThresholds, qualify


def _thresholds() -> FactoryThresholds:
    return FactoryThresholds(
        pin_identity_f1=0.98,
        pin_semantic_macro_f1=0.90,
        parametric_accuracy=0.95,
        summary_claim_precision=0.95,
        evidence_grounding_rate=1.0,
        abstention_precision=0.98,
        maximum_hallucination_rate=0.01,
        minimum_utility_gain=0.02,
        maximum_lane_regression=0.01,
        minimum_paid_call_replacement_rate=0.70,
        maximum_cost_per_admitted_pair_usd=0.20,
    )


def _metrics(value: float):
    return {
        "aggregate": {
            "pin_identity_f1": value,
            "pin_semantic_macro_f1": value,
            "parametric_accuracy": value,
            "summary_claim_precision": value,
            "evidence_grounding_rate": 1.0,
            "abstention_precision": 0.99,
            "hallucination_rate": 0.005,
            "paid_call_replacement_rate": 0.75,
        }
    }


def test_factory_retain_requires_gain_reproducibility_cost_and_cr_bundle():
    report = qualify(
        baseline=_metrics(0.92),
        candidate=_metrics(0.99),
        reproducibility={
            "first_output_sha256": "a" * 64,
            "second_output_sha256": "a" * 64,
        },
        frontier={
            "actual_batch_cost_usd": 10,
            "admitted_training_pairs": 100,
        },
        admissions={
            "cr_package_records": 20,
            "validated_cr_package_records": 20,
            "direct_database_write": False,
        },
        thresholds=_thresholds(),
    )

    assert report["decision"] == "retain"
    assert report["promotion_allowed"] is True


def test_factory_rejects_paid_call_replacement_claim_without_proof():
    candidate = _metrics(0.99)
    candidate["aggregate"]["paid_call_replacement_rate"] = 0.5
    report = qualify(
        baseline=_metrics(0.92),
        candidate=candidate,
        reproducibility={
            "first_output_sha256": "a" * 64,
            "second_output_sha256": "a" * 64,
        },
        frontier={
            "actual_batch_cost_usd": 10,
            "admitted_training_pairs": 100,
        },
        admissions={
            "cr_package_records": 20,
            "validated_cr_package_records": 20,
            "direct_database_write": False,
        },
        thresholds=_thresholds(),
    )

    assert report["decision"] == "reject"
    assert report["gates"]["paid_call_replacement_rate"]["passed"] is False
