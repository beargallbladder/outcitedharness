"""Precommitted qualification gates for the datasheet extraction factory."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from harness.electronics.claims import canonical_json


QUALIFICATION_SCHEMA = "harness.electronics-factory-qualification.v1"


class FactoryThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pin_identity_f1: float = Field(ge=0, le=1)
    pin_semantic_macro_f1: float = Field(ge=0, le=1)
    parametric_accuracy: float = Field(ge=0, le=1)
    summary_claim_precision: float = Field(ge=0, le=1)
    evidence_grounding_rate: float = Field(ge=0, le=1)
    abstention_precision: float = Field(ge=0, le=1)
    maximum_hallucination_rate: float = Field(ge=0, le=1)
    minimum_utility_gain: float = Field(ge=0)
    maximum_lane_regression: float = Field(ge=0, le=1)
    minimum_paid_call_replacement_rate: float = Field(ge=0, le=1)
    maximum_cost_per_admitted_pair_usd: float = Field(gt=0)


LANE_METRICS = (
    "pin_identity_f1",
    "pin_semantic_macro_f1",
    "parametric_accuracy",
    "summary_claim_precision",
)


def qualify(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reproducibility: Mapping[str, Any],
    frontier: Mapping[str, Any],
    admissions: Mapping[str, Any],
    thresholds: FactoryThresholds,
) -> dict[str, Any]:
    baseline_metrics = baseline.get("aggregate")
    candidate_metrics = candidate.get("aggregate")
    if not isinstance(baseline_metrics, Mapping) or not isinstance(
        candidate_metrics, Mapping
    ):
        raise ValueError("baseline and candidate require aggregate metrics")
    gates: dict[str, dict[str, Any]] = {}

    for metric in LANE_METRICS:
        value = float(candidate_metrics.get(metric, -1))
        minimum = float(getattr(thresholds, metric))
        gates[f"minimum_{metric}"] = {
            "passed": value >= minimum,
            "actual": value,
            "minimum": minimum,
        }
        baseline_value = float(baseline_metrics.get(metric, -1))
        delta = value - baseline_value
        gates[f"non_regression_{metric}"] = {
            "passed": delta >= -thresholds.maximum_lane_regression,
            "delta": delta,
            "maximum_regression": thresholds.maximum_lane_regression,
        }

    for metric in ("evidence_grounding_rate", "abstention_precision"):
        value = float(candidate_metrics.get(metric, -1))
        minimum = float(getattr(thresholds, metric))
        gates[metric] = {
            "passed": value >= minimum,
            "actual": value,
            "minimum": minimum,
        }
    hallucination = float(candidate_metrics.get("hallucination_rate", 1))
    gates["hallucination_rate"] = {
        "passed": hallucination <= thresholds.maximum_hallucination_rate,
        "actual": hallucination,
        "maximum": thresholds.maximum_hallucination_rate,
    }
    replacement = float(candidate_metrics.get("paid_call_replacement_rate", 0))
    gates["paid_call_replacement_rate"] = {
        "passed": replacement >= thresholds.minimum_paid_call_replacement_rate,
        "actual": replacement,
        "minimum": thresholds.minimum_paid_call_replacement_rate,
    }
    baseline_utility = sum(
        float(baseline_metrics.get(metric, 0)) for metric in LANE_METRICS
    ) / len(LANE_METRICS)
    candidate_utility = sum(
        float(candidate_metrics.get(metric, 0)) for metric in LANE_METRICS
    ) / len(LANE_METRICS)
    utility_gain = candidate_utility - baseline_utility
    gates["utility_gain"] = {
        "passed": utility_gain >= thresholds.minimum_utility_gain,
        "actual": utility_gain,
        "minimum": thresholds.minimum_utility_gain,
    }

    first_hash = reproducibility.get("first_output_sha256")
    second_hash = reproducibility.get("second_output_sha256")
    gates["deterministic_reproduction"] = {
        "passed": (
            isinstance(first_hash, str)
            and len(first_hash) == 64
            and first_hash == second_hash
        ),
        "first_output_sha256": first_hash,
        "second_output_sha256": second_hash,
    }
    admitted_pairs = int(frontier.get("admitted_training_pairs") or 0)
    batch_cost = float(frontier.get("actual_batch_cost_usd") or 0)
    cost_per_pair = (
        batch_cost / admitted_pairs if admitted_pairs else float("inf")
    )
    gates["frontier_cost_per_admitted_pair"] = {
        "passed": (
            admitted_pairs > 0
            and cost_per_pair <= thresholds.maximum_cost_per_admitted_pair_usd
        ),
        "actual_usd": cost_per_pair,
        "maximum_usd": thresholds.maximum_cost_per_admitted_pair_usd,
        "admitted_pairs": admitted_pairs,
    }
    cr_records = int(admissions.get("cr_package_records") or 0)
    cr_validated = int(admissions.get("validated_cr_package_records") or 0)
    direct_write = bool(admissions.get("direct_database_write"))
    gates["cr_import_bundle"] = {
        "passed": cr_records > 0 and cr_records == cr_validated and not direct_write,
        "records": cr_records,
        "validated_records": cr_validated,
        "direct_database_write": direct_write,
    }

    retain = all(gate["passed"] for gate in gates.values())
    core = {
        "schema": QUALIFICATION_SCHEMA,
        "decision": "retain" if retain else "reject",
        "promotion_allowed": retain,
        "gates": gates,
        "utility": {
            "baseline": baseline_utility,
            "candidate": candidate_utility,
            "gain": utility_gain,
        },
        "policy": thresholds.model_dump(mode="json"),
    }
    core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
    return core


__all__ = [
    "FactoryThresholds",
    "QUALIFICATION_SCHEMA",
    "qualify",
]
