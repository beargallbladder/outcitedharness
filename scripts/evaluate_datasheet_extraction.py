#!/usr/bin/env python3
"""Score a local extraction bundle on a sealed source-grounded cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from harness.electronics.claims import canonical_json
from harness.electronics.local_verification import (
    verify_parametrics,
    verify_pin_or_ball,
    verify_pin_semantics,
)


SCHEMA = "harness.electronics-extraction-evaluation.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _artifact(root: Path, manifest: dict[str, Any], name: str) -> Path:
    path = root / name
    receipt = manifest["artifacts"][name]
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != receipt["bytes"]
        or _sha256(path) != receipt["sha256"]
    ):
        raise ValueError(f"artifact differs from manifest: {path}")
    return path


def _normalized(value: Any) -> str:
    return "".join(
        character
        for character in str("" if value is None else value).upper()
        if character.isalnum()
    )


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return _normalized(row.get("pin_no")), _normalized(row.get("name"))


def _parametric_identity(
    fact: dict[str, Any],
) -> tuple[str, str, str, str]:
    return (
        _normalized(fact.get("field")),
        _normalized(fact.get("value")),
        _normalized(fact.get("value_role")),
        _normalized(fact.get("unit")),
    )


def _f1(true_positive: int, predicted: int, reference: int) -> float:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / reference if reference else 0.0
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def _write_new(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"immutable output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ).encode()
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--local-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cohort_root = args.cohort.expanduser().resolve(strict=True)
    cohort_manifest = json.loads(
        (cohort_root / "manifest.json").read_text(encoding="utf-8")
    )
    if cohort_manifest.get("schema") != (
        "harness.electronics-extraction-evaluation-cohort.v1"
    ):
        raise ValueError("unsupported evaluation cohort")
    queue_path = _artifact(
        cohort_root,
        cohort_manifest,
        "work-queue.json",
    )
    labels_path = _artifact(cohort_root, cohort_manifest, "labels.jsonl")
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    work = {item["work_id"]: item for item in queue["work"]}
    labels = {item["work_id"]: item for item in _jsonl(labels_path)}
    if set(work) != set(labels):
        raise ValueError("cohort work and labels differ")

    local_root = args.local_bundle.expanduser().resolve(strict=True)
    local_manifest = json.loads(
        (local_root / "manifest.json").read_text(encoding="utf-8")
    )
    if local_manifest.get("sources", {}).get(
        "structural_queue_sha256"
    ) != _sha256(queue_path):
        raise ValueError("local extraction did not use the frozen work queue")
    results = {
        item["work_id"]: item
        for item in _jsonl(
            _artifact(local_root, local_manifest, "local-results.jsonl")
        )
    }
    pages = {
        item["work_id"]: item["page"]
        for item in _jsonl(
            _artifact(local_root, local_manifest, "pillar-evidence.jsonl")
        )
    }

    totals: Counter[str] = Counter()
    semantic: Counter[str] = Counter()
    # Which pipeline stage actually answered each example. A healthy vision
    # evaluation is dominated by focused_local_vision; a fallback-dominated
    # run means the served model was not really measured (the deterministic
    # normalizer or text path answered instead) and the scores say nothing
    # about the checkpoint under test.
    answer_stages: Counter[str] = Counter()
    details = []
    vendor_values: dict[str, Counter[str]] = defaultdict(Counter)
    for work_id, expected_record in labels.items():
        item = work[work_id]
        result = results.get(work_id)
        capability = item["capability"]
        vendor = item.get("vendor") or "unknown"
        totals["examples"] += 1
        totals["json_valid"] += result is not None
        answer_stages[
            (result or {}).get("local_pillar_stage") or "no_result"
        ] += 1
        if capability == "parametrics":
            expected_facts = expected_record["expected_facts"]
            predicted_facts = (
                result.get("result", {}).get("facts", [])
                if result is not None
                else []
            )
            expected_identities = {
                _parametric_identity(fact)
                for fact in expected_facts
                if isinstance(fact, dict)
                and all(_parametric_identity(fact)[:3])
            }
            predicted_identities = {
                _parametric_identity(fact)
                for fact in predicted_facts
                if isinstance(fact, dict)
                and all(_parametric_identity(fact)[:3])
            }
            matched = expected_identities & predicted_identities
            exact = predicted_identities == expected_identities
            totals["parametric_examples"] += 1
            totals["parametric_true_positive"] += len(matched)
            totals["parametric_predicted"] += len(predicted_identities)
            totals["parametric_reference"] += len(expected_identities)
            totals["parametric_exact"] += exact
            vendor_values[vendor]["parametric_true_positive"] += len(matched)
            vendor_values[vendor]["parametric_predicted"] += len(
                predicted_identities
            )
            vendor_values[vendor]["parametric_reference"] += len(
                expected_identities
            )
            vendor_values[vendor]["parametric_examples"] += 1

            grounded = False
            reason = "missing local result"
            if result is not None:
                verdict = verify_parametrics(result, pages[work_id])
                grounded = verdict.passed
                reason = verdict.reason
            totals["grounded_examples"] += grounded
            replaceable = grounded and exact
            totals["replaceable_examples"] += replaceable
            details.append(
                {
                    "work_id": work_id,
                    "vendor": vendor,
                    "capability": capability,
                    "expected": len(expected_identities),
                    "predicted": len(predicted_identities),
                    "matched": len(matched),
                    "fact_exact": exact,
                    "source_grounded": grounded,
                    "paid_call_replaceable": replaceable,
                    "grounding_reason": reason,
                }
            )
            continue

        expected_rows = expected_record["expected_pins"]
        predicted_rows = (
            result.get("result", {}).get("pins", [])
            if result is not None
            else []
        )
        expected_by_identity = {
            _identity(row): row for row in expected_rows if all(_identity(row))
        }
        predicted_by_identity = {
            _identity(row): row
            for row in predicted_rows
            if isinstance(row, dict) and all(_identity(row))
        }
        expected_identities = set(expected_by_identity)
        predicted_identities = set(predicted_by_identity)
        matched = expected_identities & predicted_identities
        totals["pin_examples"] += 1
        totals["identity_true_positive"] += len(matched)
        totals["identity_predicted"] += len(predicted_identities)
        totals["identity_reference"] += len(expected_identities)
        totals["identity_exact"] += (
            predicted_identities == expected_identities
        )
        vendor_values[vendor]["pin_true_positive"] += len(matched)
        vendor_values[vendor]["pin_predicted"] += len(predicted_identities)
        vendor_values[vendor]["pin_reference"] += len(expected_identities)
        vendor_values[vendor]["pin_examples"] += 1

        for identity in matched:
            expected = expected_by_identity[identity]
            predicted = predicted_by_identity[identity]
            for field in ("type", "dir", "supply_domain"):
                if expected.get(field) is None:
                    continue
                semantic[f"{field}_reference"] += 1
                semantic[f"{field}_correct"] += (
                    _normalized(predicted.get(field))
                    == _normalized(expected[field])
                )
            expected_functions = {
                _normalized(value)
                for value in expected.get("functions") or []
                if _normalized(value)
            }
            if expected_functions:
                predicted_functions = {
                    _normalized(value)
                    for value in predicted.get("functions") or []
                    if _normalized(value)
                }
                semantic["function_true_positive"] += len(
                    expected_functions & predicted_functions
                )
                semantic["function_predicted"] += len(predicted_functions)
                semantic["function_reference"] += len(expected_functions)

        grounded = False
        reason = "missing local result"
        if result is not None:
            verifier = (
                verify_pin_or_ball
                if item["capability"] == "pin_or_ball"
                else verify_pin_semantics
            )
            verdict = verifier(result, pages[work_id])
            grounded = verdict.passed
            reason = verdict.reason
        totals["grounded_examples"] += grounded
        replaceable = grounded and predicted_identities == expected_identities
        totals["replaceable_examples"] += replaceable
        details.append(
            {
                "work_id": work_id,
                "vendor": vendor,
                "capability": item["capability"],
                "expected": len(expected_identities),
                "predicted": len(predicted_identities),
                "matched": len(matched),
                "identity_exact": predicted_identities
                == expected_identities,
                "source_grounded": grounded,
                "paid_call_replaceable": replaceable,
                "grounding_reason": reason,
            }
        )

    identity_f1 = _f1(
        totals["identity_true_positive"],
        totals["identity_predicted"],
        totals["identity_reference"],
    )
    parametric_f1 = (
        _f1(
            totals["parametric_true_positive"],
            totals["parametric_predicted"],
            totals["parametric_reference"],
        )
        if totals["parametric_examples"]
        else None
    )
    type_accuracy = (
        semantic["type_correct"] / semantic["type_reference"]
        if semantic["type_reference"]
        else None
    )
    direction_accuracy = (
        semantic["dir_correct"] / semantic["dir_reference"]
        if semantic["dir_reference"]
        else None
    )
    supply_domain_accuracy = (
        semantic["supply_domain_correct"]
        / semantic["supply_domain_reference"]
        if semantic["supply_domain_reference"]
        else None
    )
    function_f1 = (
        _f1(
            semantic["function_true_positive"],
            semantic["function_predicted"],
            semantic["function_reference"],
        )
        if semantic["function_reference"]
        else None
    )
    semantic_values = [
        value
        for value in (
            type_accuracy,
            direction_accuracy,
            supply_domain_accuracy,
            function_f1,
        )
        if value is not None
    ]
    semantic_macro = (
        sum(semantic_values) / len(semantic_values)
        if semantic_values
        else 0.0
    )
    hallucinations = (
        totals["identity_predicted"] - totals["identity_true_positive"]
        + totals["parametric_predicted"]
        - totals["parametric_true_positive"]
    )
    all_predictions = (
        totals["identity_predicted"] + totals["parametric_predicted"]
    )
    aggregate = {
        "examples": totals["examples"],
        "json_valid_rate": totals["json_valid"] / totals["examples"],
        "identity_exact_rate": (
            totals["identity_exact"] / totals["pin_examples"]
            if totals["pin_examples"]
            else None
        ),
        "pin_identity_f1": identity_f1,
        "pin_semantic_macro_f1": semantic_macro,
        "parametric_accuracy": parametric_f1,
        "parametric_accuracy_definition": (
            "micro_f1(field,value,value_role,unit)"
        ),
        "parametric_exact_rate": (
            totals["parametric_exact"] / totals["parametric_examples"]
            if totals["parametric_examples"]
            else None
        ),
        "type_accuracy": type_accuracy,
        "direction_accuracy": direction_accuracy,
        "supply_domain_accuracy": supply_domain_accuracy,
        "function_f1": function_f1,
        "evidence_grounding_rate": (
            totals["grounded_examples"] / totals["examples"]
        ),
        "hallucination_rate": (
            hallucinations / all_predictions
            if all_predictions
            else 0.0
        ),
        "paid_call_replacement_rate": (
            totals["replaceable_examples"] / totals["examples"]
        ),
        "identity_counts": {
            "true_positive": totals["identity_true_positive"],
            "predicted": totals["identity_predicted"],
            "reference": totals["identity_reference"],
        },
        "parametric_counts": {
            "true_positive": totals["parametric_true_positive"],
            "predicted": totals["parametric_predicted"],
            "reference": totals["parametric_reference"],
        },
        "semantic_counts": dict(semantic),
        "answer_stages": dict(sorted(answer_stages.items())),
        "model_under_test_answer_rate": (
            answer_stages.get("focused_local_vision", 0) / totals["examples"]
        ),
        "unevaluated_lanes": ["series_summary", "opn_decoder"],
    }
    by_vendor = {
        vendor: {
            "pin_examples": values["pin_examples"],
            "pin_identity_f1": _f1(
                values["pin_true_positive"],
                values["pin_predicted"],
                values["pin_reference"],
            ),
            "parametric_examples": values["parametric_examples"],
            "parametric_accuracy": (
                _f1(
                    values["parametric_true_positive"],
                    values["parametric_predicted"],
                    values["parametric_reference"],
                )
                if values["parametric_examples"]
                else None
            ),
        }
        for vendor, values in sorted(vendor_values.items())
    }
    core = {
        "schema": SCHEMA,
        "model": local_manifest["model"],
        "cohort": {
            "path": str(cohort_root),
            "evidence_sha256": cohort_manifest["evidence_sha256"],
            "work_queue_sha256": _sha256(queue_path),
            "labels_sha256": _sha256(labels_path),
        },
        "aggregate": aggregate,
        "by_vendor": by_vendor,
        "details": details,
    }
    core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
    value = {
        "created_at": local_manifest["created_at"],
        **core,
    }
    _write_new(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
