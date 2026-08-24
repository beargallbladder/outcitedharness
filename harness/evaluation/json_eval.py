from __future__ import annotations

import json
from typing import Any

from harness.cases.loader import load_reference_json
from harness.cases.schema import Case
from harness.evaluation.base import EvalResult
from harness.evaluation.jsonutil import extract_json, values_equal


def _parse_answer(case: Case, answer: str) -> Any:
    if case.evaluation.extract_json:
        return extract_json(answer)
    return json.loads(answer)


def _clean_json_only(text: str) -> bool:
    try:
        json.loads(text.strip())
        return True
    except json.JSONDecodeError:
        return False


def evaluate_exact_json(case: Case, answer: str) -> EvalResult:
    expected = load_reference_json(case)
    if expected is None:
        return EvalResult(
            verdict="ERROR",
            evaluator="exact_json",
            reason="No reference JSON configured",
        )
    try:
        got = _parse_answer(case, answer)
    except (ValueError, Exception) as exc:
        return EvalResult(
            verdict="FAIL",
            evaluator="exact_json",
            reason=f"Could not parse model JSON: {exc}",
            format_ok=False,
            correctness_ok=None,
            detail={"parse_error": str(exc)},
        )
    correct = values_equal(
        got,
        expected,
        ignore_list_order=case.evaluation.ignore_order,
        tolerance=case.evaluation.tolerance,
    )
    clean = _clean_json_only(answer)
    # Chatty-but-correct is still a solve. Format is a separate routing signal.
    return EvalResult(
        verdict="PASS" if correct else "FAIL",
        evaluator="exact_json",
        reason=(
            "JSON match"
            if correct and clean
            else "JSON match, chatty wrapper"
            if correct
            else "JSON mismatch"
        ),
        format_ok=clean,
        correctness_ok=correct,
        detail={"expected": expected, "got": got, "clean_json": clean},
    )


def evaluate_json_fields(case: Case, answer: str) -> EvalResult:
    result = evaluate_required_fields(case, answer)
    result.evaluator = "json_fields"
    return result


def evaluate_required_fields(case: Case, answer: str) -> EvalResult:
    try:
        got = _parse_answer(case, answer)
    except Exception as exc:
        return EvalResult(
            verdict="FAIL",
            evaluator="required_fields",
            reason=f"Could not parse model JSON: {exc}",
            format_ok=False,
            correctness_ok=None,
        )
    if not isinstance(got, dict):
        return EvalResult(
            verdict="FAIL",
            evaluator="required_fields",
            reason="Model output is not a JSON object",
            format_ok=False,
            correctness_ok=False,
        )

    missing = []
    mismatched = []
    for key, expected in case.evaluation.fields.items():
        if key not in got:
            missing.append(key)
            continue
        if expected is not None and got[key] != expected:
            mismatched.append({"key": key, "expected": expected, "got": got[key]})

    ok = not missing and not mismatched
    return EvalResult(
        verdict="PASS" if ok else "FAIL",
        evaluator="required_fields",
        reason="required fields present" if ok else "missing or mismatched fields",
        format_ok=_clean_json_only(answer),
        correctness_ok=ok,
        detail={"missing": missing, "mismatched": mismatched},
    )


def evaluate_numeric_fields(case: Case, answer: str) -> EvalResult:
    try:
        got = _parse_answer(case, answer)
    except Exception as exc:
        return EvalResult(
            verdict="FAIL",
            evaluator="numeric_fields",
            reason=f"Could not parse model JSON: {exc}",
        )
    if not isinstance(got, dict):
        return EvalResult(
            verdict="FAIL",
            evaluator="numeric_fields",
            reason="Model output is not a JSON object",
        )

    failures = []
    for key, spec in case.evaluation.fields.items():
        if key not in got:
            failures.append({"key": key, "reason": "missing"})
            continue
        try:
            actual = float(got[key])
        except (TypeError, ValueError):
            failures.append({"key": key, "reason": "not numeric", "got": got[key]})
            continue

        expected = spec.get("equals") if isinstance(spec, dict) else spec
        if expected is None and isinstance(spec, dict):
            expected = spec.get("value")
        if expected is None:
            failures.append({"key": key, "reason": "no expected value in spec"})
            continue
        expected_f = float(expected)
        abs_tol = float(spec.get("tolerance", 0)) if isinstance(spec, dict) else 0.0
        pct = float(spec.get("tolerance_pct", 0)) if isinstance(spec, dict) else 0.0
        allowed = abs_tol
        if pct:
            allowed = max(allowed, abs(expected_f) * (pct / 100.0))
        if abs(actual - expected_f) > allowed:
            failures.append(
                {
                    "key": key,
                    "expected": expected_f,
                    "got": actual,
                    "allowed": allowed,
                }
            )

    ok = not failures
    return EvalResult(
        verdict="PASS" if ok else "FAIL",
        evaluator="numeric_fields",
        reason="numeric fields within tolerance" if ok else "numeric mismatch",
        detail={"failures": failures},
    )
