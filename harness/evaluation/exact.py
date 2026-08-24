from __future__ import annotations

import re

from harness.cases.loader import load_reference_text
from harness.cases.schema import Case
from harness.evaluation.base import EvalResult


def _normalize(text: str, case: Case) -> str:
    value = text.strip()
    if case.evaluation.case_insensitive:
        value = value.lower()
    if case.evaluation.normalize_whitespace:
        value = re.sub(r"\s+", " ", value).strip()
    return value


def evaluate_exact_text(case: Case, answer: str) -> EvalResult:
    expected = load_reference_text(case)
    if expected is None:
        return EvalResult(
            verdict="ERROR",
            evaluator="exact_text",
            reason="No reference answer configured",
        )
    got = _normalize(answer, case)
    want = _normalize(expected, case)
    ok = got == want
    return EvalResult(
        verdict="PASS" if ok else "FAIL",
        evaluator="exact_text",
        reason="exact match" if ok else "text mismatch",
        detail={"expected": want, "got": got},
    )


def evaluate_keyword_rubric(case: Case, answer: str) -> EvalResult:
    groups = case.evaluation.groups
    if not groups and isinstance(case.evaluation.fields.get("must_contain_any_of_each"), list):
        groups = case.evaluation.fields["must_contain_any_of_each"]
    if not groups:
        return EvalResult(
            verdict="ERROR",
            evaluator="keyword_rubric",
            reason="evaluation.groups is required",
        )
    haystack = answer.lower()
    hit = []
    missed = []
    for idx, group in enumerate(groups):
        matched = next(
            (str(option) for option in group if str(option).lower() in haystack),
            None,
        )
        if matched is None:
            missed.append({"index": idx, "group": group})
        else:
            hit.append({"index": idx, "matched": matched})
    if not missed:
        verdict = "PASS"
        reason = f"all {len(groups)} rubric groups hit"
    elif hit:
        verdict = "PARTIAL"
        reason = f"{len(hit)}/{len(groups)} rubric groups hit"
    else:
        verdict = "FAIL"
        reason = "no rubric groups hit"
    return EvalResult(
        verdict=verdict,
        evaluator="keyword_rubric",
        reason=reason,
        format_ok=True,
        correctness_ok=not missed,
        detail={
            "groups_total": len(groups),
            "groups_hit": len(hit),
            "hit": hit,
            "missed_groups": missed,
        },
    )


def evaluate_regex(case: Case, answer: str) -> EvalResult:
    pattern = case.evaluation.pattern
    if not pattern:
        return EvalResult(
            verdict="ERROR",
            evaluator="regex",
            reason="evaluation.pattern is required",
        )
    flags = 0
    if case.evaluation.case_insensitive or (case.evaluation.flags or "").find("i") >= 0:
        flags |= re.IGNORECASE
    if (case.evaluation.flags or "").find("m") >= 0:
        flags |= re.MULTILINE
    if (case.evaluation.flags or "").find("s") >= 0:
        flags |= re.DOTALL
    matched = re.search(pattern, answer, flags) is not None
    return EvalResult(
        verdict="PASS" if matched else "FAIL",
        evaluator="regex",
        reason="pattern matched" if matched else "pattern not found",
        detail={"pattern": pattern},
    )
