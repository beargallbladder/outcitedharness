from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from build_coding_frozen_fixture import build  # noqa: E402
from compare_coding_qualification import compare  # noqa: E402
from evaluate_coding_repair_fixture import (  # noqa: E402
    _extract_patch,
    _run_tests,
    load_fixture,
)


def _evaluation(model: str, passes: tuple[bool, ...], latency: float) -> dict:
    cases = [
        {
            "case_id": f"case-{index}",
            "passed": passed,
        }
        for index, passed in enumerate(passes)
    ]
    return {
        "fixture_sha256": "a" * 64,
        "model": model,
        "sample_count": len(cases),
        "passed": sum(passes),
        "verified_success_rate": sum(passes) / len(passes),
        "p95_latency_ms": latency,
        "evidence_sha256": ("b" if model == "base" else "c") * 64,
        "cases": cases,
    }


def test_frozen_fixture_digest_and_parents(tmp_path: Path) -> None:
    path = tmp_path / "fixture.json"
    value = build(path)
    assert load_fixture(path) == value
    assert len(value["cases"]) == 6
    assert all(not _run_tests(case, None)["passed"] for case in value["cases"])


def test_patch_parser_allows_only_solution_file() -> None:
    patch = (
        "diff --git a/solution.py b/solution.py\n"
        "--- a/solution.py\n"
        "+++ b/solution.py\n"
        "@@ -1 +1 @@\n"
        "-value = 1\n"
        "+value = 2\n"
    )
    assert _extract_patch(f"```diff\n{patch}```") == patch
    headerless = patch.split("\n", 1)[1]
    assert _extract_patch(f"```diff\n{headerless}```") == patch
    with pytest.raises(ValueError, match="only solution.py"):
        _extract_patch(patch.replace("solution.py", "hidden_test_solution.py"))


def test_qualification_requires_gain_without_regressions() -> None:
    baseline = _evaluation(
        "base",
        (True, True, False, False, False, False),
        1000,
    )
    improved = _evaluation(
        "adapter",
        (True, True, True, False, False, False),
        1050,
    )
    passed = compare(
        baseline=baseline,
        candidate=improved,
        minimum_gain=0.02,
        maximum_latency_regression=0.10,
    )
    assert passed["passed"]
    assert passed["action"] == "shadow"

    regressed = _evaluation(
        "adapter",
        (False, True, True, True, False, False),
        1050,
    )
    rejected = compare(
        baseline=baseline,
        candidate=regressed,
        minimum_gain=0.02,
        maximum_latency_regression=0.10,
    )
    assert not rejected["passed"]
    assert not rejected["checks"]["no_case_regressions"]
