from pathlib import Path

from harness.cases.schema import Case, EvaluationSpec, ReferenceAnswer
from harness.evaluation import evaluate


def _case(**kwargs) -> Case:
    defaults = dict(
        id="t",
        title="t",
        path=Path("."),
        prompt="p",
        evaluation=EvaluationSpec(type="exact_text"),
    )
    defaults.update(kwargs)
    return Case(**defaults)


def test_keyword_rubric():
    case = _case(
        evaluation=EvaluationSpec(
            type="keyword_rubric",
            groups=[["process group", "session"], ["setsid", "daemon"]],
        )
    )
    full = evaluate(case, "The wrapper reaped the process group; use setsid.")
    assert full.verdict == "PASS"
    assert full.detail["groups_hit"] == 2
    partial = evaluate(case, "The parent process group was torn down.")
    assert partial.verdict == "PARTIAL"
    assert partial.detail["groups_hit"] == 1
    assert evaluate(case, "Use nohup harder.").verdict == "FAIL"


def test_exact_json_chatty_is_correct_not_format():
    case = _case(
        reference_answer=ReferenceAnswer(data={"adc_channels": 0}),
        evaluation=EvaluationSpec(type="exact_json"),
    )
    result = evaluate(case, 'Here you go:\n{"adc_channels": 0}\n')
    assert result.verdict == "PASS"
    assert result.correctness_ok is True
    assert result.format_ok is False


def test_exact_json_tolerance_and_null():
    case = _case(
        reference_answer=ReferenceAnswer(data=[2.4, 0.9, None]),
        evaluation=EvaluationSpec(type="exact_json", ignore_order=False, tolerance=0.01),
    )
    assert evaluate(case, "[2.405, 0.9, null]").verdict == "PASS"
    assert evaluate(case, "[2.4, 1.2, null]").verdict == "FAIL"


def test_json_fields():
    case = _case(
        evaluation=EvaluationSpec(type="json_fields", fields={"adc_channels": 0}),
    )
    assert evaluate(case, '{"adc_channels": 0, "reasoning": "no ADC module"}').verdict == "PASS"
    assert evaluate(case, '{"adc_channels": 8}').verdict == "FAIL"
