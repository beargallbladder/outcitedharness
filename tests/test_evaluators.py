from pathlib import Path

from harness.cases.schema import Case, EvaluationSpec, ReferenceAnswer
from harness.evaluation import evaluate
from harness.evaluation.jsonutil import extract_json, values_equal


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


def test_exact_text_normalizes_whitespace():
    case = _case(
        reference_answer=ReferenceAnswer(text="hello world"),
        evaluation=EvaluationSpec(type="exact_text"),
    )
    assert evaluate(case, "  hello   world\n").verdict == "PASS"
    assert evaluate(case, "hello there").verdict == "FAIL"


def test_exact_json_ignores_key_order_and_fences():
    case = _case(
        reference_answer=ReferenceAnswer(data={"b": 2, "a": 1}),
        evaluation=EvaluationSpec(type="exact_json"),
    )
    result = evaluate(case, '```json\n{"a": 1, "b": 2}\n```')
    assert result.verdict == "PASS"


def test_required_fields():
    case = _case(
        evaluation=EvaluationSpec(
            type="required_fields",
            fields={"mcu": "STM32", "ok": None},
        )
    )
    assert evaluate(case, '{"mcu":"STM32","ok":true}').verdict == "PASS"
    assert evaluate(case, '{"mcu":"ESP32","ok":true}').verdict == "FAIL"


def test_numeric_fields_tolerance():
    case = _case(
        evaluation=EvaluationSpec(
            type="numeric_fields",
            fields={"v": {"equals": 3.3, "tolerance": 0.1}},
        )
    )
    assert evaluate(case, '{"v": 3.25}').verdict == "PASS"
    assert evaluate(case, '{"v": 3.6}').verdict == "FAIL"


def test_regex():
    case = _case(evaluation=EvaluationSpec(type="regex", pattern=r"USART1"))
    assert evaluate(case, "use USART1 at 115200").verdict == "PASS"
    assert evaluate(case, "UART2 only").verdict == "FAIL"


def test_human_pending():
    case = _case(evaluation=EvaluationSpec(type="human"))
    result = evaluate(case, "anything")
    assert result.verdict == "PENDING"


def test_error_short_circuits():
    case = _case(evaluation=EvaluationSpec(type="exact_text"))
    result = evaluate(case, "", error="timeout")
    assert result.verdict == "ERROR"


def test_json_extract_and_list_order():
    assert extract_json('prefix {"x": 1} suffix') == {"x": 1}
    assert values_equal([1, 2], [2, 1], ignore_list_order=True)
    assert not values_equal([1, 2], [2, 1], ignore_list_order=False)
