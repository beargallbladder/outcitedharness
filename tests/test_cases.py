from pathlib import Path

from harness.cases.loader import discover_cases, load_case
from harness.cases.prompt import build_prompt
from harness.config import Capabilities, ModelConfig, Settings


def test_load_example_case():
    case = load_case(Path("cases/example_001"))
    assert case.id == "example_001"
    assert case.evaluation.type == "exact_json"
    assert "USART1" in case.prompt


def test_discover_cases_folder():
    cases = discover_cases(Path("cases"))
    assert any(c.id == "example_001" for c in cases)


def test_prompt_is_identical_for_text_models():
    case = load_case(Path("cases/example_001"))
    settings = Settings(results_dir=Path("results"), db_path=Path("results/harness.db"))
    a = ModelConfig(
        key="a",
        tier=0,
        display_name="A",
        short_name="A",
        provider="openai_compatible",
        base_url="http://localhost/v1",
        model="a",
        capabilities=Capabilities(vision=False),
    )
    b = ModelConfig(
        key="b",
        tier=1,
        display_name="B",
        short_name="B",
        provider="openai_compatible",
        base_url="http://localhost/v1",
        model="b",
        capabilities=Capabilities(vision=False),
    )
    pa = build_prompt(case, settings, a)
    pb = build_prompt(case, settings, b)
    assert pa.user == pb.user
    assert "115200" in pa.user
