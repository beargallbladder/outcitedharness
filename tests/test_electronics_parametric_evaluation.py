import importlib.util
from pathlib import Path


def _script(name: str):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_parametric_identity = _script(
    "evaluate_datasheet_extraction"
)._parametric_identity
_parametric_labels = _script(
    "freeze_datasheet_extraction_evaluation"
)._parametric_labels


def test_parametric_labels_are_frozen_from_selected_source_rows() -> None:
    item = {
        "document_sha256": "a" * 64,
        "page_1based": 7,
    }
    page = {
        "document_sha256": "a" * 64,
        "page_1based": 7,
        "blocks": [],
        "tables": [
            {
                "table_index": 3,
                "bbox": [10, 20, 400, 500],
                "rows": [
                    [
                        "Parameter",
                        "Condition",
                        "Min",
                        "Typ",
                        "Max",
                        "Unit",
                    ],
                    [
                        "Supply voltage",
                        "TA = 25 C",
                        "1.7",
                        "1.8",
                        "3.6",
                        "V",
                    ],
                ],
            }
        ],
        "structural_evidence": {
            "regions": [
                {"table_index": 3, "bbox": [10, 20, 400, 500]}
            ],
            "package_scope": None,
        },
    }

    facts, quotes = _parametric_labels(item, page)

    assert {
        _parametric_identity(fact)
        for fact in facts
    } == {
        ("SUPPLYVOLTAGE", "17", "MIN", "V"),
        ("SUPPLYVOLTAGE", "18", "TYP", "V"),
        ("SUPPLYVOLTAGE", "36", "MAX", "V"),
    }
    assert len(quotes) == 3
    assert all("Supply voltage" in quote for quote in quotes)
