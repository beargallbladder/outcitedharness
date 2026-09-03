import importlib.util
from pathlib import Path


def _script(name: str):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


freezer = _script("freeze_datasheet_extraction_evaluation")
evaluator = _script("evaluate_datasheet_extraction")


def test_frozen_label_does_not_promote_description_to_type() -> None:
    page = {
        "document_sha256": "a" * 64,
        "page_1based": 1,
        "blocks": [],
        "tables": [
            {
                "table_index": 0,
                "rows": [
                    ["Pin Name", "Pin(s)", "Description"],
                    ["PC00", "1", "GPIO"],
                ],
            }
        ],
    }

    resolved = freezer._grounded_label(
        "pin_semantics",
        {
            "pin_no": 1,
            "name": "PC00",
            "type": "gpio",
            "dir": None,
            "functions": ["GPIO"],
        },
        freezer._source_rows("pin_semantics", page),
    )

    assert resolved is not None
    label, _quote = resolved
    assert label == {
        "pin_no": 1,
        "name": "PC00",
        "type": None,
        "dir": None,
        "supply_domain": None,
        "functions": ["GPIO"],
    }


def test_frozen_label_maps_selected_package_semantics_by_column() -> None:
    page = {
        "document_sha256": "a" * 64,
        "page_1based": 1,
        "blocks": [],
        "tables": [
            {
                "table_index": 0,
                "bbox": [0, 0, 100, 100],
                "rows": [
                    [
                        "Pin/ball name",
                        None,
                        None,
                        "Pin name",
                        "Pin type",
                        "I/O structure",
                        "Alternate functions",
                    ],
                    [
                        "LQFP144",
                        "LQFP176",
                        "TFBGA240+25",
                        None,
                        None,
                        None,
                        None,
                    ],
                    ["1", "3", "D1", "PE2", "I/O", "FT_h", "TRACECLK"],
                ],
            }
        ],
        "structural_evidence": {
            "regions": [{"table_index": 0, "bbox": [0, 0, 100, 100]}],
            "package_scope": {
                "package": "LQFP176",
                "column_header": "LQFP176",
            },
        },
    }

    resolved = freezer._grounded_label(
        "pin_semantics",
        {
            "pin_no": 3,
            "name": "PE2",
            "type": None,
            "dir": "I/O",
            "functions": ["TRACECLK"],
        },
        freezer._source_rows("pin_semantics", page),
    )

    assert resolved is not None
    label, _quote = resolved
    assert label["pin_no"] == 3
    assert label["type"] == "FT_h"
    assert label["dir"] == "I/O"
    assert label["supply_domain"] is None
    assert label["functions"] == ["TRACECLK"]


def test_frozen_and_scored_identity_preserve_numeric_zero() -> None:
    page = {
        "document_sha256": "a" * 64,
        "page_1based": 1,
        "blocks": [],
        "tables": [
            {
                "table_index": 0,
                "rows": [
                    ["Pin Name", "Pin(s)", "Description"],
                    ["VSS", 0, "Ground"],
                ],
            }
        ],
    }

    resolved = freezer._grounded_label(
        "pin_semantics",
        {
            "pin_no": 0,
            "name": "VSS",
            "type": None,
            "dir": None,
            "functions": ["Ground"],
        },
        freezer._source_rows("pin_semantics", page),
    )

    assert resolved is not None
    label, quote = resolved
    assert label["pin_no"] == 0
    assert quote == "VSS | 0 | Ground"
    assert evaluator._identity(label) == ("0", "VSS")
