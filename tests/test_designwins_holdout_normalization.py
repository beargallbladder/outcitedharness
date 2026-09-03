from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "normalize_designwins_holdout.py"
SPEC = importlib.util.spec_from_file_location("normalize_designwins_holdout", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
normalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(normalizer)


def _record() -> dict:
    return {
        "instruction": (
            "Extract only this package. One entry per unique pin name. "
            "Return JSON."
        ),
        "input": "",
        "output": json.dumps(
            {
                "pins": [
                    {"pin_no": 1, "name": "VSS"},
                    {"pin_no": 17, "name": "VSS"},
                ]
            }
        ),
    }


def test_normalizer_preserves_labels_and_corrects_prompt_contract():
    source = _record()

    result = normalizer.normalize([source])

    assert result[0]["output"] == source["output"]
    assert normalizer.OLD_CONTRACT not in result[0]["instruction"]
    assert normalizer.NEW_CONTRACT in result[0]["instruction"]


def test_normalizer_rejects_duplicate_physical_pin():
    source = _record()
    value = json.loads(source["output"])
    value["pins"].append(dict(value["pins"][0]))
    source["output"] = json.dumps(value)

    with pytest.raises(ValueError, match="duplicates a physical pin"):
        normalizer.normalize([source])
