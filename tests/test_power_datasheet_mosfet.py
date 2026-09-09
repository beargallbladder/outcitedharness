"""MOSFET table helpers: polarity, empty-symbol ID, abs-max ranges."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from harness.electronics.power_datasheet import (
    _header_looks_fragmented,
    _is_characteristics_header,
    _rows_from_fact,
    _unit_from_symbol,
    canonical_symbols,
    classify,
)

_SCORE = importlib.util.spec_from_file_location(
    "score_key_features_gold",
    Path(__file__).resolve().parents[1] / "scripts" / "score_key_features_gold.py",
)
_score = importlib.util.module_from_spec(_SCORE)
assert _SCORE.loader
_SCORE.loader.exec_module(_score)
values_equal = _score.values_equal


def test_pchannel_current_matches_unsigned_gold():
    assert values_equal({"value": 1.6, "unit": "A"}, {"value": -1.6, "unit": "A"})
    assert values_equal({"value": -20.0, "unit": "V"}, {"value": 20.0, "unit": "V"})
    assert not values_equal({"value": -40.0, "unit": "C"}, {"value": 40.0, "unit": "C"})


def test_operating_current_is_id():
    assert "ID" in canonical_symbols("Operating current")
    assert "ID" in canonical_symbols("Continuous Drain-to-Drain Current")
    assert classify("", "Operating current", "features") == "switching"


def test_empty_symbol_filled_from_parameter():
    rows = _rows_from_fact(
        {
            "symbol": "",
            "symbol_as_printed": "",
            "parameter": "Operating current",
            "section": "recommended",
            "table_kind": "recommended",
            "table_title": "Recommended Operating Conditions",
            "table_condition": None,
            "condition_verbatim": None,
            "unit": "A",
            "max": 20.0,
            "verbatim": "Operating current 20 A",
            "page": 3,
        },
        {"vendor": "ti.com"},
    )
    assert rows[0]["symbol"] == "ID"
    assert rows[0]["group"] == "switching"
    assert rows[0]["value"] == 20.0


def test_abs_max_min_and_max_emit_range():
    rows = _rows_from_fact(
        {
            "symbol": "TJ",
            "symbol_as_printed": "TJ",
            "parameter": "Operating junction, TJ",
            "section": "absolute_maximum",
            "table_kind": "absolute_maximum",
            "table_title": "Absolute Maximum Ratings",
            "table_condition": None,
            "condition_verbatim": None,
            "unit": "°C",
            "min": -55.0,
            "max": 150.0,
            "verbatim": "Operating junction, TJ -55 150",
            "page": 3,
        },
        {"vendor": "ti.com"},
    )
    assert len(rows) == 1
    assert rows[0]["value"] == [-55.0, 150.0]


def test_symbol_value_is_a_characteristics_header():
    assert _is_characteristics_header({0: "symbol", 1: "value"})
    assert _unit_from_symbol("V/DSS") == "V"
    assert _unit_from_symbol("ID") == "A"
    assert _unit_from_symbol("I *1/D") == "A"


def test_q1_q2_header_is_fragmented():
    smashed = ["", "PA", "RAMETER", "TEST CONDITIONS", "Q1", "Control FE", "T", "Q", "2 Sync F", "ET\nMAX", "UNIT"]
    assert _header_looks_fragmented(smashed)
    merged = ["", "PARAMETER", "TEST CONDITIONS", "Q1 Control FET MIN TYP MAX", "Q2 Sync FET MIN TYP MAX", "UNIT"]
    assert not _header_looks_fragmented(merged)
    assert not _header_looks_fragmented(["SYMBOL", "PARAMETER", "MIN", "TYP", "MAX", "UNIT"])


def test_vdss_parameter_recovers_switching_symbol():
    from harness.electronics.power_datasheet import _recover_symbol

    assert _recover_symbol("", "VDSS over full Tj,range", "VDSS over full Tj,range 750 V") == "VDS"
    rows = _rows_from_fact(
        {
            "symbol": "",
            "symbol_as_printed": "",
            "parameter": "VDSS over full Tj,range",
            "section": "",
            "table_kind": "characteristics",
            "table_title": "Key performance parameters",
            "table_condition": None,
            "condition_verbatim": None,
            "unit": "V",
            "value": 750.0,
            "verbatim": "VDSS over full Tj,range 750 V",
            "page": 1,
        },
        {"vendor": "infineon.com"},
    )
    assert rows[0]["symbol"] == "VDS"
    assert rows[0]["group"] == "switching"
    assert rows[0]["value"] == 750.0


def test_continuous_dc_drain_current_is_id():
    assert "ID" in canonical_symbols("Continuous DC drain current")
