"""MOSFET table helpers: polarity, empty-symbol ID, abs-max ranges."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from harness.electronics.power_datasheet import (
    _header_looks_fragmented,
    _header_looks_like_data_row,
    _headers_align,
    _inline_conditions,
    _is_characteristics_header,
    _looks_like_symbol,
    _roles_for_header,
    _rows_from_fact,
    _split_multi_role_header,
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


def test_typ_max_units_header_cell_splits():
    cols = _split_multi_role_header("Typ. Max. Units", 339.6, 425.5)
    assert [c[2] for c in cols] == ["Typ.", "Max.", "Units"]
    roles, _ = _roles_for_header([n for _, _, n in cols], [])
    assert set(roles.values()) == {"typ", "max", "unit"}


def test_continuation_qg_row_is_not_a_header():
    assert _header_looks_like_data_row(["Q/g", "Total Gate Charge", "–––", "150", "210", "nC"])
    assert _header_looks_like_data_row(["Q\ngs", "V DD=50 V, I D=85 A", "-", "23", "30"])
    assert _header_looks_like_data_row(["C\niss", "V GS=0 V", "-", "4927", "6405"])
    assert _header_looks_like_data_row(["Qg Qgs Qgd", "Total Gate Charge Gate-to-Source Charge", "150 35 43", "210 ––– –––", "nC"])
    assert _header_looks_like_data_row(["Ciss Coss Crss", "VGS=0 V", "- - -", "4927 791 32", "6405 1029 48"])
    assert not _header_looks_like_data_row(["Symbol", "Conditions", "Values"])
    assert not _header_looks_like_data_row(["Parameter", "Min.", "Typ.", "Max.", "Units"])
    assert not _header_looks_like_data_row(["ID", "Continuous drain current", "11.4", "A"])
    assert _looks_like_symbol("Q/g")
    assert _looks_like_symbol("Ciss")
    assert not _looks_like_symbol("Gate charge total")
    assert _headers_align([(218.3, 250.0, "Symbol"), (400.0, 493.6, "max")], [(218.0, 249.0, "Qgs"), (401.0, 494.0, "30")])
    assert not _headers_align([(50.0, 80.0, "A"), (500.0, 560.0, "B")], [(218.0, 250.0, "Qgs"), (400.0, 493.0, "30")])


def test_qg_typ_recovers_switching_symbol():
    from harness.electronics.power_datasheet import _recover_symbol

    assert _recover_symbol("", "Qg,typ", "Qg,typ 41 nC") == "Qg"
    rows = _rows_from_fact(
        {
            "symbol": "",
            "symbol_as_printed": "",
            "parameter": "Qg,typ",
            "section": "",
            "table_kind": "summary",
            "table_title": "Key performance parameters",
            "table_condition": None,
            "condition_verbatim": None,
            "unit": "nC",
            "value": 41.0,
            "verbatim": "Qg,typ 41 nC",
            "page": 1,
        },
        {"vendor": "infineon.com"},
    )
    assert rows[0]["symbol"] == "Qg"
    assert rows[0]["group"] == "switching"
    assert rows[0]["value"] == 41.0


def test_inline_conditions_from_parameter_cell():
    captured = _inline_conditions(
        "Drain-source on-state ID = 1 A Tvj = 25 °C, resistance VGS(on) = 20 V"
    )
    assert captured == "ID = 1 A; Tvj = 25 °C; VGS(on) = 20 V"


def test_inline_conditions_skip_own_symbol():
    assert (
        _inline_conditions(
            "Drain-source voltage VDS = 600 V", excluded_symbol="VDS"
        )
        is None
    )
    kept = _inline_conditions(
        "on-state resistance at VDS = 600 V, ID = 10 A",
        excluded_symbol="RDS(on)",
    )
    assert kept == "VDS = 600 V; ID = 10 A"


def test_inline_conditions_clean_text_yields_none():
    assert _inline_conditions("Gate threshold voltage") is None
    assert _inline_conditions(None, "") is None
