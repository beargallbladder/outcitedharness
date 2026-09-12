from __future__ import annotations

import sys
import json
import subprocess
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from power_gold_adjudication import (  # noqa: E402
    context_gate,
    label_readings,
    normalize,
    spec_grade,
    same_context,
)


class TestNormalizeRds:
    def test_milli_glyphs_stay_milli(self):
        for unit in ("mΩ", "m\u2126", "mohm", "mΩ\uf020"):
            assert normalize("rds_on_mohm", unit, 3.5) == 3.5

    def test_plain_ohm_scales_to_milli(self):
        for unit in ("Ω", "\u2126", "Ohm", "ohm"):
            assert normalize("rds_on_mohm", unit, 0.0035) == 3.5

    def test_unparseable_units_return_none(self):
        for unit in ("nA", "V/°C", "", None, "\x01", ":", "mW", "W", "Â", "mΩV", "m#", "m\x00", "kOhm", "MOhm"):
            assert normalize("rds_on_mohm", unit, 3.5) is None

    def test_ampere_metric_prefixes(self):
        assert normalize("id_a", "A", 40.0) == 40.0
        assert normalize("id_a", "mA", 40000.0) == 40.0
        assert normalize("id_a", "kA", 0.04) == 40.0

    def test_volt_and_charge_prefixes(self):
        assert normalize("vds_v", "V", 100.0) == 100.0
        assert normalize("vds_v", "mV", 100000.0) == 100.0
        assert normalize("qg_nc", "nC", 12.0) == 12.0
        assert normalize("qg_nc", "µC", 0.012) == 12.0
        assert normalize("qg_nc", "uC", 0.012) == 12.0

    def test_temperature_kelvin_offset(self):
        assert normalize("temp_max_c", "C", 150.0) == 150.0
        assert normalize("temp_max_c", "K", 423.15) == 150.0


class TestLabelReadings:
    def test_rds_respects_declared_unit(self):
        readings = dict(label_readings("rds_on_mohm", "Ohm", 3.5))
        assert readings["as_printed"] == 3500.0
        assert readings == {"as_printed": 3500.0}

    def test_other_fields_single_reading(self):
        readings = label_readings("id_a", "A", 40.0)
        assert readings == [("as_printed", 40.0)]

    def test_non_numeric_label_has_no_readings(self):
        assert label_readings("rds_on_mohm", "Ohm", None) == []


def _id_row(**over):
    row = {
        "symbol": "ID",
        "parameter": "Continuous drain current",
        "label": "ID Continuous drain current 40 A",
        "table_kind": "absolute_maximum",
    }
    row.update(over)
    return row


class TestSpecGradeId:
    def test_absolute_maximum_id_is_spec_grade(self):
        assert spec_grade("id_a", _id_row()) is True

    def test_prose_continuous_drain_is_spec_grade(self):
        assert spec_grade("id_a", _id_row(table_kind="prose", parameter="Drain current")) is True

    def test_pulsed_and_peak_rows_are_not(self):
        assert spec_grade("id_a", _id_row(parameter="Drain current, pulsed")) is False
        assert spec_grade("id_a", _id_row(symbol="IDM", parameter="Peak drain current")) is False

    def test_multicolumn_symbol_with_idm_is_not(self):
        assert spec_grade("id_a", _id_row(symbol="ID@TC=25°CID@TC=100°CIDM")) is False

    def test_silicon_and_chip_limited_are_not(self):
        assert spec_grade("id_a", _id_row(parameter="Continuous Drain Current, VGS @ 10V (Silicon Limited)")) is False
        assert spec_grade("id_a", _id_row(parameter="Continuous drain current, chip limited")) is False

    def test_output_characteristics_and_summary_are_not(self):
        assert spec_grade("id_a", _id_row(table_kind="characteristics", label="ID Typical output characteristic 25 A")) is False
        assert spec_grade("id_a", _id_row(table_kind="summary")) is False

    def test_other_fields_stay_spec_grade(self):
        assert spec_grade("rds_on_mohm", {"symbol": "RDS(on)", "table_kind": "summary"}) is True


def test_context_uses_lane_containment_semantics():
    """CR power-gold-fixtures-20260909: the label's condition is contained in
    the row's; a row may state MORE context than the label, never less."""
    miss = {"condition_verbatim": "VGS = 10 V", "quantity_qualifier": "maximum"}
    assert same_context(miss, dict(miss))
    # Row states more than the label: contained, still verifiable.
    assert same_context(miss, {**miss, "condition_verbatim": "VGS = 10 V, ID = 18 A"})
    # Ambient table context is emitted, not a conflict.
    assert same_context(miss, {**miss, "table_condition": "TA = 25 C"})
    assert context_gate(miss, {**miss, "table_condition": "TA = 25 C"}) == "pass"


def test_context_gate_reasons():
    miss = {"condition_verbatim": "VGS = 10 V", "quantity_qualifier": "maximum"}
    assert context_gate(miss, dict(miss)) == "pass"
    assert context_gate(miss, {**miss, "condition_verbatim": "VGS = 4.5 V"}) == "row_condition_mismatch"
    assert context_gate(miss, {**miss, "condition_verbatim": "VGS = -10 V"}) == "row_condition_mismatch"
    assert context_gate(miss, {**miss, "condition_verbatim": None}) == "row_condition_absent"
    assert context_gate(miss, {**miss, "quantity_qualifier": "typical"}) == "row_qualifier_mismatch"
    # A label stating more than the row is not contained: unverifiable.
    assert context_gate(miss, {**miss, "condition_verbatim": "VGS = 10"}) == "row_condition_mismatch"
    assert context_gate({"condition_verbatim": "VGS = 10 V"}, dict(miss)) == "label_qualifier_absent"
    assert context_gate({"quantity_qualifier": "maximum"}, dict(miss)) == "label_condition_absent"
    assert context_gate({}, dict(miss)) == "label_qualifier_absent"
    assert not same_context({}, {})


def test_id_exclusions_cover_conditions_and_labels():
    assert not spec_grade("id_a", _id_row(label="IDM 40 A"))
    assert not spec_grade("id_a", _id_row(condition_verbatim="chip limited"))


def test_cli_does_not_promote_numeral_or_context_coincidences(tmp_path):
    context = {"condition_verbatim": "VGS = 10 V", "quantity_qualifier": "maximum"}
    cases = [
        (0.0035, "mOhm", 3.5, {}, "unit_identity_match"),
        (0.0035, "mW", 3.5, {}, "unit_identity_match"),
        # Row states more context than the label: contained, still a verdict.
        (0.0035, "mOhm", 3.5, {"condition_verbatim": "VGS = 10 V, ID = 18 A"}, "unit_identity_match"),
        (3.5, "mOhm", 3.5, {}, "label_document_disagree"),
        (0.0035, "mOhm", 3.5, {"condition_verbatim": "VGS = 4.5 V"}, "ambiguous_comparison"),
        (0.0035, "mOhm", 3.5, {"condition_verbatim": None}, "ambiguous_comparison"),
        (0.0035, "mOhm", 3.5, {"quantity_qualifier": "typical"}, "ambiguous_comparison"),
        (0.0035, "mOhm", [3.5, 4.5], {}, "ambiguous_comparison"),
    ]
    grids, documents = [], []
    for i, (label, unit, value, changes, _) in enumerate(cases):
        grids.append({"_meta": {"source_artifact": str(i)}, "rows": [{
            "symbol": "RDS(on)", "label": "RDS(on) " + "x" * 150, "group": "switching",
            "table_kind": "characteristics", "unit": unit, "value": value,
            "verbatim": "RDS(on)(Max.) 3.5 mΩ VGS = 10 V", "table_title": "Table 5.1 Electrical Characteristics",
            **context, **changes,
        }]})
        documents.append({"source_artifact": str(i), "misses": [{
            "source_field": "rds_on_mohm", "value": label, "unit": "Ohm", **context,
        }]})
    grid_file, report, output = (tmp_path / n for n in ("grids.jsonl", "report.json", "out.jsonl"))
    grid_file.write_text("\n".join(json.dumps(g) for g in grids))
    report.write_text(json.dumps({"documents": documents}))
    subprocess.run([sys.executable, str(SCRIPT_ROOT / "power_gold_adjudication.py"),
                    "--grids", str(grid_file), "--gold-report", str(report), "--out", str(output)], check=True)
    results = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["kind"] for r in results] == [c[-1] for c in cases]
    # Full source context survives in the payload, uncapped.
    first = results[0]["document_rows"][0]
    assert first["label"] == "RDS(on) " + "x" * 150
    assert first["verbatim"] == "RDS(on)(Max.) 3.5 mΩ VGS = 10 V"
    assert first["table_title"] == "Table 5.1 Electrical Characteristics"
    assert first["context_gate"] == "pass"
    assert first["same_context"] is True
    assert results[0]["document_values_comparable"] == [3.5]


def test_cli_holds_verdicts_when_selector_label_lacks_qualifier(tmp_path):
    """The power-lane fixture reality: selector labels carry no qualifier, so
    no verdict is possible regardless of value agreement — held, with the row
    qualifier visible in the payload for CR's decomposition."""
    grids = [{"_meta": {"source_artifact": "a"}, "rows": [{
        "symbol": "RDS(on)", "label": "RDS(on) 3.5 mΩ", "group": "switching",
        "table_kind": "characteristics", "unit": "mOhm", "value": 3.5,
        "quantity_qualifier": "maximum", "condition_verbatim": "VGS = 10 V",
    }]}]
    documents = [{"source_artifact": "a", "misses": [{
        "source_field": "rds_on_mohm", "value": 0.0035, "unit": "Ohm",
        "condition_verbatim": "VGS = 10 V",
    }]}]
    grid_file, report, output = (tmp_path / n for n in ("grids.jsonl", "report.json", "out.jsonl"))
    grid_file.write_text(json.dumps(grids[0]))
    report.write_text(json.dumps({"documents": documents}))
    subprocess.run([sys.executable, str(SCRIPT_ROOT / "power_gold_adjudication.py"),
                    "--grids", str(grid_file), "--gold-report", str(report), "--out", str(output)], check=True)
    result = json.loads(output.read_text())
    assert result["kind"] == "ambiguous_comparison"
    assert result["label_quantity_qualifier"] is None
    assert result["document_values"] == [3.5]
    assert result["document_values_comparable"] == []
    assert result["document_rows"][0]["context_gate"] == "label_qualifier_absent"
    assert result["document_rows"][0]["quantity_qualifier"] == "maximum"


def test_cli_fixture_join_restores_stripped_qualifier(tmp_path):
    """The lenient report strips quantity_qualifier; --gold-dir joins it back
    from the rebuilt fixture, so the gate can rule and the source qualifier
    (what the selector export stated) is emitted for the gap measurement."""
    grids = [{"_meta": {"source_artifact": "a"}, "rows": [{
        "symbol": "RDS(on)", "label": "RDS(on)(Max.) 3.5 mΩ", "group": "switching",
        "table_kind": "characteristics", "unit": "mOhm", "value": 3.5,
        "quantity_qualifier": "rated", "condition_verbatim": "VGS = 10 V",
    }]}]
    documents = [{"source_artifact": "a", "misses": [{
        # What the lenient report emits: qualifier stripped by --ignore-key.
        "source_field": "rds_on_mohm", "value": 0.0035, "unit": "Ohm",
        "condition_verbatim": "VGS = 10 V",
    }]}]
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "a.json").write_text(json.dumps({
        "source_artifact": "a",
        "expected": [{
            "group": "switching", "source_field": "rds_on_mohm",
            "value": 0.0035, "unit": "Ohm",
            "quantity_qualifier": "rated", "source_qualifier": "typical",
        }],
    }))
    grid_file, report, output = (tmp_path / n for n in ("grids.jsonl", "report.json", "out.jsonl"))
    grid_file.write_text(json.dumps(grids[0]))
    report.write_text(json.dumps({"documents": documents}))
    subprocess.run([sys.executable, str(SCRIPT_ROOT / "power_gold_adjudication.py"),
                    "--grids", str(grid_file), "--gold-report", str(report),
                    "--gold-dir", str(gold_dir), "--out", str(output)], check=True)
    result = json.loads(output.read_text())
    assert result["kind"] == "unit_identity_match"
    assert result["label_reading_matched"] == "as_printed"
    assert result["label_quantity_qualifier"] == "rated"
    assert result["fixture_source_qualifier"] == "typical"
    assert result["document_values_comparable"] == [3.5]
    assert result["document_rows"][0]["context_gate"] == "pass"
