from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from power_gold_adjudication import (  # noqa: E402
    label_readings,
    normalize,
    spec_grade,
)


class TestNormalizeRds:
    def test_milli_glyphs_stay_milli(self):
        for unit in ("mΩ", "mΩ", "mohm", "mΩV", "mΩ\uf020", "mW", "m#", "m\x00"):
            assert normalize("rds_on_mohm", unit, 3.5) == 3.5

    def test_plain_ohm_scales_to_milli(self):
        for unit in ("Ω", "Ω", "Ohm", "ohm", "W", "Â"):
            assert normalize("rds_on_mohm", unit, 0.0035) == 3.5

    def test_unparseable_units_return_none(self):
        for unit in ("nA", "V/°C", "", None, "\x01", ":"):
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
    def test_rds_gets_dual_readings(self):
        readings = dict(label_readings("rds_on_mohm", "Ohm", 3.5))
        assert readings["as_printed"] == 3500.0
        assert readings["as_mohm_numeral"] == 3.5

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
