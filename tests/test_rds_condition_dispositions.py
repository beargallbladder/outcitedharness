from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from rds_condition_dispositions import disposition_for, extract_vgs  # noqa: E402


def _row(**over):
    row = {
        "symbol": "RDS(on)", "parameter": "Drain-to-source on-resistance",
        "label": "RDS(on) 1.6 mΩ (VGS=10 V)", "group": "switching",
        "table_kind": "characteristics", "unit": "mΩ", "value": 1.6,
        "quantity_qualifier": "maximum", "condition_verbatim": "VGS=10 V",
    }
    row.update(over)
    return row


class TestExtractVgs:
    def test_from_all_text_fields(self):
        assert extract_vgs({"condition_verbatim": "VGS = 10 V"}) == 10.0
        assert extract_vgs({"verbatim": "RDS(on),max | 1.6 mW | VGS=4.5 V"}) == 4.5
        assert extract_vgs({"label": "RDS(on) 2.0 mΩ (VGS=6 V, ID=60 A)"}) == 6.0
        assert extract_vgs({"condition_verbatim": None, "verbatim": "no condition here"}) is None

    def test_first_field_wins(self):
        assert extract_vgs({"condition_verbatim": "VGS=10 V", "label": "VGS=4.5 V"}) == 10.0


class TestDispositions:
    def test_label_condition_absent(self):
        result = disposition_for({"source_field": "rds_on_mohm", "label_condition": None}, [_row()])
        assert result["disposition"] == "label_condition_absent"

    def test_doc_max_at_condition(self):
        miss = {"source_field": "rds_on_mohm", "label_condition": "VGS = 10 V", "label_value": 0.0026, "label_unit": "Ohm"}
        rows = [_row(value=1.6, quantity_qualifier="typical"),
                _row(value=2.6, quantity_qualifier="maximum"),
                _row(value=2.6, quantity_qualifier="maximum", condition_verbatim="VGS=4.5 V")]
        result = disposition_for(miss, rows)
        assert result["disposition"] == "doc_max_at_condition"
        assert result["doc_max_mohm"] == 2.6
        assert len(result["same_condition_rows"]) == 2

    def test_ambiguous_multiple_max(self):
        miss = {"source_field": "rds_on_mohm", "label_condition": "VGS = 10 V"}
        rows = [_row(value=2.6, quantity_qualifier="maximum"),
                _row(value=2.8, quantity_qualifier="maximum")]
        result = disposition_for(miss, rows)
        assert result["disposition"] == "ambiguous_multiple_max"
        assert result["doc_max_values_mohm"] == [2.6, 2.8]

    def test_typ_only_reports_equality_without_deciding(self):
        miss = {"source_field": "rds_on_mohm", "label_condition": "VGS = 10 V", "label_value": 0.0026, "label_unit": "Ohm"}
        rows = [_row(value=2.6, quantity_qualifier="typical")]
        result = disposition_for(miss, rows)
        assert result["disposition"] == "doc_typ_only_at_condition"
        assert result["typ_equals_label"] is True

    def test_no_row_at_label_condition_counts_rows_without_vgs(self):
        miss = {"source_field": "rds_on_mohm", "label_condition": "VGS = 10 V"}
        rows = [_row(condition_verbatim=None, label="RDS(on) 1.9 mW", verbatim="RDS(on),max | 3.2 | mW"),
                _row(condition_verbatim="VGS=6 V", value=2.0)]
        result = disposition_for(miss, rows)
        assert result["disposition"] == "no_row_at_label_condition"
        assert result["rows_without_vgs"] == 1

    def test_mw_glyph_row_normalizes_to_mohm(self):
        miss = {"source_field": "rds_on_mohm", "label_condition": "VGS = 10 V", "label_value": 0.0016, "label_unit": "Ohm"}
        rows = [_row(value=1.6, unit="mW", quantity_qualifier="maximum", condition_verbatim=None,
                     label="RDS(ON RDS(on),max 1.6 mW (VGS=10 V)")]
        result = disposition_for(miss, rows)
        assert result["disposition"] == "doc_max_at_condition"
        assert result["doc_max_mohm"] == 1.6
