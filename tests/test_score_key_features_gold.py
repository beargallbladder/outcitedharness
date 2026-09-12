from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from score_key_features_gold import match_row, to_base, values_equal  # noqa: E402


def test_to_base_si_normalisation():
    assert to_base(6.8, "mΩ") == (0.0068, "ohm")
    assert to_base(3.5, "Ohm") == (3.5, "ohm")
    assert to_base(0.012, "µC") == (1.2e-8, "coulomb")
    assert to_base(40, "A") == (40.0, "a")
    assert to_base(3.5, "mW") == (0.0035, "w")  # watts everywhere else
    assert to_base(3.5, "V/°C") is None


def test_values_equal_si_and_range():
    assert values_equal({"value": 0.0035, "unit": "Ohm", "source_field": "rds_on_mohm"}, {"value": 3.5, "unit": "mΩ"})
    assert values_equal({"value": 40.0, "unit": "A", "source_field": "id_a"}, {"value": [20.0, 40.0], "unit": "A"})
    # P-channel: document negative, selector unsigned.
    assert values_equal({"value": -30.0, "unit": "V", "source_field": "vds_v"}, {"value": 30.0, "unit": "V"})


def test_mw_glyph_correction_is_bounded_to_rds_document_rows():
    """CR adjudicator-fixes-20260911: the Omega->W glyph confusion means mW
    inside an RDS(on) document row is mOhm. Only there — never another field,
    never the fixture's declared unit, and plain W stays watts-failure (the
    adjudicator, not the scorer, owns the wider bounded reading)."""
    rds_exp = {"value": 0.0016, "unit": "Ohm", "source_field": "rds_on_mohm"}
    assert values_equal(rds_exp, {"value": 1.6, "unit": "mW"})
    # A factor-of-1000 mismatch must still fail: 1.6 Ohm label vs 1.6 mW row.
    assert not values_equal({"value": 1.6, "unit": "Ohm", "source_field": "rds_on_mohm"}, {"value": 1.6, "unit": "mW"})
    # Other fields: mW is not an ohm unit there, so no base match.
    assert not values_equal({"value": 0.0016, "unit": "Ohm", "source_field": "id_a"}, {"value": 1.6, "unit": "mW"})
    # Pre-existing lenient semantics, locked as-is: when the two units parse
    # to DIFFERENT bases the scorer falls back to raw numeral equality. The
    # strict instrument for that case is the adjudicator, not the scorer.


def test_match_row_accepts_mw_rds_row():
    exp = {
        "group": "switching", "label": r"R\s*DS\s*\(?on\)?|on[- ]resistance",
        "value": 0.0016, "unit": "Ohm", "source_field": "rds_on_mohm",
    }
    row = {
        "group": "switching", "tier": "grid",
        "label": "RDS(ON RDS(on),max 1.6 mW (VGS=10 V)",
        "value": 1.6, "unit": "mW",
    }
    ok, why = match_row(exp, row)
    assert ok, why
