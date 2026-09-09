"""Front-page prose forms the 100-doc TI sample still missed.

These strings are copied from page-1 Features / title lines. The table
reader never sees them; the bullet reader has to.
"""
from __future__ import annotations

from harness.electronics.power_datasheet import _facts_from_front_text, _flatten_front_text


def _facts(*chunks: str) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for chunk in chunks:
        out.extend(_facts_from_front_text(_flatten_front_text(chunk), 1, seen))
    return out


def _of(facts: list[dict], kind: str) -> list:
    return [f["value"] for f in facts if f["prose_kind"] == kind]


def test_voltage_colon_without_space_is_a_range():
    """'Input voltage: 4.5 V' — voltage\\s+ used to require a space before ':'."""
    facts = _facts(
        "• Input voltage: 4.5 V to 5.5 V\n"
        "• Adjustable output voltage: 1.21 V to 20 V\n"
        "• Output current: 1 A\n"
    )
    assert _of(facts, "vin_range") == [[4.5, 5.5]]
    assert _of(facts, "vout_range") == [[1.21, 20.0]]
    assert _of(facts, "iout_max") == [1.0]


def test_parenthesized_and_interrupted_vin():
    facts = _facts(
        "• Wide Input Voltage Range (4.5V to 18V)\n"
        "- Wide input voltage range: 3V (falling threshold) to 65V\n"
        "LMR36503E-Q1 3V to 65V, 0.3A, Automotive, Grade 0, Synchronous Buck Converter\n"
    )
    assert [4.5, 18.0] in _of(facts, "vin_range")
    assert [3.0, 65.0] in _of(facts, "vin_range")
    assert 0.3 in _of(facts, "iout_max")


def test_schematic_vout_not_stolen_as_vin():
    facts = _facts("VOUT1 1.3V-27V VOUT2 1.3V-27V VIN 4.5V-30V")
    assert [1.3, 27.0] in _of(facts, "vout_range")
    assert [4.5, 30.0] in _of(facts, "vin_range")
    assert [1.3, 27.0] not in _of(facts, "vin_range")


def test_milliamp_iout_and_as_low_as_vout():
    facts = _facts(
        "• Output Current of 100 mA\n"
        "the regulator can deliver 100 mA output current.\n"
        "-ADJ (Outputs as Low as 1.285 V)\n"
        "High Efficiency Over a 3A to Milliamperes Load Range\n"
        "LM2695 High Voltage (30V, 1.25A) Step Down Switching Regulator\n"
    )
    assert 0.1 in _of(facts, "iout_max")
    assert 3.0 in _of(facts, "iout_max")
    assert 1.25 in _of(facts, "iout_max")
    assert 1.285 in _of(facts, "vout_min")


def test_discrete_outputs_and_fixed_family():
    facts = _facts(
        "• Regulated 5 V or 3.3 V output with selectable 400 mV headroom\n"
        "• Available in Output Voltage Options\n"
        "- Metal TO Low Profile Package LM140LA-5.0 5V LM340LA-5.0 5V "
        "LM140LA-12 12V LM340LA-12 12V LM140LA-15 15V LM340LA-15 15V\n"
    )
    assert [3.3, 5.0] in _of(facts, "vout_range")
    assert [5.0, 15.0] in _of(facts, "vout_range")


def test_reference_and_to_vin_are_vout_min():
    facts = _facts(
        "- Adjustable Output Voltage From 0.6 V to VIN\n"
        "• 0.6V ±1% voltage reference overtemperature\n"
        "• Feedback Reference Voltage 0.6 V ±1%\n"
        "• 1.2-V internal voltage reference\n"
        "• Adjustable output down to 1.25 V\n"
    )
    mins = _of(facts, "vout_min")
    assert 0.6 in mins
    assert 1.2 in mins
    assert 1.25 in mins


def test_temp_features_not_storage():
    facts = _facts(
        "• AEC-Q100 qualified -40°C to +125°C\n"  # en-dash minus becomes '-' after flatten
        "• Military temperature range (-55°C to 125°C)\n"
        "• -40°C to +125°C Junction Temperature Range\n"
        "Storage Temperature Range -65°C to +150°C\n"
    )
    temps = _of(facts, "temp_range")
    assert [-40.0, 125.0] in temps
    assert [-55.0, 125.0] in temps
    assert [-65.0, 150.0] not in temps


def test_infineon_operating_and_storage_is_kept():
    """Infineon SIPMOS / CoolSiC: combined rating, not a storage-only skip."""
    facts = _facts(
        "Operating and storage temperature T j, T stg -55 ... 150 °C\n"
        "Operating junction temperature T -55 - 175 °C\n"
        "Operating junction temperature T \u201155 \u2011\n175 °C\n"
        "Operating and storage temperature T j, T stg °C ESD 55/150/56\n-55 ... 150\n"
        "Storage Temperature Range -65°C to +150°C\n"
    )
    temps = _of(facts, "temp_range")
    assert [-55.0, 150.0] in temps
    assert [-55.0, 175.0] in temps
    assert [-65.0, 150.0] not in temps

    facts = _facts("Drain-source voltage V DSS 750 V static, T = -55°C to 175°C")
    assert [-55.0, 175.0] in _of(facts, "temp_range")


def test_optimos_product_summary_vds_rds():
    facts = _facts(
        "Product Summary\n"
        "VDS 30 30 V\n"
        "RDS(on),max VGS=10 V 5 3.7 mW\n"
        "ID 40 40 A\n"
    )
    assert 30.0 in _of(facts, "vds")
    assert 0.0037 in _of(facts, "rds_max") or 0.005 in _of(facts, "rds_max")
    assert 40.0 in _of(facts, "id_max")


def test_rohm_outline_box_vds_id_rds():
    """ROHM Si MOSFET page-1 outline: VDSS / RDS(on)(Max.) / ID, including P-ch."""
    facts = _facts(
        "lOutline VDSS 40V RDS(on)(Max.) 8.0mΩ ID ±24A  HSOP8 PD 26W"
    )
    assert _of(facts, "vds") == [40.0]
    assert _of(facts, "id_max") == [24.0]
    assert _of(facts, "rds_max") == [0.008]

    facts = _facts(
        "lOutline VDSS -45V RDS(on)(Max.) 27mΩ SOP8 ID ±7.0A PD 2.0W"
    )
    assert _of(facts, "vds") == [-45.0]
    assert _of(facts, "id_max") == [7.0]
    assert _of(facts, "rds_max") == [0.027]

    facts = _facts(
        "lOutline VDSS 600V TO-220FM RDS(on)(Max.) 0.780Ω ID ±7A PD 46W"
    )
    assert _of(facts, "vds") == [600.0]
    assert _of(facts, "id_max") == [7.0]
    assert _of(facts, "rds_max") == [0.78]

    # Symbol-font ± on older ROHM abs-max text (SP8K80).
    facts = _facts("Drain current Continuous ID \uf0b10.5 A Pulsed IDP \uf0b12 A")
    assert 0.5 in _of(facts, "id_max")

    # Infineon CoolSiC Features line.
    facts = _facts("• VDSS = 1200 V at Tvj = 25°C • IDDC = 30 A at Tc = 25°C • RDS(on) = 80 mΩ at VGS = 18 V")
    assert 1200.0 in _of(facts, "vds")
    assert 30.0 in _of(facts, "id_max")
    assert 0.08 in _of(facts, "rds_max")


def test_original_headline_forms_still_match():
    facts = _facts(
        "4.5-V to 17-V input voltage range. "
        "adjustable output voltage from 0.8 V to 15 V. "
        "6-A continuous output current."
    )
    assert [4.5, 17.0] in _of(facts, "vin_range")
    assert [0.8, 15.0] in _of(facts, "vout_range")
    assert 6.0 in _of(facts, "iout_max")
