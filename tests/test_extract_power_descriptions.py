from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import pymupdf  # noqa: E402

from extract_power_descriptions import (  # noqa: E402
    _is_garbled,
    _is_part_numberish,
    meets_floor,
    page1_description,
)


def _pdf_with_lines(tmp_path, name, lines, right_column=None):
    """lines: list of (text, size, y). right_column: same shape, placed in a
    second column (x=350) to exercise two-column layouts."""
    doc = pymupdf.open()
    page = doc.new_page()
    for text, size, y in lines:
        page.insert_text((72, y), text, fontsize=size)
    for text, size, y in (right_column or []):
        page.insert_text((350, y), text, fontsize=size)
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return path


class TestFloor:
    def test_floor_requires_length_and_digit(self):
        assert meets_floor("35V Voltage Resistance 1A LDO Regulators")
        assert not meets_floor("Nch 600V 25A Power MOSFET")  # 25 chars
        assert not meets_floor("Dual N-Channel OptiMOS MOSFET transistor line")  # no digit


class TestGarbled:
    def test_cid_soup_is_garbled(self):
        assert _is_garbled("	\n")

    def test_control_chars_as_spaces_are_not_garbled(self):
        # Infineon maps the space to \x03 in some encodings.
        assert not _is_garbled("OptiMOS\x035\x03Power-Transistor,\x0330\x03V")

    def test_normal_text_is_not_garbled(self):
        assert not _is_garbled("Nch 600V 25A Power MOSFET")


class TestPartNumberish:
    def test_matches_with_suffixes(self):
        assert _is_part_numberish("IRFI3205PbF", "IRFI3205")
        assert _is_part_numberish("BSC0921NDI", "BSC0921NDI")

    def test_description_lines_do_not_match(self):
        assert not _is_part_numberish("Dual N-Channel OptiMOS MOSFET", "BSC0921NDI")


class TestPage1Description:
    def test_largest_tagline_wins(self, tmp_path):
        pdf = _pdf_with_lines(tmp_path, "a.pdf", [
            ("BD90C0AFP-C", 14, 40),
            ("Single-Output LDO Regulators", 12, 90),
            ("35V Voltage Resistance", 24, 105),
            ("1A LDO Regulators", 24, 130),
            ("Features", 12, 200),
        ])
        result = page1_description(pdf, "BD90C0AFP-C")
        assert result["candidate"] == "tagline"
        assert result["description_verbatim"] == "35V Voltage Resistance 1A LDO Regulators"

    def test_description_section_first_sentence(self, tmp_path):
        pdf = _pdf_with_lines(tmp_path, "b.pdf", [
            ("IRFI3205PbF", 14, 40),
            ("VDSS 55V", 10, 90),
            ("Description", 10, 120),
            ("Fifth generation HEXFETs from International Rectifier utilize", 10, 140),
            ("advanced processing techniques to achieve low resistance.", 10, 155),
        ])
        result = page1_description(pdf, "IRFI3205")
        assert result["candidate"] == "description_section"
        assert result["description_verbatim"].startswith("Fifth generation")
        assert result["description_verbatim"].endswith("resistance.")

    def test_footer_and_boilerplate_excluded(self, tmp_path):
        pdf = _pdf_with_lines(tmp_path, "c.pdf", [
            ("R6025JNX", 16, 40),
            ("Nch 600V 25A Power MOSFET", 24, 70),
            ("www.rohm.com", 8, 790),
            ("© 2017 ROHM Co., Ltd. All rights reserved.", 8, 795),
        ])
        result = page1_description(pdf, "R6025JNX")
        assert result["description_verbatim"] == "Nch 600V 25A Power MOSFET"

    def test_page_without_one_liner(self, tmp_path):
        pdf = _pdf_with_lines(tmp_path, "d.pdf", [
            ("BSC0921NDI", 16, 40),
            ("Features", 12, 90),
            ("Datasheet", 10, 110),
            ("Rev. 2.1, 2020-10-23", 8, 790),
        ])
        assert page1_description(pdf, "BSC0921NDI") == {"no_description_line": True}

    def test_two_column_features_never_splice_into_description(self, tmp_path):
        """CR remaining-527-ack-20260912: the y-sorted join spliced Features
        bullets (right column) into the description sentence (left column).
        Block-aware extraction must keep the columns apart."""
        pdf = _pdf_with_lines(tmp_path, "e.pdf", [
            ("LM2578A", 14, 40),
            ("Description", 10, 120),
            ("The LM2578A is a switching regulator which can", 10, 135),
            ("Inverting and Non-Inverting", 10, 150),
            ("Feedback Inputs", 10, 165),
            ("be configured as a buck, boost, or inverting", 10, 195),
            ("converter with a single ended primary.", 10, 210),
        ], right_column=[
            ("− Inverting and Non-Inverting Feedback Inputs", 10, 135),
            ("− Ideal Load and Line Transient Responses", 10, 150),
        ])
        result = page1_description(pdf, "LM2578A")
        assert result["candidate"] == "description_section"
        assert result["description_verbatim"].startswith("The LM2578A is a switching regulator which can")
        assert "Feedback Inputs" not in result["description_verbatim"] or "converter" in result["description_verbatim"]

    def test_headerless_paragraph_opener_renesas_style(self, tmp_path):
        """Renesas covers open with the description paragraph directly under
        the title, no Description header (CR recovered +14 gold from our own
        fetch cache that our picker keyed past)."""
        pdf = _pdf_with_lines(tmp_path, "f.pdf", [
            ("Datasheet", 16, 43),
            ("The RAA210130 is a fully PMBus enabled DC/DC", 10, 140),
            ("step-down power supply capable of delivering up to", 10, 153),
            ("30A of current from a compact BGA package.", 10, 166),
        ])
        result = page1_description(pdf, "RAA210130")
        assert result["candidate"] == "description_section"
        assert result["description_verbatim"] == (
            "The RAA210130 is a fully PMBus enabled DC/DC step-down power supply "
            "capable of delivering up to 30A of current from a compact BGA package."
        )
