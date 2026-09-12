from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import pymupdf  # noqa: E402

from fetch_power_description_pdfs import (  # noqa: E402
    BROWSER_ONLY,
    _pdf_link_in_html,
    is_valid_pdf,
)
from extract_power_descriptions import BOILERPLATE  # noqa: E402


def _tiny_pdf(tmp_path):
    doc = pymupdf.open()
    doc.new_page()
    path = tmp_path / "ok.pdf"
    doc.save(path)
    doc.close()
    return path


def test_valid_pdf(tmp_path):
    assert is_valid_pdf(_tiny_pdf(tmp_path))
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF not really a pdf")
    assert not is_valid_pdf(bad)


def test_pdf_link_in_landing_page(tmp_path):
    page = tmp_path / "landing.html"
    page.write_text('<html><a href="/documents/MP20041.pdf">datasheet</a></html>')
    assert _pdf_link_in_html(page) == "/documents/MP20041.pdf"
    plain = tmp_path / "plain.html"
    plain.write_text("<html>no link here</html>")
    assert _pdf_link_in_html(plain) is None


def test_browser_only_domains():
    assert "microchip.com" in BROWSER_ONLY and "st.com" in BROWSER_ONLY
    assert "ti.com" not in BROWSER_ONLY


def test_section_headings_are_boilerplate():
    for heading in ("Applications", "APPLICATIONS", "Connection Diagram", "Pin Configuration",
                    "Electrical Characteristics", "Absolute Maximum Ratings", "Ordering Information",
                    "Qualified for Automotive Applications"):
        assert BOILERPLATE.search(heading), heading
    # Real one-liners are not headings.
    assert not BOILERPLATE.search("WIDE-INPUT SYNCHRONOUS BUCK CONTROLLER")
    assert not BOILERPLATE.search("Adjustable 3-A buck converter for automotive applications")
