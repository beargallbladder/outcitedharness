from __future__ import annotations

from pathlib import Path

from harness.electronics.page_index import (
    classify_section,
    index_document,
    package_requests_from_ground_truth,
)


class _Page:
    def __init__(self, text: str):
        self.text = text

    def get_text(self, mode: str):
        assert mode == "text"
        return self.text

    def find_tables(self):
        return None


class _Document:
    def __init__(self, toc, pages):
        self.toc = toc
        self.pages = pages
        self.page_count = len(pages)

    def get_toc(self):
        return self.toc

    def __getitem__(self, index):
        return self.pages[index]


def test_section_classifier_keeps_extraction_lanes_distinct():
    assert classify_section("5. Pin and ball descriptions") == (
        "pin_or_ball",
        "pin_semantics",
    )
    assert classify_section("Absolute maximum ratings") == ("parametrics",)
    assert classify_section("Ordering information") == ("opn_decoder",)


def test_page_index_uses_toc_then_text_fallback():
    document = _Document(
        [[1, "Pin descriptions", 2]],
        [
            _Page("Features\nApplications"),
            _Page("Pin descriptions"),
            _Page("Absolute maximum ratings"),
        ],
    )

    result = index_document(
        document,
        document_sha256="a" * 64,
        source_path=Path("/corpus/atom.pdf"),
    )

    assert result["lane_pages"]["pin_or_ball"] == [2]
    assert result["lane_pages"]["pin_semantics"] == [2]
    assert result["lane_pages"]["series_summary"] == [1]
    assert result["lane_pages"]["parametrics"] == [3]


def test_package_requests_use_each_printed_package_count():
    requests = package_requests_from_ground_truth(
        [
            {
                "pinout": {
                    "packages": [
                        "144-Pin QFP (PGE)",
                        "337-Ball BGA (ZWT)",
                    ],
                    "declared_pin_total": 144,
                }
            }
        ]
    )

    assert requests == (
        ("144-Pin QFP (PGE)", 144),
        ("337-Ball BGA (ZWT)", 337),
    )
