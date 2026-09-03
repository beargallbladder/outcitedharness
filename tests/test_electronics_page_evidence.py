from __future__ import annotations

from harness.electronics.page_evidence import (
    extract_profile_evidence,
    selected_pages,
)


class _Rect:
    width = 100
    height = 200


class _Table:
    bbox = (1, 2, 90, 40)

    def extract(self):
        return [["Pin", "Name"], ["1", "PA0"]]


class _Finder:
    tables = [_Table()]


class _Page:
    rect = _Rect()

    def get_text(self, mode):
        assert mode == "blocks"
        return [(1, 2, 90, 10, "Pin descriptions", 0, 0)]

    def find_tables(self):
        return _Finder()


class _Document:
    page_count = 2

    def __getitem__(self, index):
        return _Page()


def test_exact_pin_pages_override_per_lane_limit():
    profile = {
        "lane_pages": {"parametrics": [1, 2], "pin_or_ball": [1]},
        "exact_pin_locations": [
            {"status": "send", "pages_1based": [2]},
        ],
    }

    pages = selected_pages(profile, maximum_pages_per_lane=1)

    assert pages[1] == {"parametrics", "pin_or_ball"}
    assert pages[2] == {"pin_or_ball", "pin_semantics"}


def test_page_evidence_preserves_blocks_tables_and_coordinates():
    profile = {
        "document_sha256": "a" * 64,
        "source_path": "/corpus/atom.pdf",
        "lane_pages": {"pin_or_ball": [1]},
        "exact_pin_locations": [],
    }

    rows = list(
        extract_profile_evidence(
            _Document(),
            profile,
            maximum_pages_per_lane=2,
        )
    )

    assert len(rows) == 1
    assert rows[0]["blocks"][0]["bbox"] == [1.0, 2.0, 90.0, 10.0]
    assert rows[0]["tables"][0]["rows"][1] == ["1", "PA0"]
    assert rows[0]["extractor"]["network_used"] is False
