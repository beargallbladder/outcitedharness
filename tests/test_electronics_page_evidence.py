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

    def get_text(self, mode, textpage=None):
        if mode == "blocks":
            return [(1, 2, 90, 10, "Pin descriptions", 0, 0)]
        if mode == "dict":
            return {
                "blocks": [
                    {
                        "type": 0,
                        "bbox": (1, 2, 90, 10),
                        "lines": [
                            {
                                "bbox": (1, 2, 90, 10),
                                "spans": [{"text": "Pin descriptions"}],
                            }
                        ],
                    }
                ]
            }
        raise AssertionError(mode)

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
    assert rows[0]["layout_ir"]["text_blocks"][0]["lines"][0]["text"] == "Pin descriptions"
    assert rows[0]["extractor"]["network_used"] is False
    assert rows[0]["extractor"]["ocr_used"] is False
    assert rows[0]["extraction_status"] == "text_layer"
    assert rows[0]["needs_visual_verification"] is False


class _NoTextPage:
    rect = _Rect()

    def get_text(self, mode, textpage=None):
        if mode == "blocks":
            return []
        if mode == "dict":
            return {"blocks": []}
        raise AssertionError(mode)

    def find_tables(self):
        return _Finder()


class _NoTextDocument:
    page_count = 1

    def __getitem__(self, index):
        return _NoTextPage()


def test_page_evidence_marks_no_text_pages_for_visual_verification():
    profile = {
        "document_sha256": "a" * 64,
        "source_path": "/corpus/scan.pdf",
        "lane_pages": {"pin_or_ball": [1]},
        "exact_pin_locations": [],
    }

    rows = list(
        extract_profile_evidence(
            _NoTextDocument(),
            profile,
            maximum_pages_per_lane=2,
        )
    )

    assert len(rows) == 1
    assert rows[0]["blocks"] == []
    assert rows[0]["extraction_status"] == "no_text_layer"
    assert rows[0]["needs_visual_verification"] is True


class _OCRPage(_NoTextPage):
    def get_textpage_ocr(self, language="eng"):
        assert language == "eng"
        return object()

    def get_text(self, mode, textpage=None):
        if mode == "blocks" and textpage is not None:
            return [(1, 2, 90, 10, "OCR Pin descriptions", 0, 0)]
        return super().get_text(mode, textpage=textpage)


class _OCRDocument:
    page_count = 1

    def __getitem__(self, index):
        return _OCRPage()


def test_page_evidence_uses_ocr_fallback_when_enabled():
    profile = {
        "document_sha256": "a" * 64,
        "source_path": "/corpus/scan.pdf",
        "lane_pages": {"pin_or_ball": [1]},
        "exact_pin_locations": [],
    }

    rows = list(
        extract_profile_evidence(
            _OCRDocument(),
            profile,
            maximum_pages_per_lane=2,
            enable_ocr_fallback=True,
            ocr_language="eng",
        )
    )

    assert len(rows) == 1
    assert rows[0]["blocks"][0]["text"] == "OCR Pin descriptions"
    assert rows[0]["extractor"]["ocr_attempted"] is True
    assert rows[0]["extractor"]["ocr_used"] is True
    assert rows[0]["extraction_status"] == "ocr_fallback"
    assert rows[0]["needs_visual_verification"] is False
