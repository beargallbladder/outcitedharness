from __future__ import annotations

from dataclasses import dataclass

from harness.electronics.vision_alignment import (
    align_record,
    definition_pages,
    resolve_package_header,
)


def _targets(count: int = 8) -> list[dict[str, object]]:
    return [
        {"pin_no": index, "name": f"P{index}"}
        for index in range(1, count + 1)
    ]


def _table(
    *,
    package: str = "LQFP8",
    second_package: str | None = None,
) -> dict:
    header = ["Pin/ball name"]
    packages = [package]
    if second_package:
        header.append(None)
        packages.append(second_package)
    header.append("Pin name")
    rows = [header, packages + [None]]
    for index in range(1, 9):
        numbers: list[object] = [str(index)]
        if second_package:
            numbers.append(f"A{index}")
        rows.append(numbers + [f"P{index}"])
    return {
        "index": 0,
        "bbox": (1.0, 2.0, 100.0, 200.0),
        "rows": rows,
        "row_bboxes": [
            (1.0, 2.0 + index * 10, 100.0, 12.0 + index * 10)
            for index in range(len(rows))
        ],
    }


def test_package_header_resolves_family_and_vendor_code() -> None:
    assert (
        resolve_package_header(
            "144 PGE",
            ["144-Pin QFP (PGE)", "337-Ball BGA (ZWT)"],
        )
        == "144-Pin QFP (PGE)"
    )
    assert (
        resolve_package_header(
            "LQFP176",
            ["UFBGA176", "LQFP176", "LQFP208"],
        )
        == "LQFP176"
    )


def test_align_record_requires_exact_visible_identities() -> None:
    result = align_record(
        tables_by_page={12: [_table()]},
        target_rows=_targets(),
        package_candidates=["LQFP8"],
    )

    assert result["status"] == "aligned"
    assert result["selected_package"] == "LQFP8"
    assert result["matched_rows"] == 8
    assert result["coverage"] == 1.0
    assert result["row_crop_examples"] == 1
    assert result["row_crop_target_rows"] == 8
    assert result["tables"][0]["number_column"] == 0
    assert result["tables"][0]["name_column"] == 1


def test_align_record_selects_package_column_from_target_pairs() -> None:
    result = align_record(
        tables_by_page={12: [_table(second_package="UFBGA8")]},
        target_rows=_targets(),
        package_candidates=["LQFP8", "UFBGA8"],
    )

    assert result["status"] == "aligned"
    assert result["selected_package"] == "LQFP8"
    assert result["package_candidates_scored"] == {"LQFP8": 8}


def test_align_record_withholds_low_coverage() -> None:
    table = _table()
    table["rows"] = table["rows"][:7]

    result = align_record(
        tables_by_page={12: [table]},
        target_rows=_targets(),
        package_candidates=["LQFP8"],
    )

    assert result["status"] == "withhold"
    assert result["reason"] == "fewer_than_eight_exact_visible_rows"


@dataclass
class _Page:
    text: str = ""

    def get_text(self) -> str:
        return self.text


class _Document:
    page_count = 120

    def get_toc(self) -> list[list[object]]:
        return [
            [1, "Pinout, pin description and alternate functions", 56],
            [2, "Figure 8. LQFP pinout", 58],
            [2, "Table 8. MCU pin/ball definition", 62],
            [2, "Table 9. Alternate function mapping", 71],
        ]

    def __getitem__(self, _index: int) -> _Page:
        return _Page()


def test_definition_pages_prefers_specific_definition_table() -> None:
    assert definition_pages(_Document()) == tuple(range(62, 71))
