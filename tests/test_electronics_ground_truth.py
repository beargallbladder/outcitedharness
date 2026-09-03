from harness.electronics.ground_truth import rows_for_package


def test_rows_are_used_only_for_compatible_package_scope() -> None:
    records = [
        {
            "packages": ("48LD TQFP",),
            "rows": [{"pin_no": 1, "name": "PA0"}],
        },
        {
            "packages": ("35-ball WLCSP",),
            "rows": [{"pin_no": "A1", "name": "VDD"}],
        },
    ]
    assert rows_for_package(records, "48-TQFP") == [
        {"pin_no": 1, "name": "PA0"}
    ]
    assert rows_for_package(records, "35-ball WLCSP") == [
        {"pin_no": "A1", "name": "VDD"}
    ]
    assert rows_for_package(records, "64-TQFP") == []
