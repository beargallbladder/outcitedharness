from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_datasheet_modality_fixture.py"
SPEC = importlib.util.spec_from_file_location("datasheet_fixture", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fixture_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture_builder)


def _sources(root: Path) -> Path:
    pdf = root / "source.pdf"
    pdf.write_bytes(b"%PDF-1.7\nfixture")
    truth = root / "truth.json"
    truth.write_text(
        json.dumps(
            {
                "pins": [
                    {"pin_no": 1, "name": "VDD"},
                    {"pin_no": 2, "name": "VSS"},
                ],
                "n_pins": 2,
            }
        )
    )
    gold = root / "gold.json"
    gold.write_text(
        json.dumps(
            [
                {
                    "stem": "part-1",
                    "vendor": "vendor",
                    "bucket": "small",
                    "n_pins_gt": 2,
                    "packages": ["LQFP2"],
                    "pdf_path": str(pdf),
                    "gt_snapshot": str(truth),
                }
            ]
        )
    )
    return gold


def test_fixture_builder_copies_and_hash_binds_sources(tmp_path: Path):
    gold = _sources(tmp_path)
    output = tmp_path / "fixture"

    manifest = fixture_builder.build_fixture(
        gold_set=gold,
        output_root=output,
        case_ids=["part-1"],
    )

    assert manifest["schema"] == "harness.datasheet-modality-fixture.v1"
    case = manifest["cases"][0]
    assert case["requested_package"] == "LQFP2"
    assert case["expected_ground_truth_rows"] == 2
    assert (
        fixture_builder.sha256_file(output / case["pdf"]) == case["pdf_sha256"]
    )
    assert (
        fixture_builder.sha256_file(output / case["ground_truth"])
        == case["ground_truth_sha256"]
    )


def test_fixture_builder_rejects_inconsistent_ground_truth(tmp_path: Path):
    gold = _sources(tmp_path)
    payload = json.loads(gold.read_text())
    payload[0]["n_pins_gt"] = 3
    gold.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="pin count is inconsistent"):
        fixture_builder.build_fixture(
            gold_set=gold,
            output_root=tmp_path / "fixture",
            case_ids=["part-1"],
        )


def test_fixture_builder_rejects_duplicates_that_leave_lqfp_range_incomplete(
    tmp_path: Path,
):
    gold = _sources(tmp_path)
    payload = json.loads(gold.read_text())
    truth = Path(payload[0]["gt_snapshot"])
    truth_payload = json.loads(truth.read_text())
    truth_payload["pins"][1]["pin_no"] = 1
    truth.write_text(json.dumps(truth_payload))

    with pytest.raises(ValueError, match="do not match the package range"):
        fixture_builder.build_fixture(
            gold_set=gold,
            output_root=tmp_path / "fixture",
            case_ids=["part-1"],
        )


def test_fixture_builder_rejects_lqfp_truth_outside_package_range(
    tmp_path: Path,
):
    gold = _sources(tmp_path)
    payload = json.loads(gold.read_text())
    truth = Path(payload[0]["gt_snapshot"])
    truth_payload = json.loads(truth.read_text())
    truth_payload["pins"][1]["pin_no"] = 3
    truth.write_text(json.dumps(truth_payload))

    with pytest.raises(ValueError, match="do not match the package range"):
        fixture_builder.build_fixture(
            gold_set=gold,
            output_root=tmp_path / "fixture",
            case_ids=["part-1"],
        )


def test_fixture_builder_refuses_existing_output(tmp_path: Path):
    gold = _sources(tmp_path)
    output = tmp_path / "fixture"
    output.mkdir()

    with pytest.raises(ValueError, match="output already exists"):
        fixture_builder.build_fixture(
            gold_set=gold,
            output_root=output,
            case_ids=["part-1"],
        )
