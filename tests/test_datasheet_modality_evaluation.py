from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_datasheet_modalities.py"
SPEC = importlib.util.spec_from_file_location("datasheet_evaluation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_prediction_contract_rejects_runaway_or_malformed_rows():
    pins, error = evaluation.prediction_pins(
        {"pins": [{"pin_no": index, "name": "PA0"} for index in range(5)]},
        maximum_rows=4,
    )
    assert pins == []
    assert error == "response_exceeds_pin_row_bound"

    pins, error = evaluation.prediction_pins(
        {"pins": [{"pin_no": None, "name": "PA0"}]},
        maximum_rows=4,
    )
    assert pins == []
    assert error == "pin_row_has_invalid_types"

    pins, error = evaluation.prediction_pins(
        {"pins": [{"pin_no": "-", "name": "PA0"}]},
        maximum_rows=4,
    )
    assert pins == []
    assert error == "pin_row_has_non_physical_identifier"


def test_scoring_requires_physical_pin_and_name_pair_agreement():
    truth = {
        "pins": [
            {"pin_no": 1, "name": "PA0"},
            {"pin_no": 2, "name": "PA1"},
        ]
    }
    score = evaluation.score_prediction(
        truth,
        [
            {"pin_no": "1", "name": "PA0"},
            {"pin_no": "2", "name": "WRONG"},
        ],
    )

    assert score["predicted_rows"] == 2
    assert score["count_within_tolerance"] is True
    assert score["pair_f1"] == 0.5
    assert score["pin_number_recall"] == 1.0
    assert score["pin_name_recall"] == 0.5


def test_scoring_treats_not_connected_as_nc():
    score = evaluation.score_prediction(
        {"pins": [{"pin_no": 73, "name": "NC"}]},
        [{"pin_no": "73", "name": "Not connected"}],
    )

    assert score["pair_f1"] == 1.0
    assert score["pin_name_recall"] == 1.0


def test_summary_reports_image_minus_text_delta():
    cases = [
        {
            "locator": {"status": "send", "reason": "matched"},
            "modalities": {
                "table": {
                    "complete": True,
                    "contract_valid": True,
                    "score": {
                        "count_within_tolerance": True,
                        "pair_f1": 0.8,
                        "pin_number_recall": 1.0,
                        "pin_name_recall": 1.0,
                    },
                },
                "text": {
                    "complete": True,
                    "contract_valid": False,
                    "score": {
                        "count_within_tolerance": False,
                        "pair_f1": 0.25,
                        "pin_number_recall": 0.5,
                        "pin_name_recall": 0.5,
                    },
                },
                "image": {
                    "complete": True,
                    "contract_valid": True,
                    "score": {
                        "count_within_tolerance": True,
                        "pair_f1": 0.75,
                        "pin_number_recall": 1.0,
                        "pin_name_recall": 1.0,
                    },
                },
                "image_focused": {
                    "complete": True,
                    "contract_valid": True,
                    "score": {
                        "count_within_tolerance": True,
                        "pair_f1": 0.9,
                        "pin_number_recall": 1.0,
                        "pin_name_recall": 1.0,
                    },
                },
                "image_rows": {
                    "complete": True,
                    "contract_valid": True,
                    "score": {
                        "count_within_tolerance": True,
                        "pair_f1": 0.95,
                        "pin_number_recall": 1.0,
                        "pin_name_recall": 1.0,
                    },
                },
            },
        },
        {
            "locator": {"status": "withhold", "reason": "no_exact_package"},
            "modalities": {},
        },
    ]

    summary = evaluation.summarize(cases)

    assert summary["sent"] == 1
    assert summary["withheld"] == 1
    assert summary["modes"]["image"]["count_within_tolerance"] == 1
    assert summary["modes"]["table"]["contract_valid_cases"] == 1
    assert summary["image_minus_text_pair_f1"] == 0.5
    assert summary["image_focused_minus_text_pair_f1"] == 0.65
    assert summary["image_focused_minus_image_pair_f1"] == pytest.approx(0.15)
    assert summary["image_focused_minus_table_pair_f1"] == pytest.approx(0.1)
    assert summary["image_rows_minus_image_focused_pair_f1"] == pytest.approx(
        0.05
    )
    assert summary["image_rows_minus_table_pair_f1"] == pytest.approx(0.15)


def test_focused_geometry_selects_only_package_and_pin_name_columns():
    rows = [
        ["Pin Number", None, "Pin name"],
        ["LQFP2", None, None],
        ["1", None, "PA0"],
    ]

    class Row:
        def __init__(self, cells):
            self.cells = cells

    class Table:
        bbox = (0, 0, 40, 30)

        def __init__(self):
            self.rows = [
                Row([(0, 0, 10, 10), None, (20, 0, 40, 10)]),
                Row([(0, 10, 10, 20), None, (20, 10, 40, 20)]),
                Row([(0, 20, 10, 30), None, (20, 20, 40, 30)]),
            ]

        def extract(self):
            return rows

    class Tables:
        tables = [Table()]

    class Page:
        def find_tables(self):
            return Tables()

    assert evaluation._focused_table_geometry(Page(), "LQFP2") == (
        (0.0, 0.0, 40.0, 30.0),
        (0.0, 0.0, 10.0, 30.0),
        (20.0, 0.0, 40.0, 30.0),
    )
    pins, evidence = evaluation._extract_table_page(
        Page(),
        selected_column="LQFP2",
    )
    assert pins == [{"pin_no": "1", "name": "PA0"}]
    assert evidence["rendering"] == "table_cells"
    selected, evidence_rows = evaluation._selected_physical_rows(
        rows,
        package_row=1,
        package_column=0,
        name_row=0,
        name_column=2,
    )
    assert selected == [(2, "1", "PA0")]
    assert evidence_rows == [["1", "PA0"]]


def test_result_writer_is_immutable(tmp_path: Path):
    output = tmp_path / "result.json"
    evaluation.write_new_json(output, {"passed": True})
    assert json.loads(output.read_text()) == {"passed": True}

    with pytest.raises(ValueError, match="already exists"):
        evaluation.write_new_json(output, {"passed": False})


def test_endpoint_identity_requires_one_model_and_runtime_version():
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "vision-model"}]})
        if request.url.path == "/v1/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert (
            evaluation.discover_model(client, "http://vision.test/v1")
            == "vision-model"
        )
        assert (
            evaluation.discover_runtime_version(
                client,
                "http://vision.test/v1",
            )
            == "0.21.0"
        )


def test_endpoint_identity_allows_only_declared_lora_companion():
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "qwen3-vl-8b-instruct"},
                    {"id": "datasheet-frontier-adapter"},
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert (
            evaluation.discover_model(
                client,
                "http://vision.test/v1",
                configured_model="datasheet-frontier-adapter",
                allowed_companion_models=("qwen3-vl-8b-instruct",),
            )
            == "datasheet-frontier-adapter"
        )
        with pytest.raises(ValueError, match="declared models"):
            evaluation.discover_model(
                client,
                "http://vision.test/v1",
                configured_model="datasheet-frontier-adapter",
            )


def test_runtime_identity_falls_back_to_vllm_root_version_route():
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/version":
            return httpx.Response(404)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.21.0"})
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert (
            evaluation.discover_runtime_version(
                client,
                "http://vision.test/v1",
            )
            == "0.21.0"
        )
