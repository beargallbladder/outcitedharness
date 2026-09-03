from __future__ import annotations

import json

import httpx
import pytest

from harness.electronics.local_model import (
    LocalExtractionClient,
    RESPONSE_SCHEMAS,
    focused_page_context,
    local_prompt,
    parse_json_response,
    validate_local_url,
    validate_response,
)


def test_local_url_refuses_public_or_credentialed_hosts():
    assert validate_local_url("http://127.0.0.1:8082/v1").endswith("/v1")
    assert validate_local_url("http://192.168.4.46:8900/v1").endswith("/v1")
    with pytest.raises(ValueError, match="private"):
        validate_local_url("http://8.8.8.8:8082/v1")
    with pytest.raises(ValueError, match="unauthenticated"):
        validate_local_url("http://user:pass@127.0.0.1:8082/v1")


def test_response_parser_accepts_fence_but_requires_object():
    assert parse_json_response('```json\n{"facts":[]}\n```') == {"facts": []}
    with pytest.raises(ValueError, match="no JSON object"):
        parse_json_response("no answer")


def test_local_client_validates_capability_schema():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["temperature"] == 0
        assert body["enable_thinking"] is False
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"facts":[{"field":"clock","value":100,"value_role":"max","unit":"MHz","conditions":{}}]}'
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    client = LocalExtractionClient(
        base_url="http://127.0.0.1:8082/v1",
        model="qwen-local",
        transport=httpx.MockTransport(handler),
    )
    result = client.extract(
        capability="parametrics",
        page_evidence={
            "document_sha256": "a" * 64,
            "page_1based": 1,
            "blocks": [{"text": "Maximum clock 100 MHz"}],
            "tables": [],
        },
    )

    assert result["provider"] == "local"
    assert result["result"]["facts"][0]["unit"] == "MHz"


def test_focused_context_projects_selected_package_columns() -> None:
    page = {
        "document_sha256": "a" * 64,
        "page_1based": 1,
        "blocks": [
            {"text": "inside", "bbox": [0, 0, 10, 10]},
            {"text": "outside", "bbox": [100, 100, 110, 110]},
        ],
        "tables": [
            {
                "table_index": 0,
                "bbox": [0, 0, 20, 20],
                "rows": [
                    ["PKG-A", None, "PKG-B", None],
                    ["Pin", "Name", "Pin", "Name"],
                    ["1", "PA0", "9", "PB9"],
                ],
            }
        ],
        "structural_evidence": {
            "regions": [{"table_index": 0, "bbox": [0, 0, 20, 20]}],
            "package_scope": {
                "package": "PKG-A",
                "column_header": "PKG-A",
            },
        },
    }
    focused = focused_page_context("pin_or_ball", page)
    assert [block["text"] for block in focused["blocks"]] == ["inside"]
    assert focused["tables"][0]["rows"][2] == ["1", "PA0"]


def test_vision_prompt_excludes_pdf_extracted_table_text() -> None:
    prompt = local_prompt(
        "pin_or_ball",
        {
            "document_sha256": "a" * 64,
            "page_1based": 1,
            "blocks": [{"text": "DO NOT LEAK THIS EXTRACTED TEXT"}],
            "tables": [
                {
                    "table_index": 0,
                    "bbox": [0, 0, 20, 20],
                    "rows": [["Pin", "Name"], ["1", "PA0"]],
                }
            ],
            "structural_evidence": {
                "regions": [
                    {"table_index": 0, "bbox": [0, 0, 20, 20]}
                ],
                "package_scope": {
                    "package": "LQFP64",
                    "column_header": "LQFP64",
                },
            },
        },
        include_page_evidence=False,
    )

    assert "DO NOT LEAK" not in prompt
    assert '"rows"' not in prompt
    assert "Read only the supplied rendered datasheet page image" in prompt
    assert "LQFP64" in prompt


def test_focused_context_accepts_text_region_without_table_index() -> None:
    focused = focused_page_context(
        "opn_decoder",
        {
            "document_sha256": "a" * 64,
            "page_1based": 7,
            "blocks": [
                {
                    "text": "Ordering Information",
                    "bbox": [0, 0, 100, 20],
                }
            ],
            "tables": [],
            "structural_evidence": {
                "regions": [
                    {
                        "table_index": None,
                        "bbox": [0, 0, 100, 20],
                    }
                ]
            },
        },
    )

    assert focused["blocks"][0]["text"] == "Ordering Information"
    assert focused["tables"] == []


def test_focused_context_keeps_semantics_for_only_selected_package() -> None:
    page = {
        "document_sha256": "a" * 64,
        "page_1based": 1,
        "blocks": [],
        "tables": [
            {
                "table_index": 0,
                "bbox": [0, 0, 100, 100],
                "rows": [
                    [
                        "Pin/ball name",
                        None,
                        None,
                        None,
                        None,
                        "Pin name",
                        "Pin type",
                        "I/O structure",
                        "Alternate functions",
                    ],
                    [
                        "LQFP144",
                        "UFBGA176+25",
                        "LQFP176",
                        "LQFP208",
                        "TFBGA240+25",
                        None,
                        None,
                        None,
                        None,
                    ],
                    [
                        "1",
                        "C3",
                        "3",
                        "4",
                        "D1",
                        "PE2",
                        "I/O",
                        "FT_h",
                        "TRACECLK",
                    ],
                ],
            }
        ],
        "structural_evidence": {
            "regions": [{"table_index": 0, "bbox": [0, 0, 100, 100]}],
            "package_scope": {
                "package": "LQFP176",
                "column_header": "LQFP176",
            },
        },
    }

    focused = focused_page_context("pin_semantics", page)

    assert focused["tables"][0]["projected_source_columns"] == [2, 5, 6, 7, 8]
    assert focused["tables"][0]["rows"][2] == [
        "3",
        "PE2",
        "I/O",
        "FT_h",
        "TRACECLK",
    ]


def test_summary_and_opn_schemas_require_typed_items() -> None:
    validate_response(
        {
            "summary": "Low-power MCU.",
            "characteristics": ["Low-power"],
            "applications": ["Wearables"],
        },
        RESPONSE_SCHEMAS["series_summary"],
    )
    validate_response(
        {
            "series": "ABC",
            "base_part": "ABC123",
            "package_code": "QFN",
            "suffixes": [{"code": "T", "meaning": "Tape and reel"}],
        },
        RESPONSE_SCHEMAS["opn_decoder"],
    )
    with pytest.raises(ValueError, match=r"\$\.suffixes\[0\]: expected object"):
        validate_response(
            {
                "series": "ABC",
                "base_part": "ABC123",
                "package_code": "QFN",
                "suffixes": ["T"],
            },
            RESPONSE_SCHEMAS["opn_decoder"],
        )
