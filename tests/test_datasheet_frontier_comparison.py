from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path

import httpx

from harness.config import ModelConfig


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_datasheet_frontier.py"
SPEC = importlib.util.spec_from_file_location("datasheet_frontier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
frontier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frontier)


def test_three_way_consensus_is_training_eligible_only_for_candidates():
    local = [{"pin_no": "73", "name": "Not connected"}]
    table = [{"pin_no": "73", "name": "NC"}]
    teacher = [{"pin_no": "73", "name": "No connect"}]

    candidate = frontier.compare_page(
        local_pins=local,
        table_pins=table,
        frontier_pins=teacher,
        frontier_parse_error=None,
        truth_pins=[{"pin_no": 73, "name": "NC"}],
        split_role="candidate",
    )
    holdout = frontier.compare_page(
        local_pins=local,
        table_pins=table,
        frontier_pins=teacher,
        frontier_parse_error=None,
        truth_pins=[{"pin_no": 73, "name": "NC"}],
        split_role="frozen_holdout",
    )

    assert candidate["independent_three_way_consensus"] is True
    assert candidate["training_eligible"] is True
    assert holdout["independent_three_way_consensus"] is True
    assert holdout["training_eligible"] is False


def test_disagreement_is_quarantined():
    comparison = frontier.compare_page(
        local_pins=[{"pin_no": "1", "name": "PA0"}],
        table_pins=[{"pin_no": "1", "name": "PA0"}],
        frontier_pins=[{"pin_no": "1", "name": "PB0"}],
        frontier_parse_error=None,
        truth_pins=[],
        split_role="candidate",
    )

    assert comparison["independent_three_way_consensus"] is False
    assert comparison["training_eligible"] is False


def test_frozen_lineage_overlap_is_found_before_processing():
    overlaps = frontier.frozen_lineage_overlaps(
        [
            {"id": "train-a", "pdf_sha256": "a" * 64},
            {"id": "train-b", "pdf_sha256": "b" * 64},
        ],
        {"b" * 64},
    )

    assert overlaps == ["train-b"]


def test_frontier_chat_sends_anthropic_image_without_persisting_key(monkeypatch):
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "test-secret-value")
    model = ModelConfig(
        key="frontier",
        tier=4,
        display_name="Frontier",
        short_name="FRONTIER",
        provider="anthropic",
        base_url="https://api.anthropic.test/v1",
        model="claude-test",
        api_key_env="TEST_ANTHROPIC_KEY",
        capabilities={"vision": True},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        image = payload["messages"][0]["content"][0]["source"]["data"]
        assert base64.b64decode(image) == b"png-bytes"
        assert request.headers["x-api-key"] == "test-secret-value"
        return httpx.Response(
            200,
            headers={"request-id": "request-1"},
            json={
                "id": "message-1",
                "model": "claude-test",
                "content": [
                    {
                        "type": "text",
                        "text": '{"pins":[{"pin_no":"1","name":"PA0"}]}',
                    }
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        text, evidence = frontier.frontier_chat(
            client,
            model=model,
            instruction="Extract pins.",
            image=b"png-bytes",
            max_tokens=256,
        )

    assert json.loads(text)["pins"][0]["name"] == "PA0"
    assert evidence["request_id"] == "request-1"
    assert evidence["model"] == "claude-test"
    assert "test-secret-value" not in json.dumps(evidence)
