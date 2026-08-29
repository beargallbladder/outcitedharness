from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "serve_bge_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("serve_bge_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


def test_embed_request_is_bounded_and_preserves_texts() -> None:
    texts, batch_size = server.validate_embed_request(
        {"texts": ["alpha", "beta"], "batch_size": 1},
        maximum_batch=8,
    )

    assert texts == ["alpha", "beta"]
    assert batch_size == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"texts": []},
        {"texts": [""]},
        {"texts": ["a", "b", "c"], "batch_size": 4},
        {"texts": ["a"], "batch_size": 0},
    ],
)
def test_embed_request_rejects_invalid_payloads(payload: object) -> None:
    with pytest.raises(ValueError):
        server.validate_embed_request(payload, maximum_batch=2)
