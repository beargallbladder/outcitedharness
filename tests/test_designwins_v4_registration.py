from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.training.models import SourceProvenance, TextPair
from harness.training.registry import canonical_json


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "register_designwins_v4_dataset.py"
SPEC = importlib.util.spec_from_file_location("register_designwins_v4", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
registration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(registration)


def _provenance(content_sha256: str) -> SourceProvenance:
    return SourceProvenance(
        source_kind="designwins",
        source_uri="designwins://owned/part-1",
        source_record_id="part-1:text",
        collected_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        content_sha256=content_sha256,
        lineage_id="designwins:part-1",
        license="owned-internal",
        revision="a" * 64,
    )


def _pairs() -> tuple[TextPair, TextPair, dict]:
    parent_source = (
        "1\nPA0\nI/O\nGPIO zero and UART TX.\n"
        "2\nPA1\nI/O\nGPIO one and UART RX.\n"
    )
    parent_response = json.dumps(
        {
            "pins": [
                {
                    "pin_no": 1,
                    "name": "PA0",
                    "type": "gpio",
                    "functions": ["GPIO zero", "UART TX"],
                    "dir": "I/O",
                },
                {
                    "pin_no": 2,
                    "name": "PA1",
                    "type": "gpio",
                    "functions": ["GPIO one", "UART RX"],
                    "dir": "I/O",
                },
            ]
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    parent = TextPair(
        pair_id="part-1",
        prompt=f"Extract pins.{registration.MARKER}{parent_source}",
        response=parent_response,
        provenance=_provenance("a" * 64),
        metadata={"part": "part-1"},
    )
    child_source = parent_source[:35].strip()
    child_prompt = (
        "Extract every pin definition present."
        f"{registration.MARKER}{child_source}"
    )
    child_response = json.dumps(
        {"pins": [json.loads(parent_response)["pins"][0]]},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(
        canonical_json({"prompt": child_prompt, "response": child_response})
    ).hexdigest()
    child = TextPair(
        pair_id="part-1-chunk-0000",
        prompt=child_prompt,
        response=child_response,
        provenance=_provenance(digest).model_copy(
            update={"source_record_id": "part-1:text:chunk:0"}
        ),
        metadata={
            "part": "part-1",
            "parent_pair_id": "part-1",
            "source_start": 0,
            "source_end": 35,
            "sequence_tokens": 100,
        },
    )
    llama = {
        "instruction": child.prompt,
        "input": "",
        "output": child.response,
    }
    return parent, child, llama


def test_chunk_registration_proof_binds_parent_and_llamafactory_record():
    parent, child, llama = _pairs()

    proof = registration.validate_chunk(
        child,
        parent,
        llama,
        cutoff_len=4096,
    )

    assert proof["parent_pair_id"] == "part-1"
    assert proof["parent_content_sha256"] == "a" * 64
    assert proof["sequence_tokens"] == 100


def test_chunk_registration_rejects_response_not_in_parent():
    parent, child, _llama = _pairs()
    response = json.dumps(
        {
            "pins": [
                {
                    "pin_no": 99,
                    "name": "PA0",
                    "type": "power",
                    "functions": [],
                    "dir": "P",
                }
            ]
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(
        canonical_json({"prompt": child.prompt, "response": response})
    ).hexdigest()
    child = child.model_copy(
        update={
            "response": response,
            "provenance": child.provenance.model_copy(
                update={"content_sha256": digest}
            ),
        }
    )
    llama = {"instruction": child.prompt, "input": "", "output": response}

    with pytest.raises(ValueError, match="not a subset"):
        registration.validate_chunk(child, parent, llama, cutoff_len=4096)


def test_chunk_event_identity_includes_dataset_manifest():
    _parent, child, _llama = _pairs()

    assert registration._chunk_event_id(
        child, "a" * 64
    ) != registration._chunk_event_id(child, "b" * 64)


def test_parent_duplicate_does_not_invalidate_deduplicated_chunk():
    parent, child, llama = _pairs()
    value = json.loads(parent.response)
    value["pins"].append(dict(value["pins"][0]))
    parent = parent.model_copy(
        update={
            "response": json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            )
        }
    )

    proof = registration.validate_chunk(
        child,
        parent,
        llama,
        cutoff_len=4096,
    )

    assert proof["parent_pair_id"] == parent.pair_id
