from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "chunk_designwins_text.py"
SPEC = importlib.util.spec_from_file_location("chunk_designwins_text", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
chunker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chunker)


class FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return messages[0]["content"]

    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(max(1, len(text) // 4)))}


def _record() -> dict:
    return {
        "pair_id": "part-1",
        "prompt": (
            "Extract pins."
            "\n\nPIN TABLE TEXT:\n"
            "1\nPA0\nI/O\nGPIO zero and UART TX.\n"
            "2\nPA1\nI/O\nGPIO one and UART RX.\n"
        ),
        "response": json.dumps(
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
            }
        ),
        "provenance": {
            "source_record_id": "part-1:text",
            "content_sha256": "a" * 64,
            "lineage_id": "designwins:part-1",
        },
        "metadata": {"part": "part-1", "modality": "text"},
        "data_use": "training",
        "label": "pinout",
    }


def test_chunker_preserves_lineage_and_fits_cutoff():
    chunks, rejections = chunker.chunk_records(
        [_record()],
        tokenizer=FakeTokenizer(),
        cutoff_len=512,
        max_pins_per_chunk=16,
    )

    assert rejections == []
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["sequence_tokens"] <= 512
    assert chunks[0]["metadata"]["parent_pair_id"] == "part-1"
    assert chunks[0]["provenance"]["lineage_id"] == "designwins:part-1"
    assert chunks[0]["provenance"]["content_sha256"] != "a" * 64
    assert [pin["name"] for pin in json.loads(chunks[0]["response"])["pins"]] == [
        "PA0",
        "PA1",
    ]


def test_chunker_rejects_labels_not_grounded_in_source():
    record = _record()
    value = json.loads(record["response"])
    value["pins"][1]["name"] = "MISSING"
    record["response"] = json.dumps(value)

    chunks, rejections = chunker.chunk_records(
        [record],
        tokenizer=FakeTokenizer(),
        cutoff_len=512,
        max_pins_per_chunk=16,
    )

    assert len(chunks) == 1
    assert [row["pin_name"] for row in rejections] == ["MISSING"]
    assert json.loads(chunks[0]["response"])["pins"][0]["name"] == "PA0"


def test_chunker_rejects_single_pin_block_over_cutoff():
    chunks, rejections = chunker.chunk_records(
        [_record()],
        tokenizer=FakeTokenizer(),
        cutoff_len=64,
        max_pins_per_chunk=1,
    )

    assert chunks == []
    assert {row["reason"] for row in rejections} == {
        "single grounded pin block exceeds cutoff"
    }


def test_repeated_physical_pin_name_stays_grounded_when_chunks_split():
    record = _record()
    record["prompt"] = (
        "Extract pins."
        "\n\nPIN TABLE TEXT:\n"
        "1\nVSS\nP\nGround for bank one.\n"
        "17\nVSS\nP\nGround for bank two.\n"
    )
    record["response"] = json.dumps(
        {
            "pins": [
                {
                    "pin_no": 1,
                    "name": "VSS",
                    "type": "power",
                    "functions": ["Ground for bank one"],
                    "dir": "P",
                },
                {
                    "pin_no": 17,
                    "name": "VSS",
                    "type": "power",
                    "functions": ["Ground for bank two"],
                    "dir": "P",
                },
            ]
        }
    )

    chunks, rejections = chunker.chunk_records(
        [record],
        tokenizer=FakeTokenizer(),
        cutoff_len=512,
        max_pins_per_chunk=1,
    )

    assert rejections == []
    assert len(chunks) == 2
    for chunk in chunks:
        pin = json.loads(chunk["response"])["pins"][0]
        source = chunk["prompt"].split(chunker.MARKER, 1)[1]
        assert pin["name"] in source


def test_duplicate_physical_pin_label_is_quarantined():
    record = _record()
    value = json.loads(record["response"])
    value["pins"].append(dict(value["pins"][0]))
    record["response"] = json.dumps(value)

    chunks, rejections = chunker.chunk_records(
        [record],
        tokenizer=FakeTokenizer(),
        cutoff_len=512,
        max_pins_per_chunk=16,
    )

    output_pins = [
        pin
        for chunk in chunks
        for pin in json.loads(chunk["response"])["pins"]
    ]
    assert len(output_pins) == 2
    assert [row["reason"] for row in rejections] == [
        "duplicate physical pin label"
    ]
