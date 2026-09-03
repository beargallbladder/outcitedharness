from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_pinout_vision.py"
SPEC = importlib.util.spec_from_file_location("pinout_vision_evaluation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_response_parser_and_identity_metric_are_fail_closed() -> None:
    value = evaluation._parse_response(
        '```json\n{"pins":[{"pin_no":1,"name":"PA0"}]}\n```'
    )
    pins = evaluation._pin_map(value)
    metric = evaluation._set_metric(set(pins), {("1", "PA0"), ("2", "GND")})

    assert metric["true_positive"] == 1
    assert metric["precision"] == 1.0
    assert metric["recall"] == 0.5
    assert metric["exact"] is False


def test_family_key_collapses_automotive_suffix() -> None:
    assert evaluation._family_key("ti_mspm0g3507-q1") == "ti_mspm0g3507"
    assert evaluation._family_key("st_stm32f469zi") == "st_stm32f469zi"


def test_freeze_cohort_selects_and_hash_binds_each_pdf_lineage(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / evaluation.DATASET_RELATIVE
    (dataset / "canonical").mkdir(parents=True)
    (dataset / "images").mkdir()
    images = []
    for index in range(2):
        image = dataset / "images" / f"{index}.png"
        image.write_bytes(f"image-{index}".encode())
        images.append(
            (
                image.relative_to(dataset).as_posix(),
                evaluation._sha256(image),
            )
        )
    rows = []
    for index, lineage in enumerate(("pdf:a", "pdf:b")):
        rows.append(
            {
                "example_id": f"example-{index}",
                "split": "test",
                "lineage_id": lineage,
                "record_id": f"vendor_part_{index}",
                "prompt": "Extract pins",
                "response": '{"pins":[{"pin_no":1,"name":"VDD"}]}',
                "images": [value[0] for value in images],
                "image_sha256": [value[1] for value in images],
            }
        )
    train_path = dataset / "canonical" / "train.jsonl"
    train_path.write_text("")
    test_path = dataset / "canonical" / "test.jsonl"
    test_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    core = {
        "schema": "harness.pinout-vision-row-dataset.v1",
        "authorization": {"training_authorized": True},
        "artifacts": {
            "canonical/train.jsonl": {
                "sha256": evaluation._sha256(train_path),
                "bytes": train_path.stat().st_size,
            },
            "canonical/test.jsonl": {
                "sha256": evaluation._sha256(test_path),
                "bytes": test_path.stat().st_size,
            }
        },
    }
    core["evidence_sha256"] = hashlib.sha256(
        evaluation._canonical(core)
    ).hexdigest()
    (dataset / "manifest.json").write_text(
        json.dumps({"created_at": "2026-09-01T00:00:00Z", **core})
    )

    cohort = evaluation.freeze_cohort(
        root=tmp_path,
        examples_per_lineage=1,
    )

    assert cohort["selection"]["lineages"] == 2
    assert cohort["selection"]["examples"] == 2
    assert {row["lineage_id"] for row in cohort["examples"]} == {
        "pdf:a",
        "pdf:b",
    }
