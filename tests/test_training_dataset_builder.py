from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_training_datasets.py"
SPEC = importlib.util.spec_from_file_location("build_training_datasets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
build_training_datasets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_training_datasets)
build_designwins = build_training_datasets.build_designwins


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_designwins_builder_enforces_quality_and_provenance_audits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "training_data"
    images = source / "images"
    source.mkdir()
    rows = []
    vision_rows = []
    target = json.dumps({"pins": [{"pin_no": 1, "name": "VCC"}]})
    for part in ("part-good", "part-suspect", "part-failed"):
        part_images = images / part
        part_images.mkdir(parents=True)
        image = part_images / "page_1.png"
        image.write_bytes(part.encode())
        rows.append({"part": part, "prompt": "Extract pins", "target": target})
        vision_rows.append(
            {
                "part": part,
                "prompt": "Extract pins",
                "target": target,
                "images": [str(image)],
            }
        )
    _write_jsonl(source / "text_pairs.jsonl", rows)
    _write_jsonl(source / "vision_pairs.jsonl", vision_rows)
    (tmp_path / "gt_audit.json").write_text(
        json.dumps(
            {
                "part-good": {"ok": True, "problems": []},
                "part-suspect": {"ok": True, "problems": []},
                "part-failed": {"ok": False, "problems": ["duplicate pins"]},
            }
        )
    )
    (tmp_path / "gt_provenance_audit.json").write_text(
        json.dumps({"suspect": [{"part": "part-suspect"}]})
    )

    destination = tmp_path / "dataset"
    manifest = build_designwins(
        source,
        destination,
        audit_root=tmp_path,
    )

    assert manifest["eligibility"] == {
        "audit_required": True,
        "provenance_suspect_included": False,
        "eligible_parts": 1,
    }
    assert sum(manifest["counts"]["text"].values()) == 1
    assert sum(manifest["counts"]["vision"].values()) == 1
    assert manifest["counts"]["quarantine"] == {"text": 2, "vision": 2}
    assert (destination / "images" / "part-good" / "page_1.png").is_file()
    assert not (destination / "images" / "part-suspect").exists()
    assert not (destination / "images" / "part-failed").exists()
    assert "gt_audit.json" in manifest["sources"]
    assert "gt_provenance_audit.json" in manifest["sources"]
    dataset_info = json.loads(
        (destination / "llamafactory" / "dataset_info.json").read_text()
    )
    vision_tags = dataset_info["designwins_vision_train"]["tags"]
    assert vision_tags["user_tag"] == "user"
    assert vision_tags["assistant_tag"] == "assistant"
    vision_rows = []
    for split in ("train", "validation", "test"):
        vision_rows.extend(
            json.loads(
                (
                    destination
                    / "llamafactory"
                    / f"designwins_vision_{split}.json"
                ).read_text()
            )
        )
    assert vision_rows[0]["messages"][0]["content"].count("<image>") == 1
