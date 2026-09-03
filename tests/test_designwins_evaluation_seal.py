from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "seal_designwins_evaluation.py"
SPEC = importlib.util.spec_from_file_location("seal_designwins_evaluation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sealer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sealer)


def _evaluation(samples: int) -> dict:
    details = [
        {
            "index": index,
            "part": f"part-{index}",
            "family": "family",
            "score": {
                "valid_json": True,
                "exact": False,
                "leaf_precision": 0.5,
                "leaf_recall": 0.5,
                "leaf_f1": 0.5,
            },
            "generated_tokens": 10,
            "hit_generation_limit": False,
        }
        for index in range(samples)
    ]
    return {
        "model": "/model",
        "adapter": None,
        "dataset": "/dataset",
        "summary": {},
        "details": details,
    }


def test_sealer_binds_complete_evaluation_to_inputs(tmp_path: Path):
    source = tmp_path / "raw.json"
    output = tmp_path / "sealed.json"
    dataset = tmp_path / "test.json"
    model_manifest = tmp_path / "model.manifest.json"
    scorer = tmp_path / "scorer.py"
    source.write_text(json.dumps(_evaluation(2)))
    dataset.write_text("[]")
    model_manifest.write_text('{"files":[]}')
    scorer.write_text("pass\n")

    identity = sealer.seal(
        source,
        output,
        dataset=dataset,
        model_manifest=model_manifest,
        scorer=scorer,
        runtime_image_id="sha256:" + "a" * 64,
        adapter_manifest=None,
        max_samples=2,
        cutoff_len=4096,
        max_new_tokens=8192,
        batch_size=8,
        generation_slack_tokens=256,
    )

    assert identity["schema"] == "harness.designwins-evaluation-identity.v1"
    assert len(identity["core_sha256"]) == 64
    assert json.loads(output.read_text())["identity"] == identity


def test_sealer_rejects_partial_evaluation(tmp_path: Path):
    source = tmp_path / "raw.json"
    source.write_text(json.dumps(_evaluation(1)))
    files = [tmp_path / name for name in ("dataset", "manifest", "scorer")]
    for path in files:
        path.write_text("x")

    with pytest.raises(ValueError, match="exactly 2"):
        sealer.seal(
            source,
            tmp_path / "sealed.json",
            dataset=files[0],
            model_manifest=files[1],
            scorer=files[2],
            runtime_image_id="sha256:" + "a" * 64,
            adapter_manifest=None,
            max_samples=2,
            cutoff_len=4096,
            max_new_tokens=8192,
            batch_size=8,
            generation_slack_tokens=256,
        )
