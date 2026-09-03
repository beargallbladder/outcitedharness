import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _script(name: str):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge = _script("merge_datasheet_local_bundles").merge


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(root: Path, work_ids: list[str], offset: int) -> Path:
    root.mkdir()
    artifacts = {}
    for name in ("local-results.jsonl", "pillar-evidence.jsonl"):
        path = root / name
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "work_id": work_id,
                        **(
                            {"result": {"pins": []}}
                            if name == "local-results.jsonl"
                            else {"page": {"tables": []}}
                        ),
                    },
                    sort_keys=True,
                )
                + "\n"
                for work_id in work_ids
            )
        )
        artifacts[name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest = {
        "schema": "harness.electronics-structural-local-extraction.v1",
        "artifacts": artifacts,
        "counts": {"result:focused_local_vision": len(work_ids)},
        "model": {
            "provider": "local",
            "model": "candidate",
            "base_url": f"http://node-{offset}/v1",
        },
        "selection": {
            "offset": offset,
            "limit": len(work_ids),
            "work_items": len(work_ids),
        },
        "sources": {
            "structural_queue_sha256": "a" * 64,
            "structural_queue_evidence_sha256": "b" * 64,
            "page_evidence_sha256": "c" * 64,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_merge_combines_disjoint_extraction_shards(tmp_path: Path) -> None:
    first = _bundle(tmp_path / "first", ["work-a", "work-b"], 0)
    second = _bundle(tmp_path / "second", ["work-c"], 2)

    manifest = merge(
        [first, second],
        tmp_path / "merged",
        expected_items=3,
    )

    assert manifest["selection"] == {
        "work_items": 3,
        "results": 3,
        "shards": 2,
    }
    assert manifest["model"]["base_urls"] == [
        "http://node-0/v1",
        "http://node-2/v1",
    ]
    assert manifest["artifacts"]["local-results.jsonl"]["bytes"] > 0


def test_merge_rejects_duplicate_work_ids(tmp_path: Path) -> None:
    first = _bundle(tmp_path / "first", ["work-a"], 0)
    second = _bundle(tmp_path / "second", ["work-a"], 1)

    with pytest.raises(ValueError, match="duplicate"):
        merge([first, second], tmp_path / "merged", expected_items=2)
