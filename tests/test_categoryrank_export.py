from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_categoryrank.py"
SPEC = importlib.util.spec_from_file_location("export_categoryrank", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
export_categoryrank = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_categoryrank)


def test_query_is_bounded_stitched_and_uses_canonical_tensor_grain() -> None:
    query = export_categoryrank._query("electronics", "2026-W34")

    assert "cm.time_window <= '2026-W34'" in query
    assert "cr_brand_canonical_successor_map" in query
    assert "sm.grounding_status LIKE 'verified_%'" in query
    assert "b.primary_vertical = 'electronics'" in query
    assert "b.verification_tier IS DISTINCT FROM 'flagged_noise'" in query
    assert "cm.kim_category_id AS kim_slug" in query
    assert "'model_id', model_id" in query
    assert "cm.category," not in query
    assert "COPY (" not in query
    for sentinel in ("__unknown__", "'n'", "-unknown-", "'unknown'"):
        assert sentinel in query


def test_query_rejects_non_iso_week() -> None:
    with pytest.raises(ValueError, match="ISO"):
        export_categoryrank._query("electronics", "2026-W34'; DROP TABLE brands; --")


def test_export_normalizes_stream_and_writes_retrieval_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "categoryrank.jsonl"
    row = {
        "brand_domain": "st.com",
        "kim_slug": "microcontrollers",
        "model_id": "anthropic",
        "time_window": "2026-W34",
        "n_mentions": 4,
        "avg_strength": 88.5,
        "avg_rank": 2.5,
    }

    monkeypatch.setattr(
        export_categoryrank,
        "_connection_environment",
        lambda _postgres_id: {},
    )

    def fake_psql_to_path(
        _environment: dict[str, str], query: str, path: Path
    ) -> subprocess.CompletedProcess[str]:
        assert "COPY (" not in query
        path.write_text(json.dumps(row) + "\n")
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(export_categoryrank, "_psql_to_path", fake_psql_to_path)
    export_categoryrank.export(
        "postgres-1",
        destination,
        week="2026-W34",
        layer="electronics",
    )

    assert json.loads(destination.read_text()) == row
    manifest = json.loads(
        destination.with_suffix(".jsonl.manifest.json").read_text()
    )
    assert manifest["artifact"]["rows"] == 1
    assert manifest["filters"]["data_use"] == "retrieval_only"
    assert manifest["filters"]["through_week"] == "2026-W34"
    assert manifest["filters"]["identity_policy"] == "verified-successor-one-hop"
