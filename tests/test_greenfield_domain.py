from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.gci.models import GCIHit
from harness.greenfield.discovery import (
    CATEGORY_LIMITS,
    MAX_DISCOVERY_HITS,
    SELECTED_LIMITS,
    compile_discovery_packet,
    gather_discovery,
)
from harness.greenfield.manifest import ManifestDriftError
from harness.greenfield.planner import (
    build_milestone_plan,
    build_product_spec,
    validate_dependency,
)
from harness.greenfield.service import GreenfieldService
from harness.storage.db import Store


def _hit(index: int, path: str | None = None) -> GCIHit:
    return GCIHit(
        repo_id=f"repo-{index % 3}",
        source_host="m5",
        repo_root=f"/repos/repo-{index % 3}",
        revision=f"rev-{index}",
        state_hash=f"state-{index}",
        path=path or f"src/pattern_{index}.py",
        symbol=f"pattern_{index}",
        symbol_type="function",
        start_line=10,
        end_line=20,
        score=0.9 - index / 100,
        match_type="semantic",
        text=f"def pattern_{index}(): return 'bounded advisory excerpt'",
    )


def _planned(tmp_path: Path):
    calls = []

    def search(query: str, *, limit: int, mode: str):
        calls.append((query, limit, mode))
        return [_hit(index) for index in range(limit)]

    discovery = gather_discovery(
        SimpleNamespace(gci_enabled=False),
        "Build a FastAPI service for electronics-family comparisons",
        "python",
        search=search,
    )
    spec = build_product_spec(
        name="electronics-compare",
        stack="python",
        intent="Build a FastAPI service for electronics-family comparisons",
        discovery=discovery,
    )
    plan = build_milestone_plan(spec)
    return calls, discovery, spec, plan


def test_discovery_is_bounded_provenance_bearing_and_advisory(tmp_path: Path):
    calls, discovery, spec, _plan = _planned(tmp_path)
    assert [limit for _query, limit, _mode in calls] == list(CATEGORY_LIMITS.values())
    assert len(discovery.repo_hits) <= MAX_DISCOVERY_HITS
    assert len(discovery.selected_patterns) <= sum(SELECTED_LIMITS.values())
    assert discovery.selected_patterns
    assert all(pattern.provenance.startswith("gci://") for pattern in discovery.selected_patterns)
    packet = compile_discovery_packet(discovery)
    assert "ADVISORY ONLY" in packet
    assert "do not" in packet.lower()
    assert all(
        pattern.provenance in packet
        for pattern in discovery.selected_patterns
    )
    assert spec.discovery_patterns == tuple(discovery.selected_patterns)


def test_planning_persists_without_creating_destination_or_workspace(tmp_path: Path):
    _calls, discovery, spec, plan = _planned(tmp_path)
    destination = tmp_path / "published"
    service = GreenfieldService(Store(tmp_path / "harness.sqlite"))
    run = service.create(
        intent=spec.purpose,
        name=spec.project_name,
        stack=spec.stack,
        destination=destination,
        discovery=discovery,
        spec=spec,
        plan=plan,
    )
    assert run.status == "awaiting_approval"
    assert run.workspace_root is None
    assert not destination.exists()
    assert [row.state for row in run.milestones] == ["pending", "pending", "pending"]
    assert service.events(run.run_id)[0]["kind"] == "planned"


def test_one_approval_freezes_manifest_and_drift_blocks(tmp_path: Path):
    _calls, discovery, spec, plan = _planned(tmp_path)
    store = Store(tmp_path / "harness.sqlite")
    service = GreenfieldService(store)
    run = service.create(
        intent=spec.purpose,
        name=spec.project_name,
        stack=spec.stack,
        destination=tmp_path / "published",
        discovery=discovery,
        spec=spec,
        plan=plan,
    )
    approved = service.approve(run.run_id)
    assert approved.status == "provisioning"
    assert approved.manifest is not None
    assert approved.approved_at
    with store.connect() as conn:
        raw = json.loads(
            conn.execute(
                "SELECT spec_json FROM greenfield_runs WHERE run_id=?",
                (run.run_id,),
            ).fetchone()[0]
        )
        raw["security_constraints"] = ["silently changed"]
        conn.execute(
            "UPDATE greenfield_runs SET spec_json=? WHERE run_id=?",
            (json.dumps(raw), run.run_id),
        )
    with pytest.raises(ManifestDriftError):
        service.assert_approved(run.run_id)
    assert service.get(run.run_id).status == "blocked"


def test_dependency_policy_rejects_urls_flags_and_shell():
    for value in (
        "https://example.com/pkg.tgz",
        "../local",
        "--pre",
        "fastapi;curl bad",
        "git+ssh://host/repo",
    ):
        with pytest.raises(ValueError):
            validate_dependency(value)
    assert validate_dependency("fastapi>=0.100") == "fastapi>=0.100"
