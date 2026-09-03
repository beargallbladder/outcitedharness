from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from harness.electronics.claims import canonical_json
from harness.electronics.factory_control import (
    ElectronicsFactoryState,
    FactoryStateError,
)


NOW = datetime(2026, 9, 2, 7, 0, tzinfo=timezone.utc)


def _write_pdf(path: Path, body: bytes = b"datasheet") -> None:
    path.write_bytes(
        b"%PDF-1.7\n"
        + body
        + b"\n"
        + (b"x" * 128)
        + b"\n%%EOF\n"
    )


def _queue(path: Path, count: int = 5) -> dict:
    core = {
        "schema": "harness.electronics-structural-local-work.v1",
        "policy": {"purpose": "test"},
        "sources": {},
        "counts": {"selected": count},
        "work": [
            {
                "work_id": f"work-{index}",
                "document_sha256": f"{index + 1:064x}",
                "page_1based": 1,
                "capability": "pin_semantics",
            }
            for index in range(count)
        ],
    }
    value = {
        "created_at": NOW.isoformat(),
        **core,
        "evidence_sha256": hashlib.sha256(canonical_json(core)).hexdigest(),
    }
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return value


def _complete_output(lease) -> str:
    lease.output_directory.mkdir(parents=True)
    manifest = {
        "schema": "harness.electronics-structural-local-extraction.v1",
        "selection": {
            "offset": lease.offset,
            "limit": lease.item_count,
            "work_items": lease.item_count,
        },
        "sources": {
            "structural_queue_sha256": lease.queue_sha256,
        },
    }
    path = lease.output_directory / "manifest.json"
    path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_incremental_intake_waits_hashes_deduplicates_and_quarantines(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    _write_pdf(downloads / "a.pdf")
    _write_pdf(downloads / "copy.pdf")
    (downloads / "partial.pdf").write_bytes(
        b"%PDF-1.7\n" + (b"x" * 128)
    )
    state = ElectronicsFactoryState(tmp_path / "state")

    first = state.discover_pdfs(
        [downloads],
        stability_seconds=60,
        now=NOW,
    )
    assert first.new_observations == 3
    assert first.waiting_for_stability == 3
    assert first.ready == 0

    second = state.discover_pdfs(
        [downloads],
        stability_seconds=60,
        now=NOW + timedelta(seconds=61),
    )
    assert second.ready == 1
    assert second.duplicates == 1
    assert second.quarantined == 1
    assert state.status()["sources"] == {
        "duplicate": 1,
        "quarantined": 1,
        "ready": 1,
    }

    snapshot_path = tmp_path / "ready.json"
    snapshot = state.seal_ready_source_snapshot(snapshot_path)
    assert snapshot["counts"] == {"documents": 1}
    assert snapshot["documents"][0]["source_path"].endswith("a.pdf")
    assert snapshot_path.stat().st_mode & 0o222 == 0

    cohort_path = tmp_path / "cohort.json"
    cohort = state.seal_unassigned_source_snapshot(
        cohort_path,
        cohort_id="download-wave-1",
        now=NOW + timedelta(seconds=62),
    )
    assert cohort is not None
    assert cohort["counts"] == {"documents": 1}
    assert (
        state.seal_unassigned_source_snapshot(
            tmp_path / "unused.json",
            cohort_id="download-wave-2",
        )
        is None
    )
    assert state.status()["unassigned_ready_sources"] == 0


def test_changed_download_creates_new_observation_and_supersedes_old(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    path = downloads / "part.pdf"
    _write_pdf(path, b"first")
    state = ElectronicsFactoryState(tmp_path / "state")
    state.discover_pdfs(
        [downloads],
        stability_seconds=60,
        now=NOW,
    )
    _write_pdf(path, b"second-and-larger")
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1))

    report = state.discover_pdfs(
        [downloads],
        stability_seconds=60,
        now=NOW + timedelta(seconds=10),
    )
    assert report.new_observations == 1
    assert state.status()["sources"] == {
        "observed": 1,
        "superseded": 1,
    }


def test_sealed_corpus_seed_prevents_rehashing_existing_downloads(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    existing = downloads / "existing.pdf"
    _write_pdf(existing)
    digest = hashlib.sha256(existing.read_bytes()).hexdigest()
    registry_core = {
        "schema": "harness.electronics-corpus-registry.v1",
        "policy": {},
        "sources": {
            "pdf_root": str(downloads),
            "ground_truth_root": str(tmp_path),
            "validated_root": str(tmp_path),
        },
        "counts": {
            "pdf_files": 1,
            "unique_pdf_sha256": 1,
            "duplicate_pdf_files": 0,
        },
        "assets": [],
        "documents": [
            {
                "document_sha256": digest,
                "byte_size": existing.stat().st_size,
                "paths": [existing.name],
                "stems": [existing.stem],
                "vendors": [],
                "record_ids": [],
                "ground_truth": [],
                "published_pinouts": [],
                "asset_memberships": {},
            }
        ],
        "orphans": {"ground_truth": [], "published_pinouts": []},
    }
    registry = {
        "created_at": NOW.isoformat(),
        **registry_core,
        "evidence_sha256": hashlib.sha256(
            canonical_json(registry_core)
        ).hexdigest(),
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry))
    state = ElectronicsFactoryState(tmp_path / "state")

    assert state.seed_corpus_registry(registry_path, now=NOW) == {
        "inserted": 1,
        "existing": 0,
        "missing": 0,
        "unsafe": 0,
    }
    report = state.discover_pdfs(
        [downloads],
        stability_seconds=120,
        now=NOW + timedelta(seconds=1),
    )
    assert report.new_observations == 0
    assert report.unchanged == 1
    assert state.status()["sources"] == {"ready": 1}
    assert state.status()["unassigned_ready_sources"] == 0

    _write_pdf(downloads / "new.pdf", b"new")
    report = state.discover_pdfs(
        [downloads],
        stability_seconds=120,
        now=NOW + timedelta(seconds=2),
    )
    assert report.new_observations == 1
    assert report.waiting_for_stability == 1


def test_queue_registration_leases_retry_recovery_and_completion(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "queue.json"
    _queue(queue_path, count=5)
    state = ElectronicsFactoryState(tmp_path / "state")
    chunks = state.register_queue(
        queue_path,
        tmp_path / "outputs",
        chunk_size=2,
        max_attempts=2,
        now=NOW,
    )
    assert len(chunks) == 3
    assert (
        state.register_queue(
            queue_path,
            tmp_path / "outputs",
            chunk_size=2,
            max_attempts=2,
            now=NOW,
        )
        == chunks
    )

    first = state.claim_chunk("dgx2", lease_seconds=60, now=NOW)
    second = state.claim_chunk("asus1", lease_seconds=60, now=NOW)
    assert first is not None and first.offset == 0 and first.item_count == 2
    assert second is not None and second.offset == 2
    assert state.fail_chunk(
        first,
        "temporary endpoint failure",
        retry_delay_seconds=0,
        now=NOW,
    ) == "queued"

    retried = state.claim_chunk("dgx3", lease_seconds=60, now=NOW)
    assert retried is not None
    assert retried.chunk_id == first.chunk_id
    assert retried.attempt == 2
    expected_manifest = _complete_output(retried)
    assert (
        state.complete_chunk(retried, now=NOW + timedelta(seconds=1))
        == expected_manifest
    )

    recovered = state.recover_expired(now=NOW + timedelta(seconds=61))
    assert recovered == [second.chunk_id]
    status = state.status()
    assert status["chunks"] == {
        "completed": 1,
        "queued": 2,
    }
    assert status["active_leases"] == []


def test_stale_lease_cannot_complete_another_attempt(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    _queue(queue_path, count=1)
    state = ElectronicsFactoryState(tmp_path / "state")
    state.register_queue(
        queue_path,
        tmp_path / "outputs",
        chunk_size=1,
        now=NOW,
    )
    stale = state.claim_chunk("dgx2", lease_seconds=30, now=NOW)
    assert stale is not None
    state.recover_expired(now=NOW + timedelta(seconds=31))
    current = state.claim_chunk(
        "asus1",
        lease_seconds=30,
        now=NOW + timedelta(seconds=31),
    )
    assert current is not None
    _complete_output(stale)

    with pytest.raises(FactoryStateError, match="stale"):
        state.complete_chunk(stale, now=NOW + timedelta(seconds=32))


def test_queue_registration_can_resume_after_prior_manual_offsets(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "queue.json"
    _queue(queue_path, count=7)
    state = ElectronicsFactoryState(tmp_path / "state")

    chunks = state.register_queue(
        queue_path,
        tmp_path / "outputs",
        chunk_size=2,
        start_offset=3,
    )
    assert len(chunks) == 2
    first = state.claim_chunk("dgx2")
    second = state.claim_chunk("asus1")
    assert first is not None and (first.offset, first.item_count) == (3, 2)
    assert second is not None and (second.offset, second.item_count) == (5, 2)


def test_chunk_output_must_match_queue_and_selection(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    _queue(queue_path, count=1)
    state = ElectronicsFactoryState(tmp_path / "state")
    state.register_queue(queue_path, tmp_path / "outputs", chunk_size=1)
    lease = state.claim_chunk("dgx2")
    assert lease is not None
    lease.output_directory.mkdir(parents=True)
    (lease.output_directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "harness.electronics-structural-local-extraction.v1",
                "selection": {"offset": 99, "work_items": 1},
                "sources": {
                    "structural_queue_sha256": lease.queue_sha256,
                },
            }
        )
    )

    with pytest.raises(FactoryStateError, match="selection"):
        state.complete_chunk(lease)


def test_factory_events_are_append_only(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    _queue(queue_path, count=1)
    state = ElectronicsFactoryState(tmp_path / "state")
    state.register_queue(queue_path, tmp_path / "outputs", chunk_size=1)

    with state.connect() as connection:
        event_id = connection.execute(
            "SELECT event_id FROM factory_events LIMIT 1"
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE factory_events SET event_type = 'changed' "
                "WHERE event_id = ?",
                (event_id,),
            )


def test_frontier_run_registration_and_status_are_idempotent(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    manifest = {
        "schema": "harness.electronics-frontier-batch.v1",
        "evidence_sha256": "a" * 64,
        "counts": {"requests": 17},
        "pricing": {"estimated_maximum_usd": 2.5},
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "submission-intent.json").write_text("{}")
    state = ElectronicsFactoryState(tmp_path / "state")

    assert state.register_frontier_run(
        run_id="semantic-v6",
        prepared_bundle=prepared,
        submission_state=submission,
        lifecycle_root=tmp_path / "lifecycle",
        now=NOW,
    )
    assert not state.register_frontier_run(
        run_id="semantic-v6",
        prepared_bundle=prepared,
        submission_state=submission,
        lifecycle_root=tmp_path / "lifecycle",
        now=NOW,
    )
    status_payload = {
        "batches": [{"id": "msgbatch_1", "processing_status": "in_progress"}]
    }
    state.record_frontier_status(
        "semantic-v6",
        status_payload,
        now=NOW + timedelta(minutes=1),
    )

    assert state.status()["frontier_runs"][0] == {
        "run_id": "semantic-v6",
        "status": "processing",
        "request_count": 17,
        "estimated_maximum_usd": 2.5,
        "updated_at": (NOW + timedelta(minutes=1)).isoformat(
            timespec="microseconds"
        ),
        "error": None,
    }
