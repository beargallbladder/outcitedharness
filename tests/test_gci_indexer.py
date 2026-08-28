from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from harness.gci.encoder import (
    EncoderPolicyError,
    EncoderResponse,
    MAX_ENCODER_BATCH,
    StrictEncoder,
    validate_encoder_url,
)
from harness.gci.indexer import GCIIndexer, IndexWorker, snapshot_state_hash
from harness.gci.models import CodeDocument, RepoSnapshot
from harness.gci.storage import GCIStorageError, GCIStore


def _snapshot(
    documents: dict[str, str],
    *,
    previous: dict[str, str] | None = None,
    head: str = "abc",
) -> RepoSnapshot:
    hashes = {path: hashlib.sha256(text.encode()).hexdigest() for path, text in documents.items()}
    changed = []
    for path, text in documents.items():
        if (previous or {}).get(path) != hashes[path]:
            changed.append(
                CodeDocument(
                    path=path,
                    content=text,
                    content_hash=hashes[path],
                    language="python",
                )
            )
    deleted = tuple(sorted(set(previous or {}) - set(hashes)))
    return RepoSnapshot(
        repo_id="repo",
        source_host="m5",
        repo_root="/repos/example",
        remote=None,
        branch="main",
        head=head,
        dirty=True,
        state_hash=snapshot_state_hash(head, hashes),
        file_hashes=hashes,
        documents=tuple(changed),
        deleted=deleted,
    )


def test_encoder_policy_refuses_nonproduction_routes():
    assert (
        validate_encoder_url("http://localhost:8800/v1/embeddings")
        == "http://127.0.0.1:8800/v1/embeddings"
    )
    for url in (
        "http://127.0.0.1:8800/semantic-search",
        "http://127.0.0.1:8800/encode",
        "http://100.81.201.24:8800/v1/embeddings",
        "https://127.0.0.1:8800/v1/embeddings",
    ):
        with pytest.raises(EncoderPolicyError):
            StrictEncoder(url)


def test_strict_encoder_calls_only_v1_embeddings():
    observed = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {"data": [{"index": 0, "embedding": [0.0] * 1024}]}
            ).encode()

    class Opener:
        def open(self, request, timeout):
            observed["url"] = request.full_url
            observed["payload"] = json.loads(request.data)
            observed["timeout"] = timeout
            return Response()

    encoder = StrictEncoder(timeout=3)
    encoder._opener = Opener()
    response = encoder.embed(["safe query"])
    assert observed["url"] == "http://127.0.0.1:8800/v1/embeddings"
    assert observed["payload"]["input"] == ["safe query"]
    assert len(response.vectors[0]) == 1024


def test_indexer_is_sequential_and_batches_at_most_sixteen(tmp_path: Path):
    store = GCIStore(tmp_path / "gci.sqlite")
    active = 0
    maximum_active = 0
    sizes: list[int] = []
    lock = threading.Lock()

    def embed(texts: list[str]) -> EncoderResponse:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            sizes.append(len(texts))
        time.sleep(0.001)
        with lock:
            active -= 1
        return EncoderResponse(tuple((1.0, 0.0) for _ in texts), 1.0)

    content = "def huge_function(values):\n" + "\n".join(
        f"    item_{index} = values[{index} % 2]" for index in range(400)
    )
    stats = GCIIndexer(store, embed=embed).index(
        _snapshot({"src/huge.py": content})
    )
    assert maximum_active == 1
    assert sizes
    assert max(sizes) <= MAX_ENCODER_BATCH
    assert stats.encoder_calls == len(sizes)
    assert stats.embedded > MAX_ENCODER_BATCH


def test_incremental_refresh_embeds_only_changed_files_and_deletes_stale(tmp_path: Path):
    store = GCIStore(tmp_path / "gci.sqlite")
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> EncoderResponse:
        calls.append(list(texts))
        return EncoderResponse(tuple((1.0, 0.0) for _ in texts), 1.0)

    indexer = GCIIndexer(store, embed=embed)
    first = _snapshot(
        {
            "one.py": "def one():\n    return 'one with enough content for indexing'\n",
            "two.py": "def two():\n    return 'two with enough content for indexing'\n",
        }
    )
    first_stats = indexer.index(first)
    previous = first.file_hashes
    calls.clear()
    second = _snapshot(
        {"one.py": "def one():\n    return 'changed implementation for one'\n"},
        previous=previous,
        head="def",
    )
    second_stats = indexer.index(second)
    assert first_stats.changed == 2
    assert second_stats.changed == 1
    assert second_stats.deleted == 1
    assert not store.exact_search("two with enough")
    assert store.exact_search("changed implementation")
    assert all("two with enough" not in text for batch in calls for text in batch)


def test_indexer_rejects_manifest_drift_before_embedding(tmp_path: Path):
    store = GCIStore(tmp_path / "gci.sqlite")
    called = False

    def embed(texts: list[str]) -> EncoderResponse:
        nonlocal called
        called = True
        return EncoderResponse(tuple((1.0,) for _ in texts), 1.0)

    snapshot = _snapshot({"good.py": "def good():\n    return 'safe content here'\n"})
    invalid = RepoSnapshot(**{**vars(snapshot), "state_hash": "wrong"})
    with pytest.raises(GCIStorageError, match="state hash"):
        GCIIndexer(store, embed=embed).index(invalid)
    assert not called


def test_worker_pause_and_durable_job_state(tmp_path: Path):
    store = GCIStore(tmp_path / "gci.sqlite")
    store.set_paused(True)

    def embed(texts: list[str]) -> EncoderResponse:
        return EncoderResponse(tuple((1.0,) for _ in texts), 1.0)

    worker = IndexWorker(
        GCIIndexer(store, embed=embed),
        pause_poll_seconds=0.01,
    )
    worker.start()
    job_id = worker.submit(
        _snapshot({"job.py": "def job():\n    return 'queued while paused'\n"})
    )
    time.sleep(0.03)
    assert store.job(job_id)["state"] == "queued"
    store.set_paused(False)
    deadline = time.time() + 2
    while time.time() < deadline and store.job(job_id)["state"] != "complete":
        time.sleep(0.01)
    worker.stop()
    assert store.job(job_id)["state"] == "complete"


def test_retry_of_completed_snapshot_is_idempotent(tmp_path: Path):
    store = GCIStore(tmp_path / "gci.sqlite")
    calls = 0

    def embed(texts: list[str]) -> EncoderResponse:
        nonlocal calls
        calls += 1
        return EncoderResponse(tuple((1.0,) for _ in texts), 1.0)

    indexer = GCIIndexer(store, embed=embed)
    snapshot = _snapshot({"same.py": "def same():\n    return 'idempotent snapshot'\n"})
    indexer.index(snapshot)
    first_calls = calls
    stats = indexer.index(snapshot)
    assert calls == first_calls
    assert stats.changed == 0
    assert stats.unchanged == 1


def test_encoder_errors_back_off_and_retry(tmp_path: Path):
    store = GCIStore(tmp_path / "gci.sqlite")
    attempts = 0
    sleeps: list[float] = []

    def embed(texts: list[str]) -> EncoderResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("busy")
        return EncoderResponse(tuple((1.0,) for _ in texts), 1.0)

    stats = GCIIndexer(store, embed=embed, sleep=sleeps.append).index(
        _snapshot({"retry.py": "def retry():\n    return 'after one backoff'\n"})
    )
    assert attempts == 2
    assert stats.backoffs == 1
    assert sleeps == [1]


def test_worker_queue_is_bounded(tmp_path: Path):
    store = GCIStore(tmp_path / "gci.sqlite")

    def embed(texts: list[str]) -> EncoderResponse:
        return EncoderResponse(tuple((1.0,) for _ in texts), 1.0)

    worker = IndexWorker(GCIIndexer(store, embed=embed), max_queue_depth=1)
    first = worker.submit(_snapshot({"one.py": "def one():\n    return 1\n"}))
    with pytest.raises(GCIStorageError, match="queue is full"):
        worker.submit(_snapshot({"two.py": "def two():\n    return 2\n"}, head="def"))
    assert store.job(first)["state"] == "queued"


def test_worker_start_fails_jobs_whose_payload_was_lost_on_restart(tmp_path: Path):
    store = GCIStore(tmp_path / "gci.sqlite")
    stale = store.create_job("repo")

    def embed(texts: list[str]) -> EncoderResponse:
        return EncoderResponse(tuple((1.0,) for _ in texts), 1.0)

    worker = IndexWorker(GCIIndexer(store, embed=embed))
    worker.start()
    worker.stop()
    row = store.job(stale)
    assert row["state"] == "failed"
    assert "restarted" in row["error"]
