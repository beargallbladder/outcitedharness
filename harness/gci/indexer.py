from __future__ import annotations

import hashlib
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

from harness.gci.encoder import EncoderResponse, MAX_ENCODER_BATCH, StrictEncoder
from harness.gci.models import IndexStats, PreparedDocument, RepoSnapshot
from harness.gci.slicing import extract_imports, extract_symbols, semantic_slices
from harness.gci.storage import GCIStorageError, GCIStore


MAX_QUEUE_DEPTH = 16
MAX_RETRIES = 3
LATENCY_BACKOFF_THRESHOLD_MS = 2_000.0

EmbedFn = Callable[[list[str]], EncoderResponse]


def snapshot_state_hash(head: str, file_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(head.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    for path, content_hash in sorted(file_hashes.items()):
        digest.update(path.encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_relative_path(raw: str) -> str:
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise GCIStorageError(f"unsafe repository path: {raw!r}")
    normalized = path.as_posix()
    if normalized.startswith("/"):
        raise GCIStorageError(f"unsafe repository path: {raw!r}")
    return normalized


class GCIIndexer:
    def __init__(
        self,
        store: GCIStore,
        *,
        embed: EmbedFn | None = None,
        batch_size: int = MAX_ENCODER_BATCH,
        max_retries: int = MAX_RETRIES,
        latency_threshold_ms: float = LATENCY_BACKOFF_THRESHOLD_MS,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not 1 <= batch_size <= MAX_ENCODER_BATCH:
            raise ValueError(f"batch_size must be between 1 and {MAX_ENCODER_BATCH}")
        self.store = store
        self.embed = embed or StrictEncoder().embed
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.latency_threshold_ms = latency_threshold_ms
        self.sleep = sleep
        self._encoder_lock = threading.Lock()

    def index(self, snapshot: RepoSnapshot) -> IndexStats:
        if not snapshot.repo_id or not snapshot.source_host or not snapshot.repo_root:
            raise GCIStorageError("repository identity is incomplete")
        if snapshot.state_hash != snapshot_state_hash(snapshot.head, snapshot.file_hashes):
            raise GCIStorageError("snapshot state hash does not match file manifest")
        expected = {_safe_relative_path(path): digest for path, digest in snapshot.file_hashes.items()}
        previous = self.store.repo_manifest(snapshot.repo_id)
        previous_files = previous.get("files") or {}
        if previous.get("state_hash") == snapshot.state_hash and previous_files == expected:
            return IndexStats(files=len(expected), unchanged=len(expected))
        changed_paths = {
            path for path, digest in expected.items() if previous_files.get(path) != digest
        }
        deleted_paths = set(previous_files) - set(expected)
        submitted = {_safe_relative_path(row.path): row for row in snapshot.documents}
        if set(snapshot.deleted) != deleted_paths:
            raise GCIStorageError("submitted deletions do not match indexed manifest")
        missing = changed_paths - set(submitted)
        if missing:
            raise GCIStorageError(f"changed documents missing from snapshot: {sorted(missing)}")
        extra = set(submitted) - changed_paths
        if extra:
            raise GCIStorageError(f"unchanged documents must not be uploaded: {sorted(extra)}")
        stats = IndexStats(
            files=len(expected),
            changed=len(changed_paths),
            unchanged=len(expected) - len(changed_paths),
            deleted=len(deleted_paths),
        )
        prepared: list[PreparedDocument] = []
        pending_texts: list[str] = []
        pending_map: list[tuple[int, int]] = []
        for path in sorted(changed_paths):
            document = submitted[path]
            actual_hash = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
            if document.content_hash != expected[path] or actual_hash != expected[path]:
                raise GCIStorageError(f"content hash mismatch for {path}")
            slices = tuple(semantic_slices(path, document.content, document.language))
            prepared.append(
                PreparedDocument(
                    document=document,
                    slices=slices,
                    symbols=tuple(extract_symbols(path, document.content, document.language)),
                    imports=tuple(extract_imports(path, document.content, document.language)),
                )
            )
            item_index = len(prepared) - 1
            for slice_index, item in enumerate(slices):
                pending_texts.append(item.text)
                pending_map.append((item_index, slice_index))
        stats.slices = len(pending_texts)
        vectors_by_doc: list[list[tuple[float, ...] | None]] = [
            [None] * len(item.slices) for item in prepared
        ]
        for offset in range(0, len(pending_texts), self.batch_size):
            batch = pending_texts[offset : offset + self.batch_size]
            response = self._embed_with_backoff(batch, stats)
            for vector, (item_index, slice_index) in zip(
                response.vectors,
                pending_map[offset : offset + len(batch)],
            ):
                vectors_by_doc[item_index][slice_index] = vector
        finalized = []
        for item, vectors in zip(prepared, vectors_by_doc):
            if any(vector is None for vector in vectors):
                raise GCIStorageError(f"missing embedding for {item.document.path}")
            finalized.append(
                PreparedDocument(
                    document=item.document,
                    slices=item.slices,
                    symbols=item.symbols,
                    imports=item.imports,
                    embeddings=tuple(vector for vector in vectors if vector is not None),
                )
            )
        stats.embedded = sum(len(item.embeddings) for item in finalized)
        self.store.commit_generation(snapshot, finalized, stats)
        return stats

    def _embed_with_backoff(self, texts: list[str], stats: IndexStats) -> EncoderResponse:
        for attempt in range(self.max_retries + 1):
            try:
                stats.encoder_calls += 1
                with self._encoder_lock:
                    response = self.embed(texts)
                stats.latency_ms.append(response.latency_ms)
                if (
                    response.latency_ms > self.latency_threshold_ms
                    and attempt < self.max_retries
                ):
                    stats.backoffs += 1
                    self.sleep(min(2**attempt, 8))
                return response
            except Exception:
                if attempt >= self.max_retries:
                    raise
                stats.backoffs += 1
                self.sleep(min(2**attempt, 8))
        raise AssertionError("unreachable")

    def embed_query(self, text: str) -> EncoderResponse:
        if not text.strip():
            raise GCIStorageError("search query cannot be empty")
        stats = IndexStats()
        return self._embed_with_backoff([text[:450]], stats)


class IndexWorker:
    def __init__(
        self,
        indexer: GCIIndexer,
        *,
        max_queue_depth: int = MAX_QUEUE_DEPTH,
        pause_poll_seconds: float = 1.0,
    ):
        self.indexer = indexer
        self.store = indexer.store
        self.pause_poll_seconds = pause_poll_seconds
        self.queue: queue.Queue[tuple[str, RepoSnapshot] | None] = queue.Queue(
            maxsize=max_queue_depth
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.store.fail_incomplete_jobs("service restarted before job completion")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gci-index-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout)

    def submit(self, snapshot: RepoSnapshot) -> str:
        job_id = self.store.create_job(snapshot.repo_id)
        try:
            self.queue.put_nowait((job_id, snapshot))
        except queue.Full as exc:
            self.store.update_job(job_id, "failed", error="index queue is full")
            raise GCIStorageError("index queue is full") from exc
        return job_id

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            job_id, snapshot = item
            try:
                while self.store.paused() and not self._stop.wait(self.pause_poll_seconds):
                    pass
                if self._stop.is_set():
                    self.store.update_job(job_id, "cancelled", error="worker stopped")
                    continue
                self.store.update_job(job_id, "running")
                stats = self.indexer.index(snapshot)
                self.store.update_job(job_id, "complete", stats=stats)
            except Exception as exc:
                self.store.update_job(
                    job_id,
                    "failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                self.queue.task_done()
