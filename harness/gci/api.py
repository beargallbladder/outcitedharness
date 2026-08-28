from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from harness.gci.indexer import GCIIndexer, IndexWorker
from harness.gci.models import CodeDocument, RepoSnapshot
from harness.gci.storage import GCIStore


MAX_BODY_BYTES = 10_000_000
MAX_SEARCH_LIMIT = 100


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    kind, _, value = header.partition(" ")
    return value.strip() if kind.lower() == "bearer" else ""


def _authorized(request: Request, token: str) -> bool:
    return bool(token) and hmac.compare_digest(_bearer(request), token)


async def _json_body(request: Request) -> dict[str, Any]:
    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > MAX_BODY_BYTES:
                raise ValueError("request body is too large")
        except ValueError as exc:
            if "too large" in str(exc):
                raise
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise ValueError("request body is too large")
    data = await request.json()
    if not isinstance(data, dict):
        raise ValueError("request body must be an object")
    return data


def _snapshot(data: dict[str, Any]) -> RepoSnapshot:
    file_hashes = data.get("file_hashes")
    documents = data.get("documents") or []
    deleted = data.get("deleted") or []
    if not isinstance(file_hashes, dict):
        raise ValueError("file_hashes must be an object")
    if not isinstance(documents, list) or not isinstance(deleted, list):
        raise ValueError("documents and deleted must be arrays")
    parsed = []
    for row in documents:
        if not isinstance(row, dict):
            raise ValueError("document must be an object")
        parsed.append(
            CodeDocument(
                path=str(row.get("path") or ""),
                content=str(row.get("content") or ""),
                content_hash=str(row.get("content_hash") or ""),
                language=str(row.get("language") or ""),
            )
        )
    return RepoSnapshot(
        repo_id=str(data.get("repo_id") or ""),
        source_host=str(data.get("source_host") or ""),
        repo_root=str(data.get("repo_root") or ""),
        remote=str(data["remote"]) if data.get("remote") else None,
        branch=str(data.get("branch") or ""),
        head=str(data.get("head") or ""),
        dirty=bool(data.get("dirty")),
        state_hash=str(data.get("state_hash") or ""),
        file_hashes={str(path): str(digest) for path, digest in file_hashes.items()},
        documents=tuple(parsed),
        deleted=tuple(str(path) for path in deleted),
    )


def _limit(data: dict[str, Any], default: int) -> int:
    value = int(data.get("limit", default))
    if not 1 <= value <= MAX_SEARCH_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_SEARCH_LIMIT}")
    return value


def create_app(
    store: GCIStore,
    indexer: GCIIndexer,
    *,
    token: str,
    worker: IndexWorker | None = None,
) -> Starlette:
    if not token:
        raise ValueError("GCI bearer token is required")
    worker = worker or IndexWorker(indexer)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        worker.start()
        try:
            yield
        finally:
            worker.stop()

    async def health(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "ready": True,
                "service": "harness-global-code-intelligence",
                "queue_depth": worker.queue_depth,
                "paused": store.paused(),
            }
        )

    async def guarded(request: Request, handler):
        if not _authorized(request, token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            return await handler(request)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse(
                {"error": f"{type(exc).__name__}: {exc}"},
                status_code=409,
            )

    async def metrics_impl(request: Request) -> JSONResponse:
        return JSONResponse({**store.metrics(), "queue_depth": worker.queue_depth})

    async def repos_impl(request: Request) -> JSONResponse:
        return JSONResponse({"repos": store.repos()})

    async def job_impl(request: Request) -> JSONResponse:
        row = store.job(request.path_params["job_id"])
        return JSONResponse(row or {"error": "job not found"}, status_code=200 if row else 404)

    async def manifest_impl(request: Request) -> JSONResponse:
        data = await _json_body(request)
        return JSONResponse(store.repo_manifest(str(data.get("repo_id") or "")))

    async def submit_impl(request: Request) -> JSONResponse:
        data = await _json_body(request)
        job_id = worker.submit(_snapshot(data))
        return JSONResponse({"job_id": job_id, "state": "queued"}, status_code=202)

    async def search_impl(request: Request) -> JSONResponse:
        data = await _json_body(request)
        query = str(data.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        response = indexer.embed_query(query)
        hits = store.semantic_search(
            response.vectors[0],
            limit=_limit(data, 8),
            repo_root=str(data["repo_root"]) if data.get("repo_root") else None,
            source_host=str(data["source_host"]) if data.get("source_host") else None,
        )
        return JSONResponse(
            {
                "hits": [asdict(hit) for hit in hits],
                "query_embedding_ms": response.latency_ms,
            }
        )

    async def exact_impl(request: Request) -> JSONResponse:
        data = await _json_body(request)
        query = str(data.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        hits = store.exact_search(
            query,
            limit=_limit(data, 20),
            repo_root=str(data["repo_root"]) if data.get("repo_root") else None,
        )
        return JSONResponse({"hits": [asdict(hit) for hit in hits]})

    async def symbol_impl(request: Request) -> JSONResponse:
        data = await _json_body(request)
        query = str(data.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        hits = store.symbol_search(
            query,
            limit=_limit(data, 20),
            repo_root=str(data["repo_root"]) if data.get("repo_root") else None,
        )
        return JSONResponse({"hits": [asdict(hit) for hit in hits]})

    async def pause_impl(request: Request) -> JSONResponse:
        store.set_paused(True)
        return JSONResponse({"paused": True})

    async def resume_impl(request: Request) -> JSONResponse:
        store.set_paused(False)
        return JSONResponse({"paused": False})

    def protect(handler):
        async def endpoint(request: Request):
            return await guarded(request, handler)

        return endpoint

    return Starlette(
        lifespan=lifespan,
        routes=[
            Route("/health", protect(health), methods=["GET"]),
            Route("/metrics", protect(metrics_impl), methods=["GET"]),
            Route("/repos", protect(repos_impl), methods=["GET"]),
            Route("/jobs/{job_id}", protect(job_impl), methods=["GET"]),
            Route("/changed-since", protect(manifest_impl), methods=["POST"]),
            Route("/index-repo", protect(submit_impl), methods=["POST"]),
            Route("/refresh-repo", protect(submit_impl), methods=["POST"]),
            Route("/search", protect(search_impl), methods=["POST"]),
            Route("/search-exact", protect(exact_impl), methods=["POST"]),
            Route("/search-symbol", protect(symbol_impl), methods=["POST"]),
            Route("/admin/pause", protect(pause_impl), methods=["POST"]),
            Route("/admin/resume", protect(resume_impl), methods=["POST"]),
        ],
    )


def serve(
    *,
    host: str | None = None,
    port: int | None = None,
    db_path: Path | None = None,
    token: str | None = None,
) -> None:
    import uvicorn

    secret = token or os.environ.get("HARNESS_GCI_TOKEN", "")
    store = GCIStore(db_path or Path(os.environ.get("HARNESS_GCI_DB", "/data/harness-gci/code-intel.sqlite")))
    indexer = GCIIndexer(store)
    uvicorn.run(
        create_app(store, indexer, token=secret),
        host=host or os.environ.get("HARNESS_GCI_HOST", "100.81.201.24"),
        port=port or int(os.environ.get("HARNESS_GCI_PORT", "8810")),
        log_level="info",
    )
