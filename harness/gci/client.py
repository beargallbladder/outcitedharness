from __future__ import annotations

import os
import socket
import time
import urllib.parse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from harness.gci.models import GCIHit, RepoSnapshot


class GCIClientError(RuntimeError):
    pass


def validate_gci_url(url: str) -> str:
    value = url.rstrip("/")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "http" or parsed.port != 8810 or parsed.path not in {"", "/"}:
        raise GCIClientError("GCI URL must be an HTTP origin on port 8810")
    if parsed.query or parsed.fragment or not parsed.hostname:
        raise GCIClientError("GCI URL cannot include query or fragment")
    return value


class GCIClient:
    def __init__(
        self,
        base_url: str,
        *,
        token: str,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ):
        if not token:
            raise GCIClientError("GCI bearer token is required")
        self.base_url = validate_gci_url(base_url)
        self.token = token
        self.timeout = timeout
        self._client = client

    def _request(self, method: str, path: str, data: dict | None = None) -> dict[str, Any]:
        headers = {"authorization": f"Bearer {self.token}"}
        owned = self._client is None
        client = self._client or httpx.Client(timeout=self.timeout, follow_redirects=False)
        try:
            response = client.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                json=data,
            )
        finally:
            if owned:
                client.close()
        if response.is_redirect:
            raise GCIClientError("GCI redirect refused")
        if response.status_code >= 400:
            raise GCIClientError(f"GCI HTTP {response.status_code}: {response.text[:300]}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise GCIClientError("GCI returned a non-object response")
        return payload

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def repos(self) -> list[dict]:
        return list(self._request("GET", "/repos").get("repos") or [])

    def manifest(self, repo_id: str) -> dict:
        return self._request("POST", "/changed-since", {"repo_id": repo_id})

    def submit(self, snapshot: RepoSnapshot, *, refresh: bool) -> str:
        payload = asdict(snapshot)
        row = self._request(
            "POST",
            "/refresh-repo" if refresh else "/index-repo",
            payload,
        )
        return str(row["job_id"])

    def job(self, job_id: str) -> dict:
        return self._request("GET", f"/jobs/{job_id}")

    def wait_job(self, job_id: str, *, timeout: float = 300.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            row = self.job(job_id)
            if row.get("state") in {"complete", "failed", "cancelled"}:
                return row
            time.sleep(0.25)
        raise GCIClientError(f"timed out waiting for GCI job {job_id}")

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        repo_root: str | None = None,
        source_host: str | None = None,
        mode: str = "semantic",
    ) -> list[GCIHit]:
        route = {
            "semantic": "/search",
            "exact": "/search-exact",
            "symbol": "/search-symbol",
        }.get(mode)
        if not route:
            raise GCIClientError(f"unknown GCI search mode: {mode}")
        payload: dict[str, Any] = {"query": query, "limit": limit}
        if repo_root:
            payload["repo_root"] = str(Path(repo_root).expanduser().resolve())
        if source_host:
            payload["source_host"] = source_host
        rows = self._request("POST", route, payload).get("hits") or []
        return [GCIHit(**row) for row in rows if isinstance(row, dict)]

    def workspace_paths(
        self,
        query: str,
        *,
        workspace: Path,
        source_host: str | None = None,
        limit: int = 6,
    ) -> list[str]:
        root = workspace.expanduser().resolve()
        host = source_host or socket.gethostname()
        hits = self.search(
            query,
            limit=limit * 3,
            repo_root=str(root),
            source_host=host,
        )
        paths: list[str] = []
        for hit in hits:
            if hit.source_host != host or Path(hit.repo_root).resolve() != root:
                continue
            path = Path(hit.path)
            if path.is_absolute() or ".." in path.parts:
                continue
            value = path.as_posix()
            if value not in paths:
                paths.append(value)
            if len(paths) >= limit:
                break
        return paths

    def pause(self, paused: bool) -> bool:
        row = self._request("POST", "/admin/pause" if paused else "/admin/resume")
        return bool(row.get("paused"))


def client_from_env(
    *,
    base_url: str | None = None,
    token: str | None = None,
    timeout: float = 30.0,
) -> GCIClient:
    return GCIClient(
        base_url or os.environ.get("HARNESS_GCI_URL", "http://100.81.201.24:8810"),
        token=token or os.environ.get("HARNESS_GCI_TOKEN", ""),
        timeout=timeout,
    )
