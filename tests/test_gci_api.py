from __future__ import annotations

import hashlib
import socket
import time
from pathlib import Path

from starlette.testclient import TestClient

from harness.gci.api import create_app
from harness.gci.client import GCIClient
from harness.gci.encoder import EncoderResponse
from harness.gci.indexer import GCIIndexer, IndexWorker, snapshot_state_hash
from harness.gci.models import CodeDocument, RepoSnapshot
from harness.gci.storage import GCIStore


TOKEN = "test-secret"


def _snapshot(root: Path, source_host: str) -> RepoSnapshot:
    content = "import math\n\ndef percentile_score(values):\n    return math.floor(sum(values))\n"
    digest = hashlib.sha256(content.encode()).hexdigest()
    hashes = {"src/scoring.py": digest}
    return RepoSnapshot(
        repo_id="repo-a",
        source_host=source_host,
        repo_root=str(root.resolve()),
        remote=None,
        branch="main",
        head="abc",
        dirty=False,
        state_hash=snapshot_state_hash("abc", hashes),
        file_hashes=hashes,
        documents=(
            CodeDocument(
                path="src/scoring.py",
                content=content,
                content_hash=digest,
                language="python",
            ),
        ),
    )


def _app(tmp_path: Path):
    store = GCIStore(tmp_path / "gci.sqlite")

    def embed(texts: list[str]) -> EncoderResponse:
        return EncoderResponse(tuple((1.0, 0.0, 0.0) for _ in texts), 2.0)

    indexer = GCIIndexer(store, embed=embed)
    worker = IndexWorker(indexer, pause_poll_seconds=0.01)
    return create_app(store, indexer, token=TOKEN, worker=worker)


def test_api_requires_auth_and_indexes_searches(tmp_path: Path):
    app = _app(tmp_path)
    with TestClient(app, base_url="http://testserver:8810") as http:
        assert http.get("/health").status_code == 401
        assert http.get("/repos").status_code == 401
        client = GCIClient("http://testserver:8810", token=TOKEN, client=http)
        assert client.health()["ready"] is True
        snapshot = _snapshot(tmp_path / "repo", socket.gethostname())
        job_id = client.submit(snapshot, refresh=False)
        deadline = time.time() + 2
        while client.job(job_id)["state"] not in {"complete", "failed"} and time.time() < deadline:
            time.sleep(0.01)
        assert client.job(job_id)["state"] == "complete"
        assert client.search("percentile", mode="symbol")[0].symbol == "percentile_score"
        assert client.search("math.floor", mode="exact")[0].path == "src/scoring.py"
        assert client.search("where percentile scoring happens")[0].repo_id == "repo-a"
        assert client.repos()[0]["file_count"] == 1


def test_workspace_binding_rejects_foreign_host_and_root(tmp_path: Path):
    app = _app(tmp_path)
    root = tmp_path / "repo"
    root.mkdir()
    with TestClient(app, base_url="http://testserver:8810") as http:
        client = GCIClient("http://testserver:8810", token=TOKEN, client=http)
        job_id = client.submit(_snapshot(root, "other-host"), refresh=False)
        deadline = time.time() + 2
        while client.job(job_id)["state"] not in {"complete", "failed"} and time.time() < deadline:
            time.sleep(0.01)
        assert client.workspace_paths(
            "percentile",
            workspace=root,
            source_host=socket.gethostname(),
        ) == []
        assert client.workspace_paths(
            "percentile",
            workspace=root,
            source_host="other-host",
        ) == ["src/scoring.py"]


def test_pause_and_metrics_are_authenticated(tmp_path: Path):
    app = _app(tmp_path)
    with TestClient(app, base_url="http://testserver:8810") as http:
        client = GCIClient("http://testserver:8810", token=TOKEN, client=http)
        assert client.pause(True)
        assert http.get("/metrics").status_code == 401
        metrics = http.get(
            "/metrics",
            headers={"authorization": f"Bearer {TOKEN}"},
        ).json()
        assert metrics["paused"] is True
        assert client.pause(False) is False


def test_api_rejects_oversized_payload_before_parsing(tmp_path: Path):
    app = _app(tmp_path)
    with TestClient(app, base_url="http://testserver:8810") as http:
        response = http.post(
            "/search-exact",
            content=b"{}",
            headers={
                "authorization": f"Bearer {TOKEN}",
                "content-length": "10000001",
                "content-type": "application/json",
            },
        )
        assert response.status_code == 400
        assert "too large" in response.json()["error"]
