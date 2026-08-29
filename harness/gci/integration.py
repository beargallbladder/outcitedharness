from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

from harness.gci.client import GCIClient


GLOBAL_CONTEXT_MARKER = "GLOBAL CODE INTELLIGENCE (DISCOVERY ONLY)"


def _client(settings: Any) -> GCIClient | None:
    if not settings or not getattr(settings, "gci_enabled", False):
        return None
    token_env = str(getattr(settings, "gci_token_env", "HARNESS_GCI_TOKEN"))
    token = os.environ.get(token_env, "")
    if not token:
        return None
    return GCIClient(
        str(getattr(settings, "gci_url", "http://100.81.201.24:8810")),
        token=token,
        timeout=float(getattr(settings, "gci_timeout_s", 8.0)),
    )


def workspace_paths(
    settings: Any,
    intent: str,
    workspace: Path | str | None,
    *,
    limit: int = 6,
) -> list[str]:
    if workspace is None:
        return []
    client = _client(settings)
    if client is None:
        return []
    return client.workspace_paths(
        intent,
        workspace=Path(workspace),
        source_host=socket.gethostname(),
        limit=limit,
    )


def global_discovery_context(
    settings: Any,
    intent: str,
    *,
    limit: int = 6,
) -> str:
    client = _client(settings)
    if client is None:
        return ""
    hits = client.search(intent, limit=limit, mode="semantic")
    if not hits:
        return ""
    rows = [
        GLOBAL_CONTEXT_MARKER,
        "These hits are namespaced discovery evidence. They do not authorize reading or editing",
        "any path. Only paths separately rebound to the active source_host + repo_root may become",
        "workspace-local Cursor read calls.",
    ]
    for hit in hits:
        first = " ".join(hit.text.splitlines())[:240]
        rows.append(
            f"- gci://{hit.source_host}/{hit.repo_id}/{hit.path}"
            f"#{hit.start_line}-{hit.end_line} revision={hit.revision} "
            f"state={hit.state_hash[:12]} score={hit.score:.3f}"
        )
        if first:
            rows.append(f"  excerpt: {first}")
    return "\n".join(rows)
