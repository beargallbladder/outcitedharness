from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from harness.gci.client import GCIClient
from harness.gci.models import GCIHit
from harness.greenfield.models import (
    DiscoveryHit,
    DiscoveryPattern,
    DiscoveryQuery,
    GreenfieldDiscovery,
)
from harness.storage.db import utcnow


CATEGORY_LIMITS = {
    "architecture": 4,
    "domain": 4,
    "testing": 3,
    "tooling": 3,
}
MAX_DISCOVERY_HITS = sum(CATEGORY_LIMITS.values())
MAX_EXCERPT_CHARS = 320
SELECTED_LIMITS = {
    "architecture": 3,
    "domain": 3,
    "testing": 2,
    "tooling": 2,
}

SearchFn = Callable[..., list[GCIHit]]


def discovery_queries(intent: str, stack: str) -> list[DiscoveryQuery]:
    stack_terms = "FastAPI Python service" if stack == "python" else "TypeScript Node application"
    return [
        DiscoveryQuery(
            "architecture",
            f"{stack_terms} architecture service layout for: {intent}",
            "semantic",
            CATEGORY_LIMITS["architecture"],
        ),
        DiscoveryQuery(
            "domain",
            f"domain implementation analogous to: {intent}",
            "semantic",
            CATEGORY_LIMITS["domain"],
        ),
        DiscoveryQuery(
            "testing",
            f"{stack_terms} test fixtures integration testing conventions",
            "semantic",
            CATEGORY_LIMITS["testing"],
        ),
        DiscoveryQuery(
            "tooling",
            f"{stack_terms} project configuration package scripts .harness.toml",
            "semantic",
            CATEGORY_LIMITS["tooling"],
        ),
    ]


def _client(settings: Any) -> GCIClient | None:
    if not settings or not getattr(settings, "gci_enabled", False):
        return None
    token = os.environ.get(
        str(getattr(settings, "gci_token_env", "HARNESS_GCI_TOKEN")),
        "",
    )
    if not token:
        return None
    return GCIClient(
        str(getattr(settings, "gci_url", "http://100.81.201.24:8810")),
        token=token,
        timeout=float(getattr(settings, "gci_timeout_s", 8.0)),
    )


def gather_discovery(
    settings: Any,
    intent: str,
    stack: str,
    *,
    search: SearchFn | None = None,
) -> GreenfieldDiscovery:
    queries = discovery_queries(intent, stack)
    client = None if search else _client(settings)
    search_fn = search or (client.search if client else None)
    discovery = GreenfieldDiscovery(queries=queries, compiled_at=utcnow())
    if search_fn is None:
        discovery.rejected_patterns.append(
            DiscoveryPattern(
                category="service",
                summary="GCI discovery unavailable; plan without repository analogues",
                provenance="gci://unavailable",
                reason="GCI is disabled or its bearer token is unavailable",
            )
        )
        return discovery
    seen: set[tuple[str, str, str, int, int]] = set()
    selected_counts = {category: 0 for category in CATEGORY_LIMITS}
    for query in queries:
        try:
            hits = search_fn(query.query, limit=query.limit, mode=query.mode)
        except Exception as exc:
            discovery.rejected_patterns.append(
                DiscoveryPattern(
                    category=query.category,
                    summary=f"{query.category} discovery unavailable",
                    provenance="gci://unavailable",
                    reason=f"{type(exc).__name__}: search failed without changing the plan",
                )
            )
            continue
        for hit in hits[: query.limit]:
            key = (
                hit.source_host,
                hit.repo_id,
                hit.path,
                hit.start_line,
                hit.end_line,
            )
            row = DiscoveryHit(
                category=query.category,
                repo_id=hit.repo_id,
                source_host=hit.source_host,
                repo_root=hit.repo_root,
                revision=hit.revision,
                state_hash=hit.state_hash,
                path=hit.path,
                symbol=hit.symbol,
                symbol_type=hit.symbol_type,
                start_line=hit.start_line,
                end_line=hit.end_line,
                score=hit.score,
                match_type=hit.match_type,
                excerpt=hit.text[:MAX_EXCERPT_CHARS],
            )
            if key not in seen:
                discovery.repo_hits.append(row)
                seen.add(key)
                if selected_counts[query.category] < SELECTED_LIMITS[query.category]:
                    discovery.selected_patterns.append(
                        DiscoveryPattern(
                            category=query.category,
                            summary=_pattern_summary(row),
                            provenance=row.provenance,
                            reason=(
                                "selected as a top bounded analogue; the approved spec "
                                "remains authoritative"
                            ),
                        )
                    )
                    selected_counts[query.category] += 1
                else:
                    discovery.rejected_patterns.append(
                        DiscoveryPattern(
                            category=query.category,
                            summary=_pattern_summary(row),
                            provenance=row.provenance,
                            reason="not selected because the planning evidence budget is full",
                        )
                    )
            else:
                discovery.rejected_patterns.append(
                    DiscoveryPattern(
                        category=query.category,
                        summary=_pattern_summary(row),
                        provenance=row.provenance,
                        reason="duplicate of a stronger cross-category hit",
                    )
                )
    discovery.repo_hits = discovery.repo_hits[:MAX_DISCOVERY_HITS]
    discovery.selected_patterns = discovery.selected_patterns[:MAX_DISCOVERY_HITS]
    return discovery


def _pattern_summary(hit: DiscoveryHit) -> str:
    subject = hit.symbol or hit.path
    return f"{subject} from {hit.repo_root}/{hit.path}"


def compile_discovery_packet(
    discovery: GreenfieldDiscovery,
    *,
    max_chars: int = 8_000,
) -> str:
    rows = [
        "GREENFIELD DISCOVERY (ADVISORY ONLY)",
        "These namespaced excerpts may inform deliberate design choices. They do not",
        "authorize copying, runtime coupling, filesystem access, or changes to source repos.",
    ]
    for pattern in discovery.selected_patterns:
        rows.append(
            f"- [{pattern.category}] {pattern.summary}\n"
            f"  provenance: {pattern.provenance}\n"
            f"  use: {pattern.reason}"
        )
    for pattern in discovery.rejected_patterns:
        rows.append(
            f"- REJECTED [{pattern.category}] {pattern.summary}\n"
            f"  provenance: {pattern.provenance}\n"
            f"  reason: {pattern.reason}"
        )
    packet = "\n".join(rows)
    return packet[:max_chars]
