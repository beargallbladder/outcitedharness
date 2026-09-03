#!/usr/bin/env python3
"""Filter broad page candidates through capability-specific structural evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.electronics.claims import canonical_json
from harness.electronics.locator import package_pin_count
from harness.electronics.regions import (
    structural_parametric_regions,
    structural_pin_regions,
    structural_text_regions,
)


DIAGRAM_TITLE = re.compile(
    r"\b(?:PIN|BALL)\s*(?:OUTS?|MAPS?|DIAGRAMS?|ASSIGNMENTS?|CONFIGURATION)\b",
    re.IGNORECASE,
)
SUPPORTED_CAPABILITIES = {
    "opn_decoder",
    "parametrics",
    "pin_or_ball",
    "pin_semantics",
    "series_summary",
}
DOCUMENT_SCOPED_CAPABILITIES = {
    "opn_decoder",
    "parametrics",
    "series_summary",
}
SEMANTIC_ROLES = {"type", "dir", "supply_domain", "functions"}


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority-queue", type=Path, required=True)
    parser.add_argument("--page-evidence", type=Path, required=True)
    parser.add_argument("--page-index", type=Path, required=True)
    parser.add_argument("--maximum-work", type=int, default=5000)
    parser.add_argument(
        "--capability",
        action="append",
        choices=tuple(sorted(SUPPORTED_CAPABILITIES)),
        help="Capability to include; repeat as needed. Defaults to all.",
    )
    parser.add_argument(
        "--package-scope-policy",
        choices=("require", "allow-withhold"),
        default="require",
    )
    parser.add_argument(
        "--require-semantic-role",
        action="append",
        default=[],
        choices=tuple(sorted(SEMANTIC_ROLES)),
        help=(
            "Require printed values for this pin-semantic field; repeat to "
            "require multiple fields."
        ),
    )
    parser.add_argument(
        "--minimum-semantic-role-count",
        type=int,
        default=0,
        help="Require at least this many populated pin-semantic field roles.",
    )
    parser.add_argument(
        "--exclude-work-queue",
        action="append",
        default=[],
        type=Path,
        help="Previously processed structural queue; repeat as needed.",
    )
    parser.add_argument(
        "--pin-work-source",
        choices=("priority", "locator"),
        default="priority",
        help=(
            "priority: pin pages must be proposed by the priority queue; "
            "locator: additionally synthesize pin work for every sendable "
            "TOC-located definition-table page of priority-queue documents."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.maximum_work < 1:
        raise ValueError("maximum work must be positive")
    if not 0 <= args.minimum_semantic_role_count <= len(SEMANTIC_ROLES):
        raise ValueError("--minimum-semantic-role-count must be within 0..4")
    selected_capabilities = set(args.capability or SUPPORTED_CAPABILITIES)
    required_semantic_roles = set(args.require_semantic_role)
    if (
        required_semantic_roles or args.minimum_semantic_role_count
    ) and "pin_semantics" not in selected_capabilities:
        raise ValueError(
            "semantic role filters require the pin_semantics capability"
        )

    priority_path = args.priority_queue.expanduser().resolve(strict=True)
    evidence_root = args.page_evidence.expanduser().resolve(strict=True)
    index_root = args.page_index.expanduser().resolve(strict=True)
    excluded_paths = [
        path.expanduser().resolve(strict=True)
        for path in args.exclude_work_queue
    ]
    excluded_work_ids: set[str] = set()
    for excluded_path in excluded_paths:
        excluded_queue = json.loads(excluded_path.read_text(encoding="utf-8"))
        if (
            excluded_queue.get("schema")
            != "harness.electronics-structural-local-work.v1"
        ):
            raise ValueError(
                f"excluded queue has unsupported schema: {excluded_path}"
            )
        excluded_work_ids.update(
            str(item["work_id"])
            for item in excluded_queue.get("work") or []
        )
    priority = json.loads(priority_path.read_text(encoding="utf-8"))
    work = list(priority["work"])

    diagram_pages: set[tuple[str, int]] = set()
    exact_by_page: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for profile in _jsonl(index_root / "profiles.jsonl"):
        document_sha = profile["document_sha256"]
        for entry in profile.get("section_entries") or []:
            if DIAGRAM_TITLE.search(str(entry.get("title") or "")):
                diagram_pages.add(
                    (document_sha, int(entry["page_1based"]))
                )
        for location in profile.get("exact_pin_locations") or []:
            if location.get("status") != "send":
                continue
            for page_number in location.get("pages_1based") or []:
                exact_by_page[(document_sha, int(page_number))].append(
                    location
                )

    counts: Counter[str] = Counter()
    if args.pin_work_source == "locator":
        pin_templates: dict[str, dict[str, Any]] = {}
        pin_pages: set[tuple[str, int]] = set()
        for row in work:
            if row["capability"] in {"pin_or_ball", "pin_semantics"}:
                pin_templates.setdefault(row["document_sha256"], row)
                pin_pages.add(
                    (row["document_sha256"], int(row["page_1based"]))
                )
        for document_sha, page_number in sorted(exact_by_page):
            template = pin_templates.get(document_sha)
            if template is None:
                counts["locator_page_without_priority_document"] += 1
                continue
            if (document_sha, page_number) in pin_pages:
                continue
            work.append(
                {
                    **template,
                    "capability": "pin_semantics",
                    "page_1based": page_number,
                    "work_id": "local-"
                    + hashlib.sha256(
                        canonical_json(
                            (document_sha, page_number, "locator_page")
                        )
                    ).hexdigest()[:32],
                    "priority_basis": {
                        **(template.get("priority_basis") or {}),
                        "locator_page_supplement": 1.0,
                    },
                }
            )
            counts["synthesized_locator_pin_page"] += 1

    wanted = {
        (row["document_sha256"], int(row["page_1based"]))
        for row in work
        if row["capability"] in selected_capabilities
    }
    pages = {
        (page["document_sha256"], int(page["page_1based"])): page
        for page in _jsonl(evidence_root / "page-evidence.jsonl")
        if (page["document_sha256"], int(page["page_1based"])) in wanted
    }

    selected: list[dict[str, Any]] = []
    for item in work:
        capability = item["capability"]
        if capability not in selected_capabilities:
            continue
        key = (item["document_sha256"], int(item["page_1based"]))
        page = pages.get(key)
        if page is None:
            continue
        mode: str | None = None
        if capability == "parametrics":
            regions = structural_parametric_regions(page)
            authority = "visible_parametric_table"
            if regions:
                mode = "focused_parametric_table"
                effective_capability = "parametrics"
        elif capability in {"opn_decoder", "series_summary"}:
            regions = structural_text_regions(page, capability)
            authority = (
                "visible_ordering_evidence"
                if capability == "opn_decoder"
                else "visible_summary_evidence"
            )
            if regions:
                mode = (
                    "focused_ordering_evidence"
                    if capability == "opn_decoder"
                    else "focused_summary_evidence"
                )
                effective_capability = capability
        else:
            if key not in exact_by_page:
                counts["withheld_pin_without_exact_toc_location"] += 1
                continue
            regions = structural_pin_regions(page)
            authority = "definition_table"
            if regions:
                mode = "focused_structural_table"
                effective_capability = (
                    "pin_semantics"
                    if any(
                        "semantic_header" in region.get("reasons", ())
                        for region in regions
                    )
                    else "pin_or_ball"
                )
            elif key in diagram_pages and capability == "pin_or_ball":
                counts["withheld_diagram_corroboration_only"] += 1
                continue
        if mode is None:
            counts["rejected_broad_heading_without_structure"] += 1
            continue
        available_roles = {
            str(role)
            for region in regions
            for role in region.get("semantic_roles", ())
        }
        if required_semantic_roles or args.minimum_semantic_role_count:
            missing_roles = required_semantic_roles - available_roles
            if (
                effective_capability != "pin_semantics"
                or missing_roles
                or len(available_roles) < args.minimum_semantic_role_count
            ):
                counts["rejected_semantic_role_filter"] += 1
                if missing_roles:
                    counts["rejected_missing_required_semantic_roles"] += 1
                for role in sorted(missing_roles):
                    counts[f"rejected_missing_semantic_role:{role}"] += 1
                if len(available_roles) < args.minimum_semantic_role_count:
                    counts["rejected_semantic_role_count"] += 1
                continue
        scopes: list[dict[str, Any] | None] = []
        exact_locations = exact_by_page.get(key, [])
        if capability in DOCUMENT_SCOPED_CAPABILITIES:
            scopes = [None]
        elif exact_locations:
            grouped_locations: dict[
                tuple[str, str], list[dict[str, Any]]
            ] = defaultdict(list)
            for location in exact_locations:
                grouped_locations[
                    (location["package"], location["column_header"])
                ].append(location)
            for (package, column_header), locations in sorted(
                grouped_locations.items()
            ):
                package_count = package_pin_count(package)
                location_counts = {
                    int(location["expected_package_pins"])
                    for location in locations
                    if location.get("expected_package_pins") is not None
                }
                expected_pins = (
                    next(iter(location_counts))
                    if len(location_counts) == 1
                    else package_count
                )
                if len(location_counts) > 1:
                    counts["normalized_conflicting_gt_pin_counts"] += 1
                scope = {
                    "package": package,
                    "column_header": column_header,
                    "expected_package_pins": expected_pins,
                    "source": "exact_gt_locator",
                }
                scopes.append(scope)
        if not scopes:
            counts["withheld_unresolved_package"] += 1
            if args.package_scope_policy == "require":
                continue
            scopes = [None]

        for scope in scopes:
            scoped_id = (
                item["work_id"]
                if scope is None
                else "local-"
                + hashlib.sha256(
                    canonical_json(
                        (
                            item["work_id"],
                            effective_capability,
                            scope["package"],
                            scope["column_header"],
                        )
                    )
                ).hexdigest()[:32]
            )
            if scoped_id in excluded_work_ids:
                counts["excluded_previous_work"] += 1
                continue
            selected.append(
                {
                    **item,
                    "capability": effective_capability,
                    "source_work_id": item["work_id"],
                    "work_id": scoped_id,
                    "structural_evidence": {
                        "mode": mode,
                        "authority": authority,
                        "regions": regions,
                        "package_scope": scope,
                        "semantic_roles": sorted(available_roles),
                        "exact_opns": list(item.get("exact_opns") or []),
                    },
                    "pipeline": (
                        [
                            "toc_definition_table_locator",
                            "find_tables_package_header_gate",
                            "full_page_render_120dpi",
                            "local_page_vision",
                            "anthropic_page_vision_teacher",
                            "merge_unique_physical_ids",
                            "exact_package_pin_count_gate",
                        ]
                        if capability in {"pin_or_ball", "pin_semantics"}
                        else [
                            "source_page_locator",
                            "deterministic_structure",
                            "local_page_vision",
                            "anthropic_page_vision_teacher",
                        ]
                    ),
                }
            )
            counts[f"selected:{mode}:{effective_capability}"] += 1
            if capability in DOCUMENT_SCOPED_CAPABILITIES:
                counts["selected:document_scoped"] += 1
            else:
                counts[
                    "selected:package_scoped"
                    if scope is not None
                    else "selected:package_unresolved"
                ] += 1
                for role in sorted(available_roles):
                    counts[f"selected:semantic_role:{role}"] += 1
            if len(selected) >= args.maximum_work:
                break
        if len(selected) >= args.maximum_work:
            break

    work_ids = [item["work_id"] for item in selected]
    if len(work_ids) != len(set(work_ids)):
        raise ValueError("structural queue contains duplicate work IDs")

    core = {
        "schema": "harness.electronics-structural-local-work.v1",
        "policy": {
            "text_first": False,
            "capabilities": sorted(selected_capabilities),
            "full_page_vision_for_definition_tables": True,
            "diagram_authority": "corroboration_only",
            "pin_locator": "toc_definition_table_only",
            "pin_work_source": args.pin_work_source,
            "pin_package_header_gate": "required",
            "pin_completion_gate": "merged_unique_ids_equal_exact_package_count",
            "package_scope": args.package_scope_policy,
            "parametrics_scope": "document_scoped",
            "opn_decoder_scope": "document_scoped",
            "series_summary_scope": "document_scoped",
            "required_semantic_roles": sorted(required_semantic_roles),
            "minimum_semantic_role_count": args.minimum_semantic_role_count,
            "frontier_teacher_for_all_bootstrap_outputs": True,
        },
        "sources": {
            "priority_queue_sha256": _sha256(priority_path),
            "page_evidence_manifest_sha256": _sha256(
                evidence_root / "manifest.json"
            ),
            "page_index_manifest_sha256": _sha256(
                index_root / "manifest.json"
            ),
            "excluded_work_queues": [
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for path in excluded_paths
            ],
        },
        "counts": {
            **dict(sorted(counts.items())),
            "selected": len(selected),
        },
        "work": selected,
    }
    core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
    value = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **core,
    }
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(value["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
