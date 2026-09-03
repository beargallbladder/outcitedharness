"""Freeze document- and identity-safe datasheet factory holdouts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from harness.electronics.claims import canonical_json
from harness.electronics.corpus import (
    canonical_vendor,
    sha256_file,
    verify_corpus_registry,
)


HOLDOUT_SCHEMA = "harness.electronics-factory-holdout.v1"


def family_key(record_id: str) -> str:
    return re.sub(
        r"(?i)(?:-?q1|-(?:tr|reel|tape))$",
        "",
        record_id.strip(),
    ).casefold()


def _row_training_identities(root: Path) -> tuple[set[str], set[str], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest["artifacts"]["canonical/train.jsonl"]
    train_path = root / "canonical" / "train.jsonl"
    if (
        sha256_file(train_path) != artifact["sha256"]
        or train_path.stat().st_size != artifact["bytes"]
    ):
        raise ValueError("row training split differs from its manifest")
    lineages: set[str] = set()
    families: set[str] = set()
    with train_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            try:
                lineages.add(str(value["provenance"]["pdf_sha256"]))
                families.add(family_key(str(value["record_id"])))
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"invalid train row at line {line_number}"
                ) from exc
    receipt = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "evidence_sha256": manifest.get("evidence_sha256"),
    }
    return lineages, families, receipt


def _profiles(path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    if path is None:
        return {}, None
    root = path.expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    profiles_path = root / "profiles.jsonl"
    receipt = manifest["artifacts"]["profiles.jsonl"]
    if (
        sha256_file(profiles_path) != receipt["sha256"]
        or profiles_path.stat().st_size != receipt["bytes"]
    ):
        raise ValueError("page-index profiles differ from their manifest")
    output: dict[str, dict[str, Any]] = {}
    with profiles_path.open(encoding="utf-8") as handle:
        for line in handle:
            profile = json.loads(line)
            output[profile["document_sha256"]] = profile
    return output, {
        "path": str(root / "manifest.json"),
        "sha256": sha256_file(root / "manifest.json"),
        "evidence_sha256": manifest["evidence_sha256"],
    }


def _package_family(value: str) -> str:
    match = re.search(
        r"(?i)(TFBGA|UFBGA|LFBGA|FBGA|BGA|WLCSP|LQFP|TQFP|QFP|"
        r"VFQFPN|UFQFPN|VQFN|WQFN|QFN|DFN|TSSOP|SSOP|SOIC|PDIP)",
        value,
    )
    return match.group(1).upper() if match else "OTHER"


def freeze_factory_holdout(
    registry: Mapping[str, Any],
    *,
    row_dataset_root: Path,
    page_index_root: Path | None,
    fraction: float,
    minimum_documents: int,
    maximum_documents: int,
    temporal_cutoff: datetime,
) -> dict[str, Any]:
    verify_corpus_registry(registry)
    if not 0 < fraction < 0.5:
        raise ValueError("holdout fraction must be within (0, 0.5)")
    if not 1 <= minimum_documents <= maximum_documents:
        raise ValueError("invalid holdout document limits")
    train_lineages, train_families, row_receipt = _row_training_identities(
        row_dataset_root.expanduser().resolve(strict=True)
    )
    profiles, profile_receipt = _profiles(page_index_root)
    gt_root = Path(registry["sources"]["ground_truth_root"]).resolve()
    candidates: list[dict[str, Any]] = []
    excluded = Counter()
    for document in registry["documents"]:
        records = document.get("ground_truth") or []
        if not records:
            continue
        digest = document["document_sha256"]
        families = {family_key(record["record_id"]) for record in records}
        if digest in train_lineages:
            excluded["row_train_lineage"] += 1
            continue
        if families & train_families:
            excluded["row_train_family"] += 1
            continue
        packages: set[str] = set()
        for record in records:
            path = (gt_root / record["path"]).resolve()
            if not path.is_relative_to(gt_root):
                raise ValueError("ground-truth path escapes configured root")
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"ground truth changed after corpus seal: {path}")
            value = json.loads(path.read_text(encoding="utf-8"))
            pinout = value.get("pinout") if isinstance(value, dict) else None
            if isinstance(pinout, Mapping):
                packages.update(
                    str(package)
                    for package in pinout.get("packages") or []
                    if str(package).strip()
                )
        profile = profiles.get(digest, {})
        page_count = profile.get("page_count")
        layout_bucket = (
            f"{(int(page_count) // 100) * 100}-{(int(page_count) // 100) * 100 + 99}"
            if isinstance(page_count, int)
            else "unknown"
        )
        package_families = sorted({_package_family(value) for value in packages})
        record_vendors = sorted(
            {
                str(record["vendor"]).casefold()
                for record in records
                if record.get("vendor")
            }
        )
        vendors = sorted(
            {
                canonical_vendor(value) or "unknown"
                for value in (
                    record_vendors
                    or list(document.get("vendors") or ["unknown"])
                )
            }
        )
        vendor = str(vendors[0]).casefold()
        candidates.append(
            {
                "document_sha256": digest,
                "paths": document.get("paths") or [],
                "vendors": vendors,
                "record_ids": sorted(
                    record["record_id"] for record in records
                ),
                "family_keys": sorted(families),
                "package_families": package_families or ["UNKNOWN"],
                "layout_bucket": layout_bucket,
                "lane_coverage": sorted(
                    lane
                    for lane, pages in (profile.get("lane_pages") or {}).items()
                    if pages
                ),
                "stratum": f"{vendor}:{(package_families or ['UNKNOWN'])[0]}",
            }
        )

    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_stratum[candidate["stratum"]].append(candidate)
    selected: dict[str, dict[str, Any]] = {}
    for stratum, rows in sorted(by_stratum.items()):
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"electronics-holdout-v1:{row['document_sha256']}".encode()
            ).hexdigest(),
        )
        count = max(1, math.ceil(len(ranked) * fraction))
        for row in ranked[:count]:
            selected[row["document_sha256"]] = row
    globally_ranked = sorted(
        candidates,
        key=lambda row: hashlib.sha256(
            f"electronics-holdout-fill-v1:{row['document_sha256']}".encode()
        ).hexdigest(),
    )
    for row in globally_ranked:
        if len(selected) >= minimum_documents:
            break
        selected[row["document_sha256"]] = row
    if len(selected) < minimum_documents:
        raise ValueError(
            f"only {len(selected)} holdout documents are eligible; "
            f"{minimum_documents} required"
        )
    if len(selected) > maximum_documents:
        selected = {
            row["document_sha256"]: row
            for row in sorted(
                selected.values(),
                key=lambda row: hashlib.sha256(
                    f"electronics-holdout-cap-v1:{row['document_sha256']}".encode()
                ).hexdigest(),
            )[:maximum_documents]
        }
    rows = sorted(selected.values(), key=lambda row: row["document_sha256"])
    selected_families = {
        family for row in rows for family in row["family_keys"]
    }
    if selected_families & train_families:
        raise RuntimeError("holdout contains a row-training family")
    vendor_counts = Counter(
        vendor for row in rows for vendor in row["vendors"]
    )
    package_counts = Counter(
        family for row in rows for family in row["package_families"]
    )
    layout_counts = Counter(row["layout_bucket"] for row in rows)
    core = {
        "schema": HOLDOUT_SCHEMA,
        "purpose": "frozen_local_datasheet_capability_evaluation",
        "policy": {
            "selection_fraction": fraction,
            "minimum_documents": minimum_documents,
            "maximum_documents": maximum_documents,
            "unit": "document_sha256",
            "family_overlap_with_row_training_allowed": False,
            "document_overlap_with_row_training_allowed": False,
            "future_training_use": "prohibited",
            "temporal_holdout_cutoff": temporal_cutoff.isoformat(),
            "future_documents_after_cutoff": "temporal_evaluation_candidates",
            "algorithm": (
                "exclude row-training lineages and families; stratify by vendor "
                "and package family; rank by frozen SHA-256"
            ),
        },
        "sources": {
            "corpus_evidence_sha256": registry["evidence_sha256"],
            "row_dataset": row_receipt,
            "page_index": profile_receipt,
        },
        "counts": {
            "eligible_documents": len(candidates),
            "reserved_documents": len(rows),
            "reserved_record_ids": len(
                {record for row in rows for record in row["record_ids"]}
            ),
            "excluded": dict(sorted(excluded.items())),
            "vendors": dict(sorted(vendor_counts.items())),
            "package_families": dict(sorted(package_counts.items())),
            "layout_buckets": dict(sorted(layout_counts.items())),
        },
        "reserved_documents": rows,
    }
    core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
    return core


__all__ = [
    "HOLDOUT_SCHEMA",
    "family_key",
    "freeze_factory_holdout",
]
