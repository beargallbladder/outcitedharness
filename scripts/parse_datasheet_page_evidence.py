#!/usr/bin/env python3
"""Parse deterministic facts and queue unresolved plus shadow-learning work."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.electronics.claims import canonical_json
from harness.electronics.corpus import sha256_file
from harness.electronics.holdout import HOLDOUT_SCHEMA
from harness.electronics.table_extractors import (
    parse_parametric_table,
    parse_pin_table,
)


BUNDLE_SCHEMA = "harness.electronics-deterministic-extraction.v1"


def _verify_evidence(value: dict[str, Any], schema: str, kind: str) -> None:
    if value.get("schema") != schema:
        raise ValueError(f"{kind} schema is not supported")
    expected = value.get("evidence_sha256")
    core = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "evidence_sha256"}
    }
    if hashlib.sha256(canonical_json(core)).hexdigest() != expected:
        raise ValueError(f"{kind} evidence digest is invalid")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-evidence", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument(
        "--shadow-all-model-lanes",
        action="store_true",
        help=(
            "Queue structurally relevant pages even when deterministic parsing "
            "succeeds, so local/frontier comparisons become training pairs."
        ),
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    evidence_root = args.page_evidence.expanduser().resolve(strict=True)
    evidence_manifest_path = evidence_root / "manifest.json"
    evidence_manifest = json.loads(
        evidence_manifest_path.read_text(encoding="utf-8")
    )
    page_path = evidence_root / "page-evidence.jsonl"
    page_receipt = evidence_manifest["artifacts"]["page-evidence.jsonl"]
    if (
        sha256_file(page_path) != page_receipt["sha256"]
        or page_path.stat().st_size != page_receipt["bytes"]
    ):
        raise ValueError("page evidence differs from its manifest")
    holdout_path = args.holdout.expanduser().resolve(strict=True)
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    _verify_evidence(holdout, HOLDOUT_SCHEMA, "holdout")
    reserved = {
        row["document_sha256"] for row in holdout["reserved_documents"]
    }
    output = args.output_directory.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    pin_path = temporary / "pin-rows.jsonl"
    parametric_path = temporary / "parametric-rows.jsonl"
    queue_path = temporary / "local-model-queue.jsonl"
    counts: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    seen_queue: set[tuple[str, int, str]] = set()
    try:
        with page_path.open(encoding="utf-8") as source, pin_path.open(
            "xb"
        ) as pin_handle, parametric_path.open("xb") as parametric_handle, queue_path.open(
            "xb"
        ) as queue_handle:
            for line_number, line in enumerate(source, 1):
                page = json.loads(line)
                document_sha = page["document_sha256"]
                page_number = int(page["page_1based"])
                lanes = set(page.get("lanes") or [])
                partition = (
                    "frozen_evaluation"
                    if document_sha in reserved
                    else "factory_candidate"
                )
                page_pin_rows: list[dict[str, Any]] = []
                page_parametric_rows: list[dict[str, Any]] = []
                for table in page.get("tables") or []:
                    if lanes & {"pin_or_ball", "pin_semantics"}:
                        page_pin_rows.extend(
                            parse_pin_table(
                                table,
                                document_sha256=document_sha,
                                page_1based=page_number,
                            )
                        )
                    if "parametrics" in lanes:
                        page_parametric_rows.extend(
                            parse_parametric_table(
                                table,
                                document_sha256=document_sha,
                                page_1based=page_number,
                            )
                        )
                for row in page_pin_rows:
                    row["partition"] = partition
                    row["page_evidence_sha256"] = page["evidence_sha256"]
                    pin_handle.write(canonical_json(row) + b"\n")
                for row in page_parametric_rows:
                    row["partition"] = partition
                    row["page_evidence_sha256"] = page["evidence_sha256"]
                    parametric_handle.write(canonical_json(row) + b"\n")
                counts["pages"] += 1
                counts["pin_rows"] += len(page_pin_rows)
                counts["parametric_rows"] += len(page_parametric_rows)

                queue_capabilities: set[str] = set()
                if lanes & {"pin_or_ball", "pin_semantics"} and (
                    args.shadow_all_model_lanes
                    or not page_pin_rows
                    or any(row["package"] == "UNRESOLVED" for row in page_pin_rows)
                ):
                    queue_capabilities.update({"pin_or_ball", "pin_semantics"})
                if "parametrics" in lanes:
                    queue_capabilities.add("parametrics")
                if "series_summary" in lanes:
                    queue_capabilities.add("series_summary")
                if "opn_decoder" in lanes:
                    queue_capabilities.add("opn_decoder")
                for capability in sorted(queue_capabilities):
                    identity = (document_sha, page_number, capability)
                    if identity in seen_queue:
                        continue
                    seen_queue.add(identity)
                    queue_row = {
                        "schema": "harness.electronics-local-model-work.v1",
                        "work_id": "local-" + hashlib.sha256(
                            canonical_json(identity)
                        ).hexdigest()[:32],
                        "document_sha256": document_sha,
                        "source_path": page["source_path"],
                        "page_1based": page_number,
                        "capability": capability,
                        "partition": partition,
                        "page_evidence_sha256": page["evidence_sha256"],
                        "deterministic_result": {
                            "pin_rows": len(page_pin_rows),
                            "parametric_rows": len(page_parametric_rows),
                        },
                        "frontier_batch_eligible": False,
                        "frontier_requires": [
                            "local_model_terminal_attempt_receipt",
                            "source_evidence_artifact",
                        ],
                    }
                    queue_handle.write(canonical_json(queue_row) + b"\n")
                    routes[f"{partition}:{capability}"] += 1
            for handle in (pin_handle, parametric_handle, queue_handle):
                handle.flush()
                os.fsync(handle.fileno())
        artifacts = {}
        for path in (pin_path, parametric_path, queue_path):
            artifacts[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        core = {
            "schema": BUNDLE_SCHEMA,
            "policy": {
                "ordering": [
                    "pymupdf_deterministic_table",
                    "local_model",
                    "anthropic_message_batch",
                ],
                "parametric_shadow_learning_on_parsed_tables": True,
                "pin_shadow_learning_on_parsed_tables": (
                    args.shadow_all_model_lanes
                ),
                "holdout_model_outputs_used_for_training": False,
                "frontier_batch_requires_local_failure": False,
                "direct_database_write": False,
            },
            "sources": {
                "page_evidence": {
                    "path": str(evidence_manifest_path),
                    "sha256": sha256_file(evidence_manifest_path),
                    "evidence_sha256": evidence_manifest["evidence_sha256"],
                },
                "holdout": {
                    "path": str(holdout_path),
                    "sha256": sha256_file(holdout_path),
                    "evidence_sha256": holdout["evidence_sha256"],
                },
            },
            "artifacts": artifacts,
            "counts": {
                **dict(sorted(counts.items())),
                "local_model_work": len(seen_queue),
                "routes": dict(sorted(routes.items())),
            },
        }
        core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **core,
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("xb") as handle:
            handle.write(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        for path in (pin_path, parametric_path, queue_path, manifest_path):
            os.chmod(path, 0o444)
        os.chmod(temporary, 0o555)
        os.rename(temporary, output)
    except BaseException:
        try:
            os.chmod(temporary, 0o755)
        except OSError:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
