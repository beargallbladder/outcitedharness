#!/usr/bin/env python3
"""Freeze source-grounded labels for package/document-safe extraction evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.electronics.claims import canonical_json
from harness.electronics.corpus import verify_corpus_registry
from harness.electronics.ground_truth import (
    load_ground_truth_records,
    rows_for_package,
)
from harness.electronics.local_model import focused_page_context
from harness.electronics.table_extractors import (
    normalize_parametric_facts,
    parse_parametric_table,
    pin_identity_rows,
    pin_semantic_values,
)


def _string(value: Any) -> str:
    return str("" if value is None else value)


def _normalized(value: Any) -> str:
    return "".join(
        character
        for character in _string(value).upper()
        if character.isalnum()
    )


def _source_rows(
    capability: str,
    page: dict[str, Any],
) -> list[
    tuple[
        str,
        tuple[str, ...],
        tuple[str, ...],
        str,
        dict[str, Any],
        int,
    ]
]:
    context = focused_page_context(capability, page)
    output = []
    for table in context.get("tables") or []:
        for row_index, row in enumerate(table.get("rows") or []):
            if not isinstance(row, list):
                continue
            rendered = [" ".join(_string(cell).split()) for cell in row]
            cells = tuple(_normalized(cell) for cell in rendered if cell)
            tokens = tuple(
                token
                for cell in rendered
                for token in (
                    _normalized(part)
                    for part in re.split(r"[,;\s]+", cell)
                )
                if token
            )
            output.append(
                (
                    "".join(cells),
                    cells,
                    tokens,
                    " | ".join(rendered),
                    table,
                    row_index,
                )
            )
    return output


def _source_identities(
    capability: str,
    page: dict[str, Any],
) -> list[dict[str, Any]]:
    context = focused_page_context(capability, page)
    output = []
    for table in context.get("tables") or []:
        output.extend(pin_identity_rows(table))
    return output


def _grounded_label(
    capability: str,
    row: dict[str, Any],
    source_rows: list[
        tuple[
            str,
            tuple[str, ...],
            tuple[str, ...],
            str,
            dict[str, Any],
            int,
        ]
    ],
) -> tuple[dict[str, Any], str] | None:
    number = _normalized(row.get("pin_no"))
    name = _normalized(row.get("name"))
    matches = [
        source
        for source in source_rows
        if name in source[1] and (
            number in source[1] or number in source[2]
        )
    ]
    if not number or not name or not matches:
        return None
    _source_text, _source_cells, _tokens, quote, table, row_index = matches[0]
    label: dict[str, Any] = {
        "pin_no": row["pin_no"],
        "name": row["name"],
    }
    if capability == "pin_semantics":
        semantics = pin_semantic_values(
            table,
            row_index,
            pin_no=row["pin_no"],
            name=row["name"],
        )
        label["type"] = next(iter(semantics["type"]), None)
        label["dir"] = next(iter(semantics["dir"]), None)
        label["supply_domain"] = next(
            iter(semantics["supply_domain"]),
            None,
        )
        function_sources = [
            _normalized(value) for value in semantics["functions"]
        ]
        label["functions"] = [
            value
            for value in row.get("functions") or []
            if _normalized(value)
            and any(
                _normalized(value) in source
                for source in function_sources
            )
        ]
    return label, quote


def _parametric_identity(fact: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _normalized(fact.get("field")),
        _normalized(fact.get("value")),
        _normalized(fact.get("value_role")),
        _normalized(fact.get("unit")),
    )


def _parametric_labels(
    item: dict[str, Any],
    page: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    context = focused_page_context("parametrics", page)
    expected: list[dict[str, Any]] = []
    quotes: list[str] = []
    seen: set[tuple[str, str, str, str]] = set()
    for table in context.get("tables") or []:
        rows = table.get("rows") or []
        for parsed in parse_parametric_table(
            table,
            document_sha256=item["document_sha256"],
            page_1based=int(item["page_1based"]),
        ):
            row_index = int(parsed["row_index"])
            if row_index >= len(rows) or not isinstance(rows[row_index], list):
                continue
            quote = " | ".join(
                " ".join(_string(cell).split()) for cell in rows[row_index]
            )
            for fact in normalize_parametric_facts(parsed):
                identity = _parametric_identity(fact)
                if not all(identity[:3]) or identity in seen:
                    continue
                seen.add(identity)
                expected.append(fact)
                quotes.append(quote)
    return expected, quotes


def _verified_core(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "evidence_sha256"}
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-queue", type=Path, required=True)
    parser.add_argument("--candidate-queue", type=Path, required=True)
    parser.add_argument("--page-evidence", type=Path, required=True)
    parser.add_argument("--corpus-registry", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.structural_queue.expanduser().resolve(strict=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("schema") != "harness.electronics-structural-local-work.v1":
        raise ValueError("unsupported structural queue")
    if source.get("evidence_sha256") != hashlib.sha256(
        canonical_json(_verified_core(source))
    ).hexdigest():
        raise ValueError("structural queue digest is invalid")
    if source.get("policy", {}).get("package_scope") != "require":
        raise ValueError("evaluation requires package-scoped work")
    if any(
        item.get("partition") != "frozen_evaluation"
        for item in source["work"]
    ):
        raise ValueError("evaluation queue contains a non-holdout partition")
    candidate_path = args.candidate_queue.expanduser().resolve(strict=True)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_documents = {
        item["document_sha256"] for item in candidate["work"]
    }
    if any(
        item["document_sha256"] in candidate_documents
        for item in source["work"]
    ):
        raise ValueError("evaluation work overlaps candidate documents")

    evidence_root = args.page_evidence.expanduser().resolve(strict=True)
    wanted = {
        (item["document_sha256"], int(item["page_1based"]))
        for item in source["work"]
    }
    pages = {}
    with (evidence_root / "page-evidence.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            page = json.loads(line)
            key = (page["document_sha256"], int(page["page_1based"]))
            if key in wanted:
                pages[key] = page
    corpus_path = args.corpus_registry.expanduser().resolve(strict=True)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    verify_corpus_registry(corpus)
    ground_truth = load_ground_truth_records(
        corpus,
        args.ground_truth_root.expanduser().resolve(strict=True),
    )

    selected = []
    labels = []
    for item in source["work"]:
        page = {
            **pages[(item["document_sha256"], int(item["page_1based"]))],
            "structural_evidence": item["structural_evidence"],
        }
        capability = item["capability"]
        if capability == "parametrics":
            expected, quotes = _parametric_labels(item, page)
            if not expected:
                continue
            selected.append(item)
            labels.append(
                {
                    "schema": (
                        "harness.electronics-extraction-evaluation-label.v1"
                    ),
                    "work_id": item["work_id"],
                    "document_sha256": item["document_sha256"],
                    "page_1based": item["page_1based"],
                    "capability": capability,
                    "package": None,
                    "expected_facts": expected,
                    "quoted_source_rows": quotes,
                }
            )
            continue

        scope = item["structural_evidence"]["package_scope"]
        if (
            not isinstance(scope, dict)
            or not scope.get("package")
            or not scope.get("column_header")
            or not isinstance(scope.get("expected_package_pins"), int)
            or scope["expected_package_pins"] < 1
        ):
            continue
        package = scope["package"]
        gt_rows = rows_for_package(
            ground_truth.get(item["document_sha256"], []),
            package,
        )
        if not gt_rows:
            continue
        source_rows = _source_rows(capability, page)
        source_identities = _source_identities(capability, page)
        gt_by_name: dict[str, list[dict[str, Any]]] = {}
        for row in gt_rows:
            gt_by_name.setdefault(_normalized(row.get("name")), []).append(row)
        expected_pins = []
        quotes = []
        seen = set()
        for source_identity in source_identities:
            matching_gt = gt_by_name.get(
                _normalized(source_identity.get("name")),
                [],
            )
            if not matching_gt:
                continue
            row = {
                **matching_gt[0],
                "pin_no": source_identity["pin_no"],
                "name": source_identity["name"],
            }
            grounded = _grounded_label(capability, row, source_rows)
            if grounded is None:
                continue
            label, quote = grounded
            identity = (_normalized(label["pin_no"]), _normalized(label["name"]))
            if identity in seen:
                continue
            seen.add(identity)
            expected_pins.append(label)
            quotes.append(quote)
        if not expected_pins:
            continue
        selected.append(item)
        labels.append(
            {
                "schema": "harness.electronics-extraction-evaluation-label.v1",
                "work_id": item["work_id"],
                "document_sha256": item["document_sha256"],
                "page_1based": item["page_1based"],
                "capability": capability,
                "package": package,
                "expected_pins": expected_pins,
                "quoted_source_rows": quotes,
            }
        )
    if not selected:
        raise ValueError("no source-grounded evaluation labels were resolved")

    output = args.output_directory.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        queue_core = {
            "schema": "harness.electronics-structural-local-work.v1",
            "policy": {
                **source["policy"],
                "evaluation_only": True,
                "frontier_teacher_for_all_bootstrap_outputs": False,
                "training_admission": False,
            },
            "sources": {
                **source["sources"],
                "source_structural_queue_sha256": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "candidate_queue_sha256": hashlib.sha256(
                    candidate_path.read_bytes()
                ).hexdigest(),
                "corpus_registry_sha256": hashlib.sha256(
                    corpus_path.read_bytes()
                ).hexdigest(),
                "corpus_evidence_sha256": corpus["evidence_sha256"],
                "label_generation_code_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            "counts": {
                "selected": len(selected),
                "documents": len(
                    {item["document_sha256"] for item in selected}
                ),
                "expected_pin_rows": sum(
                    len(label.get("expected_pins") or []) for label in labels
                ),
                "expected_parametric_facts": sum(
                    len(label.get("expected_facts") or [])
                    for label in labels
                ),
            },
            "work": selected,
        }
        queue_core["evidence_sha256"] = hashlib.sha256(
            canonical_json(queue_core)
        ).hexdigest()
        queue_value = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **queue_core,
        }
        payloads = {
            "work-queue.json": (
                json.dumps(
                    queue_value,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ).encode()
                + b"\n"
            ),
            "labels.jsonl": b"".join(
                canonical_json(label) + b"\n" for label in labels
            ),
        }
        artifacts = {}
        for name, payload in payloads.items():
            path = temporary / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(path, 0o444)
            artifacts[name] = {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        manifest_core = {
            "schema": "harness.electronics-extraction-evaluation-cohort.v1",
            "purpose": "frozen_evaluation_only",
            "artifacts": artifacts,
            "counts": queue_core["counts"],
            "policy": {
                "document_overlap_with_candidate": 0,
                "labels_exposed_to_model": False,
                "training_admission": False,
                "source_row_grounding_required": True,
            },
        }
        manifest_core["evidence_sha256"] = hashlib.sha256(
            canonical_json(manifest_core)
        ).hexdigest()
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **manifest_core,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(manifest_path, 0o444)
        os.chmod(temporary, 0o555)
        os.rename(temporary, output)
    except BaseException:
        os.chmod(temporary, 0o755)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
