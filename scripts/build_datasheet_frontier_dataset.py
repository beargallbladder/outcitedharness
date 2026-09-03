#!/usr/bin/env python3
"""Build an immutable vision dataset from admitted frontier comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.config import load_config
from harness.storage.db import Store
from harness.training.ledger import LearningLedger
from harness.training.models import (
    DataUse,
    FactValue,
    SourceKind,
    SourceProvenance,
    VisionPair,
)


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from compare_datasheet_frontier import POLICY_VERSION  # noqa: E402
from evaluate_datasheet_modalities import load_fixture, sha256_file  # noqa: E402


SCHEMA = "harness.dataset.datasheet-frontier.v1"
DATASET_NAME = "datasheet_frontier_vision_train"
REQUIRED_ARTIFACTS = {
    "comparison",
    "frontier_prediction",
    "rendered_page_rows",
    "training_prompt",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _load_artifact_json(ledger: LearningLedger, digest: str) -> dict[str, Any]:
    value = json.loads(ledger.vault.path_for(digest).read_text())
    if not isinstance(value, dict):
        raise ValueError("learning artifact must contain a JSON object")
    return value


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.chmod(path, 0o444)


def _write_json(path: Path, value: Any) -> None:
    _write(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    payload = b"".join(_canonical(value) + b"\n" for value in values)
    _write(path, payload)


def _admitted_rows(store: Store) -> list[Any]:
    with store.connect() as connection:
        return connection.execute(
            """
            SELECT
                e.*,
                a.admission_id,
                a.verification_id AS admission_verification_id,
                a.policy_version,
                a.admission_sha256
            FROM learning_events AS e
            JOIN learning_admissions AS a ON a.event_id = e.event_id
            WHERE e.event_type = 'datasheet_frontier_vision_comparison'
              AND a.decision = 'eligible'
              AND a.policy_version = ?
            ORDER BY e.lineage_id, e.event_id
            """,
            (POLICY_VERSION,),
        ).fetchall()


def build_dataset(
    *,
    destination: Path,
    frozen_fixtures: list[Path],
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"destination already exists: {destination}")
    if not frozen_fixtures:
        raise ValueError("at least one frozen fixture is required")

    frozen_pdf_digests: set[str] = set()
    frozen_fixture_digests: list[str] = []
    for path in frozen_fixtures:
        manifest, _ = load_fixture(path)
        frozen_pdf_digests.update(
            str(case["pdf_sha256"]) for case in manifest["cases"]
        )
        frozen_fixture_digests.append(sha256_file(path.resolve()))

    config = load_config()
    store = Store(config.settings.db_path)
    ledger = LearningLedger(store, config.settings.learning_artifact_root)
    rows = _admitted_rows(store)
    if not rows:
        raise ValueError(f"no admitted events satisfy {POLICY_VERSION}")

    pairs: list[VisionPair] = []
    images: dict[str, bytes] = {}
    source_events: list[dict[str, Any]] = []
    source_pages: set[str] = set()
    training_lineages: set[str] = set()
    for row in rows:
        metadata = json.loads(row["metadata_json"])
        if (
            metadata.get("data_use") != "training"
            or metadata.get("disposition") != "verified"
            or metadata.get("training_eligible") is not True
            or metadata.get("verification_policy") != POLICY_VERSION
        ):
            raise ValueError(f"{row['event_id']}: invalid training admission metadata")
        for field in (
            "frontier_model",
            "input_sha256",
            "local_evaluation_sha256",
            "local_model",
            "local_model_manifest_sha256",
            "local_runtime_image_id",
            "local_runtime_version",
        ):
            if not isinstance(metadata.get(field), str) or not metadata[field]:
                raise ValueError(
                    f"{row['event_id']}: missing provenance field {field}"
                )
        if row["source_revision"] in frozen_pdf_digests:
            raise ValueError(f"{row['event_id']}: frozen lineage cannot enter training")
        if row["source_uri"] in source_pages:
            raise ValueError(f"{row['event_id']}: duplicate admitted source page")
        source_pages.add(row["source_uri"])
        training_lineage = f"datasheet:{row['source_revision']}"
        training_lineages.add(training_lineage)

        capture = ledger.verify_event(row["event_id"])
        artifacts = {artifact.kind: artifact for artifact in capture.artifacts}
        missing = REQUIRED_ARTIFACTS - artifacts.keys()
        if missing:
            raise ValueError(
                f"{row['event_id']}: missing artifacts {sorted(missing)}"
            )
        comparison = _load_artifact_json(
            ledger,
            artifacts["comparison"].sha256,
        )
        if (
            comparison.get("independent_three_way_consensus") is not True
            or comparison.get("training_eligible") is not True
        ):
            raise ValueError(f"{row['event_id']}: comparison is not trainable")
        frontier = _load_artifact_json(
            ledger,
            artifacts["frontier_prediction"].sha256,
        )
        pins = frontier.get("pins")
        if not isinstance(pins, list) or not pins:
            raise ValueError(f"{row['event_id']}: frontier target has no pins")
        prompt = ledger.vault.path_for(
            artifacts["training_prompt"].sha256
        ).read_text()
        image_artifact = artifacts["rendered_page_rows"]
        image = ledger.vault.path_for(image_artifact.sha256).read_bytes()
        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"{row['event_id']}: image artifact is not PNG")
        images.setdefault(image_artifact.sha256, image)

        provenance = SourceProvenance(
            source_kind=SourceKind.OTHER,
            source_uri=row["source_uri"],
            source_record_id=row["event_id"],
            collected_at=row["created_at"],
            content_sha256=row["event_sha256"],
            lineage_id=training_lineage,
            license="public-vendor-datasheet",
            revision=row["source_revision"],
            mutable_facts=False,
            data_use=DataUse.TRAINING,
        )
        pairs.append(
            VisionPair(
                pair_id=f"pair-{row['event_id']}",
                prompt=prompt,
                response=json.dumps(
                    {"pins": pins},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                provenance=provenance,
                label=FactValue.POSITIVE,
                data_use=DataUse.TRAINING,
                image_uris=(
                    f"dataset://datasheet-frontier/images/{image_artifact.sha256}.png",
                ),
                image_sha256=(image_artifact.sha256,),
                metadata={
                    "learning_event_id": row["event_id"],
                    "admission_id": row["admission_id"],
                    "admission_sha256": row["admission_sha256"],
                    "verification_id": row["admission_verification_id"],
                    "verification_policy": row["policy_version"],
                    "frontier_model": metadata["frontier_model"],
                    "local_model": metadata["local_model"],
                    "local_evaluation_sha256": metadata[
                        "local_evaluation_sha256"
                    ],
                },
            )
        )
        source_events.append(
            {
                "event_id": row["event_id"],
                "event_sha256": row["event_sha256"],
                "admission_id": row["admission_id"],
                "admission_sha256": row["admission_sha256"],
                "source_lineage_id": row["lineage_id"],
                "training_lineage_id": training_lineage,
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        for digest, image in sorted(images.items()):
            _write(temporary / "images" / f"{digest}.png", image)
        canonical_rows = [
            pair.model_dump(mode="json", exclude_none=True) for pair in pairs
        ]
        _write_jsonl(
            temporary / "canonical" / "vision" / "train.jsonl",
            canonical_rows,
        )
        llamafactory_rows = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "\n".join("<image>" for _ in pair.image_uris)
                            + "\n"
                            + pair.prompt
                        ),
                    },
                    {"role": "assistant", "content": pair.response},
                ],
                "images": [
                    "../images/" + uri.rsplit("/", 1)[-1]
                    for uri in pair.image_uris
                ],
            }
            for pair in pairs
        ]
        _write_json(
            temporary / "llamafactory" / f"{DATASET_NAME}.json",
            llamafactory_rows,
        )
        _write_json(
            temporary / "llamafactory" / "dataset_info.json",
            {
                DATASET_NAME: {
                    "file_name": f"{DATASET_NAME}.json",
                    "formatting": "sharegpt",
                    "columns": {
                        "messages": "messages",
                        "images": "images",
                    },
                    "tags": {
                        "role_tag": "role",
                        "content_tag": "content",
                        "user_tag": "user",
                        "assistant_tag": "assistant",
                    },
                }
            },
        )
        generated = sorted(
            path
            for path in temporary.rglob("*")
            if path.is_file()
        )
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "admission_policy": POLICY_VERSION,
            "counts": {
                "train_pairs": len(pairs),
                "lineages": len(training_lineages),
                "images": len(images),
            },
            "frozen_fixture_sha256": sorted(frozen_fixture_digests),
            "source_events": source_events,
            "artifacts": {
                path.relative_to(temporary).as_posix(): {
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in generated
            },
        }
        manifest["core_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--frozen-fixture",
        required=True,
        action="append",
        dest="frozen_fixtures",
        type=Path,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_dataset(
        destination=args.destination,
        frozen_fixtures=args.frozen_fixtures,
    )
    print(
        json.dumps(
            {
                "destination": str(args.destination),
                **manifest["counts"],
                "core_sha256": manifest["core_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
