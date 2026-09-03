#!/usr/bin/env python3
"""Build and register lineage-safe datasets from executable code curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.storage.db import Store
from harness.training.code_curriculum import ADMISSION_POLICY
from harness.training.ledger import LearningLedger
from harness.training.queue import DatasetMember, DatasetVersionRegistry
from harness.training.security import assert_no_secrets
from harness.training.split import Split, assert_no_lineage_leakage, grouped_lineage_split


SCHEMA = "harness.dataset.owned-code-curriculum.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o444)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _write(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    _write(path, b"".join(_canonical(value) + b"\n" for value in values))


def _artifact_text(
    ledger: LearningLedger,
    artifacts: dict[str, Any],
    kind: str,
) -> str:
    artifact = artifacts.get(kind)
    if artifact is None or not artifact.redacted:
        raise ValueError(f"curriculum event lacks redacted {kind}")
    data = ledger.vault.path_for(artifact.sha256).read_bytes()
    if len(data) != artifact.byte_size or _sha256(data) != artifact.sha256:
        raise ValueError(f"curriculum artifact changed: {kind}")
    return data.decode("utf-8")


def _load_records(store: Store, artifact_root: Path) -> list[dict[str, Any]]:
    ledger = LearningLedger(store, artifact_root)
    with store.connect() as connection:
        rows = connection.execute(
            """
            SELECT
                event.*,
                admission.admission_id,
                admission.admission_sha256,
                admission.policy_version,
                admission.decision AS admission_decision
            FROM learning_events AS event
            JOIN learning_admissions AS admission
              ON admission.event_id = event.event_id
            WHERE admission.decision = 'eligible'
              AND admission.policy_version = ?
              AND event.event_type = 'coding_executable_curriculum'
            ORDER BY event.event_id
            """,
            (ADMISSION_POLICY,),
        ).fetchall()
    if not rows:
        raise ValueError("no admitted executable curriculum events are available")

    records = []
    for row in rows:
        capture = ledger.verify_event(row["event_id"])
        if len(capture.verifications) != 1:
            raise ValueError("curriculum event must contain one verification")
        proof = capture.verifications[0]
        if (
            proof.status != "pass"
            or proof.kind != "executable_mutation_fail_before_canonical_pass"
            or row["admission_decision"] != "eligible"
        ):
            raise ValueError("curriculum event lacks executable proof")
        metadata = json.loads(row["metadata_json"])
        if (
            metadata.get("content_class") != "owned_source_code"
            or metadata.get("owner_attested") is not True
            or metadata.get("data_paths_excluded") is not True
            or metadata.get("data_use") != "training"
            or metadata.get("disposition") != "verified"
        ):
            raise ValueError("curriculum ownership boundary is incomplete")
        artifacts = {artifact.kind: artifact for artifact in capture.artifacts}
        prompt = _artifact_text(ledger, artifacts, "coding_prompt")
        patch = _artifact_text(ledger, artifacts, "coding_chosen_patch")
        mutant_source = _artifact_text(ledger, artifacts, "coding_mutant_source")
        verification = json.loads(
            _artifact_text(ledger, artifacts, "coding_verification")
        )
        if (
            verification.get("baseline_returncode") != 0
            or verification.get("mutant_returncode") == 0
            or not patch.startswith("diff --git ")
        ):
            raise ValueError("curriculum proof does not show fail-before/pass-after")
        source_file_sha256 = str(metadata.get("source_file_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", source_file_sha256):
            raise ValueError("curriculum source file digest is missing")
        records.append(
            {
                "event_id": row["event_id"],
                "event_sha256": capture.event_sha256,
                "admission_id": row["admission_id"],
                "admission_sha256": row["admission_sha256"],
                "artifact_id": artifacts["coding_chosen_patch"].artifact_id,
                "patch_sha256": artifacts["coding_chosen_patch"].sha256,
                "repository_id": str(metadata["repository_id"]),
                "source_revision": str(row["source_revision"]),
                "source_file_sha256": source_file_sha256,
                "lineage_id": str(row["lineage_id"]),
                "component_family": _component(str(metadata["source_path"])),
                "mutation_operator": str(metadata["mutation_operator"]),
                "observed_at": datetime.fromisoformat(str(row["created_at"])),
                "prompt": prompt,
                "patch": patch,
                "mutant_source": mutant_source,
            }
        )
    return records


def _component(path: str) -> str:
    parts = Path(path).parts
    return "/".join(parts[:2]) if len(parts) > 1 else parts[0]


def build(
    *,
    store: Store,
    artifact_root: Path,
    destination: Path,
    max_prompt_chars: int = 28_000,
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"destination already exists: {destination}")
    if max_prompt_chars < 1:
        raise ValueError("maximum prompt characters must be positive")
    available_records = _load_records(store, artifact_root)
    records = [
        record
        for record in available_records
        if len(record["prompt"]) <= max_prompt_chars
    ]
    if not records:
        raise ValueError("no curriculum record fits the prompt length ceiling")
    partitions = grouped_lineage_split(
        records,
        lineage_key="source_file_sha256",
        seed="owned-code-curriculum-v1",
    )
    assert_no_lineage_leakage(partitions, lineage_key="lineage_id")
    if any(not partitions[split] for split in Split):
        raise ValueError("curriculum requires non-empty train, validation, and test splits")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        generated: list[Path] = []
        membership = []
        source_events = []
        dataset_info = {}
        for split in Split:
            canonical_rows = []
            llama_rows = []
            for record in sorted(
                partitions[split],
                key=lambda value: (value["lineage_id"], value["event_id"]),
            ):
                canonical_rows.append(
                    {
                        "pair_id": f"sft-{record['event_id']}",
                        "event_id": record["event_id"],
                        "lineage_id": record["lineage_id"],
                        "repository_id": record["repository_id"],
                        "source_revision": record["source_revision"],
                        "source_file_sha256": record["source_file_sha256"],
                        "mutation_operator": record["mutation_operator"],
                        "prompt": record["prompt"],
                        "response": record["patch"],
                        "mutant_source": record["mutant_source"],
                    }
                )
                llama_rows.append(
                    {
                        "messages": [
                            {"role": "user", "content": record["prompt"]},
                            {"role": "assistant", "content": record["patch"]},
                        ]
                    }
                )
                membership.append(
                    {
                        "event_id": record["event_id"],
                        "artifact_id": record["artifact_id"],
                        "split": split.value,
                        "lineage_id": record["lineage_id"],
                        "repository_id": record["repository_id"],
                        "source_revision": record["source_revision"],
                        "source_file_sha256": record["source_file_sha256"],
                        "component_family": record["component_family"],
                        "temporal_bucket": record["observed_at"].strftime("%Y-%m"),
                    }
                )
            canonical_path = temporary / "canonical" / f"{split.value}.jsonl"
            llama_path = (
                temporary / "llamafactory" / f"coding_sft_{split.value}.json"
            )
            _write_jsonl(canonical_path, canonical_rows)
            _write_json(llama_path, llama_rows)
            generated.extend((canonical_path, llama_path))
            dataset_info[f"coding_sft_{split.value}"] = {
                "file_name": llama_path.name,
                "formatting": "sharegpt",
                "columns": {"messages": "messages"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            }
        info_path = temporary / "llamafactory" / "dataset_info.json"
        _write_json(info_path, dataset_info)
        generated.append(info_path)
        for record in records:
            source_events.append(
                {
                    "event_id": record["event_id"],
                    "event_sha256": record["event_sha256"],
                    "admission_id": record["admission_id"],
                    "admission_sha256": record["admission_sha256"],
                    "chosen_patch_sha256": record["patch_sha256"],
                }
            )
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "created_at": max(record["observed_at"] for record in records).isoformat(),
            "admission_policy": ADMISSION_POLICY,
            "split_policy": {
                "kind": "source_file_lineage_hash",
                "seed": "owned-code-curriculum-v1",
                "leakage_keys": ["lineage_id", "source_document_sha256"],
            },
            "source_events": sorted(source_events, key=lambda row: row["event_id"]),
            "membership": sorted(
                membership,
                key=lambda row: (row["split"], row["lineage_id"], row["event_id"]),
            ),
            "counts": {
                split.value: len(partitions[split])
                for split in Split
            },
            "prompt_length_policy": {
                "max_chars": max_prompt_chars,
                "excluded_records": len(available_records) - len(records),
            },
            "artifacts": {
                path.relative_to(temporary).as_posix(): {
                    "sha256": _file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(generated)
            },
        }
        manifest["core_sha256"] = _sha256(_canonical(manifest))
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("curriculum manifest must be a regular file")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported curriculum dataset manifest")
    expected = value.get("core_sha256")
    core = {key: item for key, item in value.items() if key != "core_sha256"}
    if expected != _sha256(_canonical(core)):
        raise ValueError("curriculum manifest digest mismatch")
    root = path.parent
    for relative, receipt in value.get("artifacts", {}).items():
        candidate = (root / relative).resolve()
        if (
            not candidate.is_relative_to(root.resolve())
            or candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size != receipt.get("bytes")
            or _file_sha256(candidate) != receipt.get("sha256")
        ):
            raise ValueError(f"curriculum dataset artifact changed: {relative}")
    return value


def _load_import_report(path: Path | None) -> dict[str, tuple[str, str]]:
    if path is None:
        return {}
    value = json.loads(path.read_text())
    if value.get("schema") != "harness.learning-transfer-import.v1":
        raise ValueError("unsupported learning import report")
    expected = value.get("result_sha256")
    core = {key: item for key, item in value.items() if key != "result_sha256"}
    if expected != _sha256(_canonical(core)):
        raise ValueError("learning import report digest mismatch")
    output = {}
    for row in value.get("events", []):
        output[str(row["event_id"])] = (
            str(row["source_admission_sha256"]),
            str(row["destination_admission_sha256"]),
        )
    return output


def register(
    *,
    store: Store,
    manifest: dict[str, Any],
    dataset_version_id: str,
    version: str,
    sequence_audit: dict[str, Any],
    sequence_audit_sha256: str,
    import_report: dict[str, tuple[str, str]] | None = None,
) -> str:
    source_by_event = {
        row["event_id"]: row for row in manifest["source_events"]
    }
    imported = import_report or {}
    members = []
    with store.connect() as connection:
        for row in manifest["membership"]:
            source = source_by_event[row["event_id"]]
            current = connection.execute(
                """
                SELECT
                    event.event_sha256,
                    event.metadata_json,
                    artifact.artifact_id,
                    artifact.sha256,
                    admission.admission_sha256,
                    admission.policy_version,
                    admission.decision,
                    verification.status AS verification_status
                FROM learning_events AS event
                JOIN learning_artifacts AS artifact
                  ON artifact.event_id = event.event_id
                 AND artifact.kind = 'coding_chosen_patch'
                JOIN learning_admissions AS admission
                  ON admission.event_id = event.event_id
                JOIN learning_verifications AS verification
                  ON verification.verification_id = admission.verification_id
                WHERE event.event_id = ?
                """,
                (row["event_id"],),
            ).fetchone()
            accepted_admissions = {source["admission_sha256"]}
            if row["event_id"] in imported:
                source_sha, destination_sha = imported[row["event_id"]]
                if source_sha != source["admission_sha256"]:
                    raise ValueError("source admission differs from import report")
                accepted_admissions.add(destination_sha)
            metadata = json.loads(current["metadata_json"]) if current else {}
            if (
                current is None
                or current["event_sha256"] != source["event_sha256"]
                or current["artifact_id"] != row["artifact_id"]
                or current["sha256"] != source["chosen_patch_sha256"]
                or current["admission_sha256"] not in accepted_admissions
                or current["policy_version"] != ADMISSION_POLICY
                or current["decision"] != "eligible"
                or current["verification_status"] != "pass"
                or metadata.get("repository_id") != row["repository_id"]
            ):
                raise ValueError("curriculum admission evidence is inconsistent")
            members.append(
                DatasetMember(
                    event_id=row["event_id"],
                    artifact_id=row["artifact_id"],
                    split=Split(row["split"]),
                    lineage_id=row["lineage_id"],
                    source_document_sha256=row["source_file_sha256"],
                    repository_id=row["repository_id"],
                    component_family=row["component_family"],
                    temporal_bucket=row["temporal_bucket"],
                )
            )
    train_path = "llamafactory/coding_sft_train.json"
    train_receipt = manifest["artifacts"][train_path]
    if (
        sequence_audit.get("schema")
        != "harness.coding-sequence-length-audit.v1"
        or sequence_audit.get("dataset_sha256") != train_receipt["sha256"]
        or sequence_audit.get("records") != manifest["counts"]["train"]
        or sequence_audit.get("truncated_records") != 0
    ):
        raise ValueError("sequence audit does not cover curriculum training data")
    return DatasetVersionRegistry(store).create(
        dataset_version_id=dataset_version_id,
        name="owned-code-curriculum",
        version=version,
        source_revision=manifest["core_sha256"],
        split_policy={
            **manifest["split_policy"],
            "admission_policy": ADMISSION_POLICY,
            "dataset_manifest_sha256": manifest["core_sha256"],
            "sequence_audit_sha256": sequence_audit_sha256,
            "sequence_cutoff_len": sequence_audit["cutoff_len"],
            "sequence_model_config_sha256": sequence_audit[
                "model_config_sha256"
            ],
        },
        members=members,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--database", required=True, type=Path)
    build_parser.add_argument("--artifact-root", required=True, type=Path)
    build_parser.add_argument("--destination", required=True, type=Path)
    build_parser.add_argument("--max-prompt-chars", type=int, default=28_000)
    register_parser = commands.add_parser("register")
    register_parser.add_argument("--database", required=True, type=Path)
    register_parser.add_argument("--manifest", required=True, type=Path)
    register_parser.add_argument("--dataset-version-id", required=True)
    register_parser.add_argument("--version", required=True)
    register_parser.add_argument("--sequence-audit", required=True, type=Path)
    register_parser.add_argument("--import-report", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    store = Store(arguments.database)
    if arguments.command == "build":
        manifest = build(
            store=store,
            artifact_root=arguments.artifact_root,
            destination=arguments.destination,
            max_prompt_chars=arguments.max_prompt_chars,
        )
        output = {
            "dataset_manifest_sha256": manifest["core_sha256"],
            "counts": manifest["counts"],
        }
    else:
        manifest = load_manifest(arguments.manifest)
        sequence = json.loads(arguments.sequence_audit.read_text())
        digest = register(
            store=store,
            manifest=manifest,
            dataset_version_id=arguments.dataset_version_id,
            version=arguments.version,
            sequence_audit=sequence,
            sequence_audit_sha256=_file_sha256(arguments.sequence_audit),
            import_report=_load_import_report(arguments.import_report),
        )
        output = {
            "dataset_version_id": arguments.dataset_version_id,
            "registry_manifest_sha256": digest,
            "members": len(manifest["membership"]),
        }
    assert_no_secrets(json.dumps(output), field="curriculum dataset result")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
