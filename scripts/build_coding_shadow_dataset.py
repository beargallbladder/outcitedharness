#!/usr/bin/env python3
"""Build and register immutable code datasets from proven shadow comparisons."""

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
from harness.training.ledger import LearningLedger
from harness.training.queue import DatasetMember, DatasetVersionRegistry
from harness.training.security import assert_no_secrets
from harness.training.split import (
    Split,
    assert_no_lineage_leakage,
    grouped_lineage_split,
)


SCHEMA = "harness.dataset.cursor-shadow-code.v1"
POLICY_VERSION = "cursor-shadow-mechanical-comparison-v1"
_DIFF_PATH = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    _write(
        path,
        json.dumps(value, indent=2, sort_keys=True).encode() + b"\n",
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    body = b"".join(_canonical(value) + b"\n" for value in values)
    _write(path, body)


def _read_artifact(
    ledger: LearningLedger,
    artifacts: dict[str, Any],
    kind: str,
    *,
    required: bool = True,
) -> str | None:
    artifact = artifacts.get(kind)
    if artifact is None:
        if required:
            raise ValueError(f"admitted coding event is missing {kind}")
        return None
    if not artifact.redacted:
        raise ValueError(f"{kind} did not pass redaction")
    data = ledger.vault.path_for(artifact.sha256).read_bytes()
    if len(data) != artifact.byte_size or _sha256(data) != artifact.sha256:
        raise ValueError(f"{kind} artifact changed after verification")
    try:
        return data.decode()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{kind} is not UTF-8 text") from exc


def _changed_paths(patch: str) -> tuple[str, ...]:
    output = set()
    for before, after in _DIFF_PATH.findall(patch):
        for path in (before, after):
            if path != "/dev/null":
                output.add(path)
    if not output:
        raise ValueError("chosen coding patch has no git diff paths")
    return tuple(sorted(output))


def _component(path: str) -> str:
    parts = Path(path).parts
    if not parts:
        raise ValueError("empty coding patch path")
    if parts[0] in {"app", "components", "harness", "lib", "services", "tests"}:
        return "/".join(parts[:2]) if len(parts) > 1 else parts[0]
    return parts[0]


def _context_prompt(record: dict[str, Any]) -> str:
    prompt = str(record["prompt"])
    local_attempt = record["local_attempt"]
    comparison = record["comparison"]
    sections = [
        "Repair the repository task below. Return only a unified git diff.",
        f"Task:\n{prompt}",
        (
            "Local shadow result:\n"
            + json.dumps(
                {
                    "status": local_attempt.get("status"),
                    "answer": local_attempt.get("answer"),
                    "patch": local_attempt.get("patch"),
                },
                sort_keys=True,
            )
        ),
        (
            "Mechanical comparison:\n"
            + json.dumps(
                {
                    "decision": comparison.get("decision"),
                    "reason": comparison.get("reason"),
                    "local": comparison.get("local"),
                    "frontier": comparison.get("frontier"),
                },
                sort_keys=True,
            )
        ),
    ]
    value = "\n\n".join(sections)
    assert_no_secrets(value, field="coding dataset prompt")
    return value


def _load_records(
    store: Store,
    artifact_root: Path,
) -> list[dict[str, Any]]:
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
              AND event.event_type LIKE 'coding_%'
            ORDER BY event.created_at, event.event_id
            """,
            (POLICY_VERSION,),
        ).fetchall()
    if not rows:
        raise ValueError("no mechanically admitted coding events are available")

    output = []
    for row in rows:
        capture = ledger.verify_event(row["event_id"])
        if len(capture.verifications) != 1:
            raise ValueError("coding event must have exactly one verification")
        proof = capture.verifications[0]
        if (
            proof.status != "pass"
            or proof.kind != "same_parent_fail_before_pass_after"
            or row["admission_decision"] != "eligible"
        ):
            raise ValueError("coding event lacks passing mechanical proof")
        metadata = json.loads(row["metadata_json"])
        if (
            metadata.get("content_class") != "owned_source_code"
            or metadata.get("owner_attested") is not True
            or metadata.get("data_paths_excluded") is not True
            or metadata.get("data_use") != "training"
            or metadata.get("disposition") != "verified"
        ):
            raise ValueError("coding event ownership boundary is incomplete")
        artifacts = {artifact.kind: artifact for artifact in capture.artifacts}
        if len(artifacts) != len(capture.artifacts):
            raise ValueError("coding event has duplicate artifact kinds")
        prompt = _read_artifact(ledger, artifacts, "coding_prompt")
        chosen_patch = _read_artifact(
            ledger,
            artifacts,
            "coding_chosen_patch",
        )
        comparison_text = _read_artifact(
            ledger,
            artifacts,
            "coding_comparison",
        )
        attempt_text = _read_artifact(
            ledger,
            artifacts,
            "coding_local_attempt",
        )
        snapshot_text = _read_artifact(
            ledger,
            artifacts,
            "coding_parent_snapshot",
        )
        assert prompt is not None
        assert chosen_patch is not None
        assert comparison_text is not None
        assert attempt_text is not None
        assert snapshot_text is not None
        comparison = json.loads(comparison_text)
        local_attempt = json.loads(attempt_text)
        parent_snapshot = json.loads(snapshot_text)
        if (
            comparison.get("eligible") is not True
            or comparison.get("chosen") not in {"local", "frontier"}
            or comparison.get("teacher_identity_verified") is not True
            or comparison.get("evidence_sha256")
            != proof.metadata.get("comparison_evidence_sha256")
            or parent_snapshot.get("state_sha256")
            != metadata.get("parent_state_sha256")
        ):
            raise ValueError("coding comparison does not match admission proof")
        changed_paths = _changed_paths(chosen_patch)
        components = tuple(sorted({_component(path) for path in changed_paths}))
        rejected_patch = _read_artifact(
            ledger,
            artifacts,
            "coding_rejected_patch",
            required=False,
        )
        observed_at = datetime.fromisoformat(str(row["created_at"]))
        repository_id = str(metadata["repository_id"])
        revision = str(row["source_revision"])
        if row["lineage_id"] != f"git:{repository_id}":
            raise ValueError("coding event lineage must cover the entire repository")
        output.append(
            {
                "event_id": row["event_id"],
                "event_sha256": capture.event_sha256,
                "admission_id": row["admission_id"],
                "admission_sha256": row["admission_sha256"],
                "chosen_patch_artifact_id": artifacts[
                    "coding_chosen_patch"
                ].artifact_id,
                "chosen_patch_sha256": artifacts[
                    "coding_chosen_patch"
                ].sha256,
                "repository_id": repository_id,
                "source_revision": revision,
                "lineage_id": str(row["lineage_id"]),
                "source_revision_sha256": _sha256(
                    f"{repository_id}\0{revision}".encode()
                ),
                "observed_at": observed_at,
                "temporal_bucket": observed_at.strftime("%Y-%m"),
                "prompt": prompt,
                "chosen_patch": chosen_patch,
                "rejected_patch": rejected_patch,
                "comparison": comparison,
                "local_attempt": local_attempt,
                "changed_paths": changed_paths,
                "components": components,
            }
        )
    return output


def build(
    *,
    store: Store,
    artifact_root: Path,
    destination: Path,
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"destination already exists: {destination}")
    records = _load_records(store, artifact_root)
    partitions = grouped_lineage_split(
        records,
        lineage_key="lineage_id",
        seed="cursor-shadow-code-v1",
    )
    assert_no_lineage_leakage(partitions, lineage_key="lineage_id")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    generated = []
    source_events = []
    try:
        dataset_info = {}
        membership = []
        for split in Split:
            canonical_sft = []
            canonical_preference = []
            llama_sft = []
            llama_preference = []
            for record in sorted(
                partitions[split],
                key=lambda row: (row["observed_at"], row["event_id"]),
            ):
                training_prompt = _context_prompt(record)
                canonical = {
                    "pair_id": f"sft-{record['event_id']}",
                    "event_id": record["event_id"],
                    "lineage_id": record["lineage_id"],
                    "repository_id": record["repository_id"],
                    "source_revision": record["source_revision"],
                    "observed_at": record["observed_at"].isoformat(),
                    "prompt": training_prompt,
                    "response": record["chosen_patch"],
                    "chosen_patch_sha256": record["chosen_patch_sha256"],
                }
                canonical_sft.append(canonical)
                llama_sft.append(
                    {
                        "messages": [
                            {"role": "user", "content": training_prompt},
                            {
                                "role": "assistant",
                                "content": record["chosen_patch"],
                            },
                        ]
                    }
                )
                if record["rejected_patch"]:
                    preference = {
                        **canonical,
                        "pair_id": f"preference-{record['event_id']}",
                        "chosen": record["chosen_patch"],
                        "rejected": record["rejected_patch"],
                    }
                    canonical_preference.append(preference)
                    llama_preference.append(
                        {
                            "messages": [
                                {"role": "user", "content": training_prompt}
                            ],
                            "chosen": {
                                "role": "assistant",
                                "content": record["chosen_patch"],
                            },
                            "rejected": {
                                "role": "assistant",
                                "content": record["rejected_patch"],
                            },
                        }
                    )
                membership.append(
                    {
                        "event_id": record["event_id"],
                        "split": split.value,
                        "lineage_id": record["lineage_id"],
                        "repository_id": record["repository_id"],
                        "source_revision": record["source_revision"],
                        "source_revision_sha256": record[
                            "source_revision_sha256"
                        ],
                        "component_family": "+".join(record["components"]),
                        "temporal_bucket": record["temporal_bucket"],
                        "artifact_id": record["chosen_patch_artifact_id"],
                    }
                )
            paths = {
                "canonical_sft": (
                    temporary / "canonical/sft" / f"{split.value}.jsonl"
                ),
                "canonical_preference": (
                    temporary
                    / "canonical/preference"
                    / f"{split.value}.jsonl"
                ),
                "llama_sft": (
                    temporary
                    / "llamafactory"
                    / f"coding_sft_{split.value}.json"
                ),
                "llama_preference": (
                    temporary
                    / "llamafactory"
                    / f"coding_preference_{split.value}.json"
                ),
            }
            _write_jsonl(paths["canonical_sft"], canonical_sft)
            _write_jsonl(
                paths["canonical_preference"],
                canonical_preference,
            )
            _write_json(paths["llama_sft"], llama_sft)
            _write_json(
                paths["llama_preference"],
                llama_preference,
            )
            generated.extend(paths.values())
            dataset_info[f"coding_sft_{split.value}"] = {
                "file_name": paths["llama_sft"].name,
                "formatting": "sharegpt",
                "columns": {"messages": "messages"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            }
            dataset_info[f"coding_preference_{split.value}"] = {
                "file_name": paths["llama_preference"].name,
                "formatting": "sharegpt",
                "ranking": True,
                "columns": {
                    "messages": "messages",
                    "chosen": "chosen",
                    "rejected": "rejected",
                },
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            }
        dataset_info_path = temporary / "llamafactory/dataset_info.json"
        _write_json(dataset_info_path, dataset_info)
        generated.append(dataset_info_path)
        for record in records:
            source_events.append(
                {
                    "event_id": record["event_id"],
                    "event_sha256": record["event_sha256"],
                    "admission_id": record["admission_id"],
                    "admission_sha256": record["admission_sha256"],
                    "chosen_patch_sha256": record["chosen_patch_sha256"],
                }
            )
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "created_at": max(record["observed_at"] for record in records).isoformat(),
            "admission_policy": POLICY_VERSION,
            "split_policy": {
                "kind": "repository_lineage_hash",
                "seed": "cursor-shadow-code-v1",
                "leakage_keys": [
                    "lineage_id",
                    "source_document_sha256",
                    "component_family",
                ],
            },
            "source_events": sorted(
                source_events,
                key=lambda row: row["event_id"],
            ),
            "membership": sorted(
                membership,
                key=lambda row: (
                    row["split"],
                    row["lineage_id"],
                    row["event_id"],
                ),
            ),
            "counts": {
                split.value: {
                    "sft": len(partitions[split]),
                    "preference": sum(
                        bool(record["rejected_patch"])
                        for record in partitions[split]
                    ),
                }
                for split in Split
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
        raise ValueError("coding dataset manifest must be a regular file")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported coding dataset manifest")
    expected = value.get("core_sha256")
    core = {key: item for key, item in value.items() if key != "core_sha256"}
    if expected != _sha256(_canonical(core)):
        raise ValueError("coding dataset manifest digest mismatch")
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
            raise ValueError(f"coding dataset artifact changed: {relative}")
    return value


def load_import_report(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("learning import report must be a regular file")
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema") != "harness.learning-transfer-import.v1"
    ):
        raise ValueError("unsupported learning import report")
    expected = value.get("result_sha256")
    core = {key: item for key, item in value.items() if key != "result_sha256"}
    if expected != _sha256(_canonical(core)):
        raise ValueError("learning import report digest mismatch")
    return value


def load_sequence_audit(
    path: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("coding sequence audit must be a regular file")
    value = json.loads(path.read_text())
    train_path = "llamafactory/coding_sft_train.json"
    receipt = manifest.get("artifacts", {}).get(train_path)
    train_count = manifest.get("counts", {}).get("train", {}).get("sft")
    if (
        not isinstance(value, dict)
        or value.get("schema") != "harness.coding-sequence-length-audit.v1"
        or not isinstance(receipt, dict)
        or not isinstance(train_count, int)
        or train_count < 1
        or value.get("dataset_sha256") != receipt.get("sha256")
        or value.get("records") != train_count
        or value.get("truncated_records") != 0
        or not isinstance(value.get("cutoff_len"), int)
        or value["cutoff_len"] < 256
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(value.get("model_config_sha256") or ""),
        )
    ):
        raise ValueError("coding sequence audit is inconsistent")
    return value, _file_sha256(path)


def register(
    *,
    store: Store,
    manifest: dict[str, Any],
    dataset_version_id: str,
    version: str,
    sequence_audit: dict[str, Any],
    sequence_audit_sha256: str,
    import_report: dict[str, Any] | None = None,
) -> str:
    source_by_event = {
        row["event_id"]: row for row in manifest["source_events"]
    }
    imported_admissions: dict[str, tuple[str, str]] = {}
    if import_report is not None:
        imported_rows = import_report.get("events")
        if not isinstance(imported_rows, list):
            raise ValueError("learning import report events are malformed")
        for row in imported_rows:
            if not isinstance(row, dict):
                raise ValueError("learning import report event is malformed")
            event_id = str(row.get("event_id") or "")
            values = (
                str(row.get("source_admission_sha256") or ""),
                str(row.get("destination_admission_sha256") or ""),
            )
            if (
                not event_id
                or event_id in imported_admissions
                or any(
                    len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)
                    for value in values
                )
            ):
                raise ValueError("learning import report admission is malformed")
            imported_admissions[event_id] = values
    members = []
    with store.connect() as connection:
        for row in manifest["membership"]:
            source = source_by_event.get(row["event_id"])
            current = connection.execute(
                """
                SELECT
                    event.event_sha256,
                    event.metadata_json,
                    artifact.artifact_id,
                    artifact.sha256,
                    admission.admission_id,
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
            if current is None:
                raise ValueError("coding dataset event is absent from destination ledger")
            metadata = json.loads(current["metadata_json"])
            accepted_admissions = {source["admission_sha256"]}
            imported = imported_admissions.get(row["event_id"])
            if imported is not None:
                if imported[0] != source["admission_sha256"]:
                    raise ValueError("source admission differs from import report")
                accepted_admissions.add(imported[1])
            if (
                current["event_sha256"] != source["event_sha256"]
                or current["artifact_id"] != row["artifact_id"]
                or current["sha256"] != source["chosen_patch_sha256"]
                or current["admission_id"] != source["admission_id"]
                or current["admission_sha256"] not in accepted_admissions
                or current["policy_version"] != POLICY_VERSION
                or current["decision"] != "eligible"
                or current["verification_status"] != "pass"
                or metadata.get("repository_id") != row["repository_id"]
            ):
                raise ValueError("coding dataset admission evidence is inconsistent")
            members.append(
                DatasetMember(
                    event_id=row["event_id"],
                    artifact_id=row["artifact_id"],
                    split=Split(row["split"]),
                    lineage_id=row["lineage_id"],
                    source_document_sha256=row["source_revision_sha256"],
                    repository_id=row["repository_id"],
                    component_family=row["component_family"],
                    temporal_bucket=row["temporal_bucket"],
                )
            )
    return DatasetVersionRegistry(store).create(
        dataset_version_id=dataset_version_id,
        name="cursor-shadow-code",
        version=version,
        source_revision=manifest["core_sha256"],
        split_policy={
            **manifest["split_policy"],
            "admission_policy": POLICY_VERSION,
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
    register_parser = commands.add_parser("register")
    register_parser.add_argument("--database", required=True, type=Path)
    register_parser.add_argument("--manifest", required=True, type=Path)
    register_parser.add_argument("--dataset-version-id", required=True)
    register_parser.add_argument("--version", required=True)
    register_parser.add_argument("--import-report", type=Path)
    register_parser.add_argument("--sequence-audit", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    store = Store(arguments.database)
    if arguments.command == "build":
        manifest = build(
            store=store,
            artifact_root=arguments.artifact_root,
            destination=arguments.destination,
        )
        output = {
            "dataset_manifest_sha256": manifest["core_sha256"],
            "counts": manifest["counts"],
        }
    else:
        manifest = load_manifest(arguments.manifest)
        sequence_audit, sequence_audit_sha256 = load_sequence_audit(
            arguments.sequence_audit,
            manifest,
        )
        digest = register(
            store=store,
            manifest=manifest,
            dataset_version_id=arguments.dataset_version_id,
            version=arguments.version,
            sequence_audit=sequence_audit,
            sequence_audit_sha256=sequence_audit_sha256,
            import_report=(
                load_import_report(arguments.import_report)
                if arguments.import_report is not None
                else None
            ),
        )
        output = {
            "dataset_version_id": arguments.dataset_version_id,
            "registry_manifest_sha256": digest,
            "members": len(manifest["membership"]),
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
