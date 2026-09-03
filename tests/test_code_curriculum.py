from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from build_owned_code_curriculum import build as build_curriculum_run  # noqa: E402
from build_owned_code_curriculum_dataset import (  # noqa: E402
    build as build_curriculum_dataset,
    register as register_curriculum_dataset,
)
from evaluate_owned_code_curriculum import (  # noqa: E402
    _apply_and_match,
    _extract_patch,
)
from harness.shadow.models import RepositorySnapshot, ShadowPolicy
from harness.storage.db import Store
from harness.training.code_curriculum import (
    ADMISSION_POLICY,
    MutationSpec,
    VerificationReceipt,
    apply_mutation,
    capture_verified_task,
    enumerate_mutations,
    make_verified_task,
    unified_patch,
)
from harness.training.ledger import LearningLedger


NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _policy() -> ShadowPolicy:
    return ShadowPolicy(
        repository_id="owner/example",
        allowed_paths=("harness/**",),
        excluded_paths=(".git", "**/.git/**"),
    )


def test_enumerates_deterministic_answer_hidden_mutations(tmp_path: Path) -> None:
    package = tmp_path / "harness"
    package.mkdir()
    source = (
        "def allowed(value: int) -> bool:\n"
        "    enabled = True\n"
        "    return enabled and value == 1\n"
    )
    path = package / "example.py"
    path.write_text(source)

    first = enumerate_mutations(tmp_path, _policy(), per_file_limit=20)
    second = enumerate_mutations(tmp_path, _policy(), per_file_limit=20)

    assert first == second
    assert {row.operator for row in first} >= {
        "flip_boolean",
        "flip_zero_one",
        "negate_comparison",
    }
    for mutation in first:
        mutant = apply_mutation(source, mutation)
        assert mutant != source
        assert _sha256(mutant) == mutation.mutant_sha256


def test_verified_mutation_is_admitted_with_executable_proof(tmp_path: Path) -> None:
    package = tmp_path / "harness"
    package.mkdir()
    source = "def allowed(value: int) -> bool:\n    return value == 1\n"
    path = package / "example.py"
    path.write_text(source)
    policy = _policy()
    mutation = next(
        row
        for row in enumerate_mutations(tmp_path, policy, per_file_limit=20)
        if row.operator == "negate_comparison"
    )
    mutant = apply_mutation(source, mutation)
    snapshot = RepositorySnapshot(
        repository_id=policy.repository_id,
        repository_root=tmp_path,
        revision="a" * 40,
        dirty_patch="",
        dirty_patch_sha256=_sha256(""),
        state_sha256="b" * 64,
        captured_at=NOW,
    )
    receipt = VerificationReceipt(
        command_name="unit",
        command="python -m pytest -q",
        baseline_returncode=0,
        baseline_stdout_sha256="c" * 64,
        baseline_stderr_sha256="d" * 64,
        mutant_returncode=1,
        mutant_stdout_sha256="e" * 64,
        mutant_stderr_sha256="f" * 64,
        mutant_stdout_tail="FAILED tests/test_example.py::test_allowed",
        mutant_stderr_tail="",
        network_isolation="sandbox-exec",
    )
    task = make_verified_task(
        snapshot=snapshot,
        policy=policy,
        spec=mutation,
        mutant_source=mutant,
        verification=receipt,
        created_at=NOW,
    )

    repair_root = tmp_path / "repair"
    repair_file = repair_root / "harness" / "example.py"
    repair_file.parent.mkdir(parents=True)
    repair_file.write_text(mutant)
    applied = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=repair_root,
        input=task.gold_patch,
        text=True,
        capture_output=True,
        check=False,
    )
    assert applied.returncode == 0, applied.stderr

    store = Store(tmp_path / "learning.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    event_id = capture_verified_task(task, policy=policy, ledger=ledger)

    verified = ledger.verify_event(event_id)
    assert {row.kind for row in verified.artifacts} == {
        "coding_bug_patch",
        "coding_chosen_patch",
        "coding_mutant_source",
        "coding_prompt",
        "coding_verification",
    }
    with store.connect() as connection:
        admission = connection.execute(
            "SELECT policy_version, decision FROM learning_admissions"
        ).fetchone()
    assert admission["policy_version"] == ADMISSION_POLICY
    assert admission["decision"] == "eligible"


def test_builds_executable_curriculum_from_owned_repository(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    (repository / "harness").mkdir(parents=True)
    (repository / "tests").mkdir()
    (repository / "harness" / "__init__.py").write_text("")
    (repository / "harness" / "example.py").write_text(
        "def allowed(value: int) -> bool:\n    return value == 1\n"
    )
    (repository / "tests" / "test_example.py").write_text(
        "from harness.example import allowed\n\n"
        "def test_allowed():\n"
        "    assert allowed(1)\n"
        "    assert not allowed(2)\n"
    )
    (repository / ".harness.toml").write_text(
        "[verification]\n"
        'required = ["unit"]\n\n'
        "[verification.commands.unit]\n"
        'argv = ["python", "-m", "pytest", "-q"]\n'
        "timeout = 30\n"
    )
    (repository / ".harness-shadow.json").write_text(
        json.dumps(
            {
                "version": 1,
                "enabled": True,
                "repository_id": "owner/example",
                "allowed_paths": ["."],
                "excluded_paths": [".git", "**/.git/**"],
            }
        )
    )
    subprocess.run(["git", "init", "-q", repository], check=True)
    subprocess.run(["git", "-C", repository, "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            repository,
            "-c",
            "user.name=Harness Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        check=True,
    )
    output = tmp_path / "curriculum.json"
    result = build_curriculum_run(
        repository=repository,
        database=tmp_path / "ledger.db",
        artifact_root=tmp_path / "artifacts",
        output=output,
        work_root=tmp_path / "work",
        command_name="unit",
        max_mutations=2,
        target_verified=1,
        workers=1,
        per_file_limit=8,
        include_roots=("harness",),
    )

    assert result["verified_tasks"] == 1
    assert result["available_mutations"] >= 2
    assert output.is_file()
    assert output.stat().st_mode & 0o222 == 0
    with Store(tmp_path / "ledger.db").connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM learning_admissions"
        ).fetchone()[0] == 1


def test_builds_and_registers_file_lineage_safe_dataset(tmp_path: Path) -> None:
    policy = _policy()
    store = Store(tmp_path / "learning.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    snapshot = RepositorySnapshot(
        repository_id=policy.repository_id,
        repository_root=tmp_path,
        revision="a" * 40,
        dirty_patch="",
        dirty_patch_sha256=_sha256(""),
        state_sha256="b" * 64,
        captured_at=NOW,
    )
    for index in range(60):
        original_value = index % 2
        replacement_value = 1 - original_value
        path = f"harness/component_{index}.py"
        source = f"def value_{index}():\n    return {original_value}\n"
        mutant = f"def value_{index}():\n    return {replacement_value}\n"
        mutation = MutationSpec(
            mutation_id=f"mutation-{index:040x}",
            path=path,
            operator="flip_zero_one",
            symbol="value",
            start_line=2,
            start_column=11,
            end_line=2,
            end_column=12,
            original=str(original_value),
            replacement=str(replacement_value),
            source_sha256=_sha256(source),
            mutant_sha256=_sha256(mutant),
        )
        receipt = VerificationReceipt(
            command_name="unit",
            command="python -m pytest -q",
            baseline_returncode=0,
            baseline_stdout_sha256="c" * 64,
            baseline_stderr_sha256="d" * 64,
            mutant_returncode=1,
            mutant_stdout_sha256="e" * 64,
            mutant_stderr_sha256="f" * 64,
            mutant_stdout_tail=f"FAILED tests/test_component_{index}.py",
            mutant_stderr_tail="",
            network_isolation="sandbox-exec",
        )
        task = make_verified_task(
            snapshot=snapshot,
            policy=policy,
            spec=mutation,
            mutant_source=mutant,
            verification=receipt,
            created_at=NOW,
        )
        capture_verified_task(task, policy=policy, ledger=ledger)

    destination = tmp_path / "dataset"
    manifest = build_curriculum_dataset(
        store=store,
        artifact_root=tmp_path / "artifacts",
        destination=destination,
    )

    assert sum(manifest["counts"].values()) == 60
    assert all(manifest["counts"][split] for split in ("train", "validation", "test"))
    train_path = destination / "llamafactory" / "coding_sft_train.json"
    sequence_audit = {
        "schema": "harness.coding-sequence-length-audit.v1",
        "model_config_sha256": "1" * 64,
        "dataset_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "records": manifest["counts"]["train"],
        "cutoff_len": 8192,
        "truncated_records": 0,
    }
    digest = register_curriculum_dataset(
        store=store,
        manifest=manifest,
        dataset_version_id="owned-code-curriculum-v1",
        version="v1",
        sequence_audit=sequence_audit,
        sequence_audit_sha256="2" * 64,
    )

    assert len(digest) == 64
    with store.connect() as connection:
        assert connection.execute(
            """
            SELECT COUNT(*) FROM dataset_members
            WHERE dataset_version_id = 'owned-code-curriculum-v1'
            """
        ).fetchone()[0] == 60


def test_curriculum_evaluator_requires_exact_single_file_repair() -> None:
    path = "harness/example.py"
    original = "value = 1\n"
    mutant = "value = 0\n"
    patch = unified_patch(mutant, original, path)

    assert _extract_patch(f"```diff\n{patch}```", path) == patch
    result = _apply_and_match(
        expected_path=path,
        mutant_source=mutant,
        expected_source_sha256=_sha256(original),
        patch=patch,
    )
    assert result["passed"]
    assert result["patch_applied"]
