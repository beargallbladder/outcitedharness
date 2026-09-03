from __future__ import annotations

import ast
import difflib
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from harness.shadow.models import RepositorySnapshot, ShadowPolicy, canonical_json
from harness.shadow.policy import path_allowed
from harness.training.ledger import (
    ArtifactPayload,
    LearningLedger,
    VerificationPayload,
)
from harness.training.models import LearningEvent, SourceKind, is_excluded_learning_source
from harness.training.security import assert_no_secrets, find_secrets, redact_text


SCHEMA = "harness.owned-code-curriculum.v1"
ADMISSION_POLICY = "owned-repository-executable-curriculum-v1"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MutationSpec(_StrictModel):
    mutation_id: str = Field(pattern=r"^mutation-[0-9a-f]{40}$")
    path: str
    operator: Literal[
        "flip_boolean",
        "flip_zero_one",
        "negate_comparison",
        "negate_condition",
        "remove_not",
    ]
    symbol: str
    start_line: int = Field(ge=1)
    start_column: int = Field(ge=0)
    end_line: int = Field(ge=1)
    end_column: int = Field(ge=0)
    original: str
    replacement: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mutant_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class VerificationReceipt(_StrictModel):
    command_name: str
    command: str
    baseline_returncode: int
    baseline_stdout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_stderr_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mutant_returncode: int
    mutant_stdout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mutant_stderr_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mutant_stdout_tail: str
    mutant_stderr_tail: str
    network_isolation: Literal["sandbox-exec", "bubblewrap"]


class VerifiedCurriculumTask(_StrictModel):
    schema_name: Literal["harness.owned-code-curriculum.v1"] = SCHEMA
    task_id: str = Field(pattern=r"^curriculum-[0-9a-f]{40}$")
    repository_id: str
    source_revision: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    parent_revision: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    parent_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lineage_id: str
    mutation: MutationSpec
    prompt: str
    bug_patch: str
    gold_patch: str
    mutant_source: str
    verification: VerificationReceipt
    created_at: datetime


def _sha256(value: str | bytes) -> str:
    data = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def unified_patch(before: str, after: str, path: str) -> str:
    """Return a normal-applying unified Git patch for one text file."""

    body = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )
    if not body:
        raise ValueError("patch inputs are identical")
    patch = f"diff --git a/{path} b/{path}\n{body}"
    if not patch.endswith("\n"):
        patch += "\n"
    return patch


def apply_mutation(source: str, spec: MutationSpec) -> str:
    start = _text_offset(source, spec.start_line, spec.start_column)
    end = _text_offset(source, spec.end_line, spec.end_column)
    if source[start:end] != spec.original:
        raise ValueError("mutation source span changed")
    value = source[:start] + spec.replacement + source[end:]
    if _sha256(value) != spec.mutant_sha256:
        raise ValueError("mutation output digest mismatch")
    ast.parse(value)
    return value


def _text_offset(text: str, line_number: int, byte_column: int) -> int:
    lines = text.splitlines(keepends=True)
    if line_number < 1 or line_number > len(lines):
        raise ValueError("mutation line is outside the source")
    line = lines[line_number - 1]
    encoded = line.encode("utf-8")
    if byte_column < 0 or byte_column > len(encoded):
        raise ValueError("mutation column is outside the source line")
    try:
        prefix = encoded[:byte_column].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("mutation column splits a UTF-8 character") from exc
    return sum(len(value) for value in lines[: line_number - 1]) + len(prefix)


class _PointCollector(ast.NodeVisitor):
    def __init__(self, source: str, path: str, source_sha256: str):
        self.source = source
        self.path = path
        self.source_sha256 = source_sha256
        self.symbols: list[str] = []
        self.points: list[MutationSpec] = []
        self.ranges: set[tuple[int, int, int, int]] = set()

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.symbols.append(node.name)
        self.generic_visit(node)
        self.symbols.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.symbols.append(node.name)
        self.generic_visit(node)
        self.symbols.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.visit_FunctionDef(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if type(node.value) is bool:
            self._add(node, "flip_boolean", "False" if node.value else "True")
        elif type(node.value) is int and node.value in {0, 1}:
            self._add(node, "flip_zero_one", "1" if node.value == 0 else "0")

    def visit_Compare(self, node: ast.Compare) -> Any:
        segment = ast.get_source_segment(self.source, node)
        if segment and len(segment) <= 500:
            self._add(node, "negate_comparison", f"not ({segment})")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if isinstance(node.op, ast.Not):
            operand = ast.get_source_segment(self.source, node.operand)
            if operand and len(operand) <= 500:
                self._add(node, "remove_not", f"({operand})")
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> Any:
        self._condition(node.test)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> Any:
        self._condition(node.test)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        self._condition(node.test)
        self.generic_visit(node)

    def _condition(self, node: ast.expr) -> None:
        if isinstance(node, (ast.Compare, ast.UnaryOp, ast.Constant)):
            return
        segment = ast.get_source_segment(self.source, node)
        if segment and len(segment) <= 500:
            self._add(node, "negate_condition", f"not ({segment})")

    def _add(self, node: ast.AST, operator: str, replacement: str) -> None:
        coordinates = (
            int(node.lineno),
            int(node.col_offset),
            int(node.end_lineno),
            int(node.end_col_offset),
        )
        if coordinates in self.ranges:
            return
        start = _text_offset(self.source, coordinates[0], coordinates[1])
        end = _text_offset(self.source, coordinates[2], coordinates[3])
        original = self.source[start:end]
        if not original or original == replacement:
            return
        mutant = self.source[:start] + replacement + self.source[end:]
        try:
            ast.parse(mutant)
        except SyntaxError:
            return
        identity = canonical_json(
            {
                "path": self.path,
                "operator": operator,
                "coordinates": coordinates,
                "source_sha256": self.source_sha256,
                "replacement": replacement,
            }
        )
        self.points.append(
            MutationSpec(
                mutation_id=f"mutation-{hashlib.sha256(identity).hexdigest()[:40]}",
                path=self.path,
                operator=operator,
                symbol=".".join(self.symbols) or "<module>",
                start_line=coordinates[0],
                start_column=coordinates[1],
                end_line=coordinates[2],
                end_column=coordinates[3],
                original=original,
                replacement=replacement,
                source_sha256=self.source_sha256,
                mutant_sha256=_sha256(mutant),
            )
        )
        self.ranges.add(coordinates)


def enumerate_mutations(
    root: Path,
    policy: ShadowPolicy,
    *,
    include_roots: Iterable[str] = ("harness", "scripts"),
    per_file_limit: int = 24,
    source_char_limit: int = 60_000,
) -> list[MutationSpec]:
    """Create deterministic, answer-hidden mutation candidates from owned code."""

    root = root.resolve(strict=True)
    if per_file_limit < 1:
        raise ValueError("per-file mutation limit must be positive")
    output: list[MutationSpec] = []
    for include in include_roots:
        directory = (root / include).resolve()
        if not directory.is_relative_to(root) or not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not path_allowed(policy, relative):
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if (
                not source
                or len(source) > source_char_limit
                or find_secrets(source)
                or is_excluded_learning_source(
                    SourceKind.OTHER,
                    f"file://{relative}",
                    source,
                )
            ):
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            collector = _PointCollector(source, relative, _sha256(source))
            collector.visit(tree)
            output.extend(
                sorted(
                    collector.points,
                    key=lambda row: (
                        row.start_line,
                        row.start_column,
                        row.operator,
                    ),
                )[:per_file_limit]
            )
    return sorted(output, key=lambda row: (row.path, row.start_line, row.mutation_id))


def build_prompt(
    spec: MutationSpec,
    mutant_source: str,
    *,
    command: str,
    stdout_tail: str,
    stderr_tail: str,
) -> str:
    failure = redact_text("\n".join(value for value in (stdout_tail, stderr_tail) if value))
    prompt = (
        "<TASK_KIND>repair</TASK_KIND>\n"
        "A single regression was introduced into an otherwise passing owned "
        "repository. Repair it without changing tests. Return only a unified Git "
        f"diff.\n\nTarget file: {spec.path}\nVerification command: {command}\n"
        f"Failing verification output:\n{failure[-6000:] or '(no output)'}\n\n"
        f"Current regressed file:\n--- {spec.path} ---\n{mutant_source}"
    )
    assert_no_secrets(prompt, field="owned-code curriculum prompt")
    return prompt


def make_verified_task(
    *,
    snapshot: RepositorySnapshot,
    policy: ShadowPolicy,
    spec: MutationSpec,
    mutant_source: str,
    verification: VerificationReceipt,
    created_at: datetime | None = None,
) -> VerifiedCurriculumTask:
    original_source = _restore_original(mutant_source, spec)
    bug_patch = unified_patch(original_source, mutant_source, spec.path)
    gold_patch = unified_patch(mutant_source, original_source, spec.path)
    prompt = build_prompt(
        spec,
        mutant_source,
        command=verification.command,
        stdout_tail=verification.mutant_stdout_tail,
        stderr_tail=verification.mutant_stderr_tail,
    )
    identity = canonical_json(
        {
            "repository_id": policy.repository_id,
            "parent_state_sha256": snapshot.state_sha256,
            "mutation_id": spec.mutation_id,
            "verification": verification.model_dump(mode="json"),
        }
    )
    return VerifiedCurriculumTask(
        task_id=f"curriculum-{hashlib.sha256(identity).hexdigest()[:40]}",
        repository_id=policy.repository_id,
        source_revision=snapshot.state_sha256,
        parent_revision=snapshot.revision,
        parent_state_sha256=snapshot.state_sha256,
        lineage_id=f"git:{policy.repository_id}:file:{spec.path}",
        mutation=spec,
        prompt=prompt,
        bug_patch=bug_patch,
        gold_patch=gold_patch,
        mutant_source=mutant_source,
        verification=verification,
        created_at=created_at or datetime.now(timezone.utc),
    )


def _restore_original(mutant_source: str, spec: MutationSpec) -> str:
    start = _text_offset(mutant_source, spec.start_line, spec.start_column)
    replacement_end = start + len(spec.replacement)
    if mutant_source[start:replacement_end] != spec.replacement:
        raise ValueError("mutant replacement span changed")
    original = mutant_source[:start] + spec.original + mutant_source[replacement_end:]
    if _sha256(original) != spec.source_sha256:
        raise ValueError("restored source digest mismatch")
    return original


def capture_verified_task(
    task: VerifiedCurriculumTask,
    *,
    policy: ShadowPolicy,
    ledger: LearningLedger,
) -> str:
    """Append one executable mutation task and its canonical repair to the ledger."""

    policy_sha256 = hashlib.sha256(canonical_json(policy)).hexdigest()
    verification_json = json.dumps(
        task.verification.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
    )
    event = LearningEvent(
        event_id=f"code-curriculum-{task.task_id.removeprefix('curriculum-')}",
        event_type="coding_executable_curriculum",
        source_kind=SourceKind.GIT,
        source_uri=(
            f"git+https://github.com/{task.repository_id}.git"
            f"?revision={task.parent_revision}&state={task.parent_state_sha256}"
        ),
        source_revision=task.source_revision,
        lineage_id=task.lineage_id,
        authorization_scope=policy.authorization_scope,
        created_at=task.created_at,
        metadata={
            "content_class": "owned_source_code",
            "owner_attested": policy.owner == "self",
            "data_paths_excluded": True,
            "repository_id": task.repository_id,
            "repository_policy_sha256": policy_sha256,
            "data_use": "training",
            "disposition": "verified",
            "curriculum_schema": SCHEMA,
            "curriculum_task_id": task.task_id,
            "mutation_operator": task.mutation.operator,
            "source_path": task.mutation.path,
            "source_file_sha256": task.mutation.source_sha256,
            "parent_revision": task.parent_revision,
            "parent_state_sha256": task.parent_state_sha256,
        },
    )
    capture = ledger.capture(
        event,
        [
            ArtifactPayload(
                kind="coding_prompt",
                content=task.prompt,
                media_type="text/plain",
            ),
            ArtifactPayload(
                kind="coding_chosen_patch",
                content=task.gold_patch,
                media_type="text/x-diff",
            ),
            ArtifactPayload(
                kind="coding_bug_patch",
                content=task.bug_patch,
                media_type="text/x-diff",
            ),
            ArtifactPayload(
                kind="coding_mutant_source",
                content=task.mutant_source,
                media_type="text/plain",
            ),
            ArtifactPayload(
                kind="coding_verification",
                content=verification_json,
                media_type="application/json",
            ),
        ],
        [
            VerificationPayload(
                kind="executable_mutation_fail_before_canonical_pass",
                status="pass",
                verifier="harness.training.code-curriculum.v1",
                output_kind="coding_verification",
                command=task.verification.command,
                metadata={
                    "baseline_returncode": task.verification.baseline_returncode,
                    "mutant_returncode": task.verification.mutant_returncode,
                    "source_sha256": task.mutation.source_sha256,
                    "mutant_sha256": task.mutation.mutant_sha256,
                },
            )
        ],
    )
    ledger.admit_verified_event(
        capture.event_id,
        capture.verifications[0].verification_id,
        policy_version=ADMISSION_POLICY,
        reason=(
            "canonical repository state passed while the isolated mutation failed "
            "the same network-denied verifier"
        ),
    )
    return capture.event_id


__all__ = [
    "ADMISSION_POLICY",
    "SCHEMA",
    "MutationSpec",
    "VerificationReceipt",
    "VerifiedCurriculumTask",
    "apply_mutation",
    "build_prompt",
    "capture_verified_task",
    "enumerate_mutations",
    "make_verified_task",
    "unified_patch",
]
