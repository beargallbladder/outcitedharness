from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from harness.training.hygiene import deduplicate_pairs
from harness.training.models import (
    DataUse,
    GitCandidate,
    SourceKind,
    SourceProvenance,
    TestEvidence,
    TextPair,
    VisionPair,
)
from harness.training.security import assert_no_secrets, redact_text


PairT = TypeVar("PairT", TextPair, VisionPair)
RecordSource = Path | str | Iterable[Mapping[str, Any]]


class ExportValidationError(ValueError):
    pass


def _records(source: RecordSource) -> list[Mapping[str, Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        output: list[Mapping[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ExportValidationError(
                        f"{path}:{line_number}: invalid JSON"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise ExportValidationError(
                        f"{path}:{line_number}: record must be an object"
                    )
                output.append(row)
        return output
    return list(source)


def _required(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    raise ExportValidationError(f"missing required field ({' or '.join(names)})")


def _provenance(
    row: Mapping[str, Any],
    *,
    source_kind: SourceKind,
    data_use: DataUse,
) -> SourceProvenance:
    raw = row.get("provenance")
    if not isinstance(raw, Mapping):
        raise ExportValidationError("complete provenance object is required")
    try:
        provenance = SourceProvenance.model_validate(raw)
    except ValidationError as exc:
        raise ExportValidationError("incomplete or invalid provenance") from exc
    if provenance.source_kind is not source_kind:
        raise ExportValidationError(
            f"expected {source_kind.value} provenance, got "
            f"{provenance.source_kind.value}"
        )
    if provenance.data_use is not data_use:
        raise ExportValidationError(
            f"source must be marked {data_use.value}, got {provenance.data_use.value}"
        )
    return provenance


def _clean_training_text(value: Any, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ExportValidationError(f"{field} cannot be empty")
    # Reject credentials before applying allowed PII redactions.
    assert_no_secrets(text, field=field)
    return redact_text(text)


def _stable_id(prefix: str, *values: str) -> str:
    payload = "\0".join(values).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:24]}"


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def write_pairs_jsonl(path: Path, pairs: Iterable[TextPair]) -> None:
    _write_jsonl(
        path,
        (pair.model_dump(mode="json", exclude_none=True) for pair in pairs),
    )


def export_harness_pass_pairs(
    cases: Iterable[Mapping[str, Any]],
    results: Iterable[Mapping[str, Any]],
    *,
    destination: Path | None = None,
) -> list[TextPair]:
    """Export only evaluator-verified Harness PASS responses."""

    cases_by_id = {
        str(_required(case, "id", "case_id")): case
        for case in cases
    }
    pairs: list[TextPair] = []
    for result in results:
        evaluation = result.get("evaluation")
        evaluation_map = evaluation if isinstance(evaluation, Mapping) else {}
        verdict = result.get("verdict", evaluation_map.get("verdict"))
        if verdict != "PASS":
            continue
        if result.get("error"):
            raise ExportValidationError("PASS result cannot contain an error")
        evaluator = result.get("evaluator", evaluation_map.get("evaluator"))
        if not evaluator:
            raise ExportValidationError("PASS result lacks evaluator verification")
        if result.get("verified") is False:
            raise ExportValidationError("explicitly unverified PASS result")

        case_id = str(_required(result, "case_id"))
        try:
            case = cases_by_id[case_id]
        except KeyError as exc:
            raise ExportValidationError(f"missing case {case_id!r}") from exc
        provenance_holder = (
            result if isinstance(result.get("provenance"), Mapping) else case
        )
        provenance = _provenance(
            provenance_holder,
            source_kind=SourceKind.HARNESS,
            data_use=DataUse.TRAINING,
        )
        prompt = _clean_training_text(_required(case, "prompt"), field="prompt")
        response = _clean_training_text(
            _required(result, "response", "answer", "text"),
            field="response",
        )
        pair_id = str(
            result.get("pair_id")
            or _stable_id("harness", case_id, provenance.source_record_id, response)
        )
        pairs.append(
            TextPair(
                pair_id=pair_id,
                prompt=prompt,
                response=response,
                provenance=provenance,
                metadata={
                    "case_id": case_id,
                    "run_id": result.get("run_id"),
                    "evaluator": str(evaluator),
                    "verdict": "PASS",
                },
            )
        )

    output = deduplicate_pairs(pairs)
    if destination is not None:
        write_pairs_jsonl(destination, output)
    return output


def load_designwins_text_pairs(
    source: RecordSource,
    *,
    destination: Path | None = None,
) -> list[TextPair]:
    pairs: list[TextPair] = []
    for row in _records(source):
        provenance = _provenance(
            row,
            source_kind=SourceKind.DESIGNWINS,
            data_use=DataUse.TRAINING,
        )
        prompt = _clean_training_text(
            _required(row, "prompt", "input", "instruction"), field="prompt"
        )
        response = _clean_training_text(
            _required(row, "response", "output", "answer"), field="response"
        )
        pairs.append(
            TextPair(
                pair_id=str(
                    row.get("pair_id")
                    or row.get("id")
                    or _stable_id(
                        "designwins-text",
                        provenance.source_record_id,
                        prompt,
                        response,
                    )
                ),
                prompt=prompt,
                response=response,
                provenance=provenance,
                metadata=dict(row.get("metadata") or {}),
            )
        )
    output = deduplicate_pairs(pairs)
    if destination is not None:
        write_pairs_jsonl(destination, output)
    return output


def _images(row: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw_images = row.get("images")
    if isinstance(raw_images, list):
        uris: list[str] = []
        digests: list[str] = []
        for image in raw_images:
            if not isinstance(image, Mapping):
                raise ExportValidationError("image entry must be an object")
            uris.append(str(_required(image, "uri", "path")))
            digests.append(str(_required(image, "sha256")))
        return tuple(uris), tuple(digests)
    uri_value = _required(row, "image_uris", "image_uri", "image")
    digest_value = _required(row, "image_sha256")
    uris = (
        tuple(str(value) for value in uri_value)
        if isinstance(uri_value, list)
        else (str(uri_value),)
    )
    digests = (
        tuple(str(value) for value in digest_value)
        if isinstance(digest_value, list)
        else (str(digest_value),)
    )
    return uris, digests


def load_designwins_vision_pairs(
    source: RecordSource,
    *,
    destination: Path | None = None,
) -> list[VisionPair]:
    pairs: list[VisionPair] = []
    for row in _records(source):
        provenance = _provenance(
            row,
            source_kind=SourceKind.DESIGNWINS,
            data_use=DataUse.TRAINING,
        )
        prompt = _clean_training_text(
            _required(row, "prompt", "input", "instruction"), field="prompt"
        )
        response = _clean_training_text(
            _required(row, "response", "output", "answer"), field="response"
        )
        image_uris, image_sha256 = _images(row)
        pairs.append(
            VisionPair(
                pair_id=str(
                    row.get("pair_id")
                    or row.get("id")
                    or _stable_id(
                        "designwins-vision",
                        provenance.source_record_id,
                        prompt,
                        response,
                        *image_sha256,
                    )
                ),
                prompt=prompt,
                response=response,
                provenance=provenance,
                image_uris=image_uris,
                image_sha256=image_sha256,
                metadata=dict(row.get("metadata") or {}),
            )
        )
    # Vision identity includes images, so do not collapse visually distinct rows.
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    output: list[VisionPair] = []
    for pair in pairs:
        key = (pair.prompt.casefold(), pair.response.casefold(), pair.image_sha256)
        if key not in seen:
            seen.add(key)
            output.append(pair)
    if destination is not None:
        write_pairs_jsonl(destination, output)
    return output


def _native_designwins_provenance(
    row: Mapping[str, Any],
    source: Path,
    *,
    part: str,
    modality: str,
    license_name: str,
) -> SourceProvenance:
    canonical = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    collected = datetime.fromtimestamp(source.stat().st_mtime, timezone.utc)
    return SourceProvenance(
        source_kind=SourceKind.DESIGNWINS,
        source_uri=source.resolve().as_uri(),
        source_record_id=f"{part}:{modality}",
        collected_at=collected,
        content_sha256=hashlib.sha256(canonical).hexdigest(),
        lineage_id=f"designwins:{part}",
        license=license_name,
        data_use=DataUse.TRAINING,
    )


def load_native_designwins_text_pairs(
    source: Path,
    *,
    license_name: str = "internal-owned",
    destination: Path | None = None,
    strict: bool = True,
    rejections: list[dict[str, Any]] | None = None,
    eligible_parts: set[str] | None = None,
    exclusion_reasons: Mapping[str, str] | None = None,
) -> list[TextPair]:
    """Load the existing ``part/prompt/target`` MCU text-pair corpus."""

    source = Path(source).resolve(strict=True)
    pairs: list[TextPair] = []
    for line_number, row in enumerate(_records(source), 1):
        part = str(row.get("part") or "").strip()
        if eligible_parts is not None and part not in eligible_parts:
            if rejections is not None:
                rejections.append(
                    {
                        "line": line_number,
                        "part": part or "unknown",
                        "reason": (exclusion_reasons or {}).get(
                            part, "part is absent from the approved audit cohort"
                        ),
                    }
                )
            continue
        try:
            pairs.append(_native_designwins_text_pair(row, source, license_name))
        except (OSError, ValueError) as exc:
            if strict:
                raise
            if rejections is not None:
                rejections.append(
                    {
                        "line": line_number,
                        "part": str(row.get("part") or "unknown"),
                        "reason": str(exc),
                    }
                )
    output = deduplicate_pairs(pairs)
    if destination is not None:
        write_pairs_jsonl(destination, output)
    return output


def _native_designwins_text_pair(
    row: Mapping[str, Any],
    source: Path,
    license_name: str,
) -> TextPair:
    part = str(_required(row, "part")).strip()
    prompt = _clean_training_text(_required(row, "prompt"), field="prompt")
    response = _clean_training_text(_required(row, "target"), field="target")
    try:
        json.loads(response)
    except json.JSONDecodeError as exc:
        raise ExportValidationError(f"{part}: target is not valid JSON") from exc
    provenance = _native_designwins_provenance(
        row,
        source,
        part=part,
        modality="text",
        license_name=license_name,
    )
    return TextPair(
        pair_id=_stable_id("designwins-text", part, response),
        prompt=prompt,
        response=response,
        provenance=provenance,
        metadata={"part": part, "modality": "text"},
    )


def load_native_designwins_vision_pairs(
    source: Path,
    *,
    license_name: str = "internal-owned",
    destination: Path | None = None,
    strict: bool = True,
    rejections: list[dict[str, Any]] | None = None,
    eligible_parts: set[str] | None = None,
    exclusion_reasons: Mapping[str, str] | None = None,
) -> list[VisionPair]:
    """Load MCU vision pairs and compute image hashes from the source files."""

    source = Path(source).resolve(strict=True)
    source_root = source.parent
    pairs: list[VisionPair] = []
    for line_number, row in enumerate(_records(source), 1):
        part = str(row.get("part") or "").strip()
        if eligible_parts is not None and part not in eligible_parts:
            if rejections is not None:
                rejections.append(
                    {
                        "line": line_number,
                        "part": part or "unknown",
                        "reason": (exclusion_reasons or {}).get(
                            part, "part is absent from the approved audit cohort"
                        ),
                    }
                )
            continue
        try:
            pairs.append(
                _native_designwins_vision_pair(
                    row,
                    source,
                    source_root,
                    license_name,
                )
            )
        except (OSError, ValueError) as exc:
            if strict:
                raise
            if rejections is not None:
                rejections.append(
                    {
                        "line": line_number,
                        "part": str(row.get("part") or "unknown"),
                        "reason": str(exc),
                    }
                )
    if destination is not None:
        write_pairs_jsonl(destination, pairs)
    return pairs


def _native_designwins_vision_pair(
    row: Mapping[str, Any],
    source: Path,
    source_root: Path,
    license_name: str,
) -> VisionPair:
    part = str(_required(row, "part")).strip()
    prompt = _clean_training_text(_required(row, "prompt"), field="prompt")
    response = _clean_training_text(_required(row, "target"), field="target")
    try:
        json.loads(response)
    except json.JSONDecodeError as exc:
        raise ExportValidationError(f"{part}: target is not valid JSON") from exc
    raw_images = _required(row, "images")
    if not isinstance(raw_images, list) or not raw_images:
        raise ExportValidationError(f"{part}: images must be a non-empty list")
    image_uris: list[str] = []
    image_sha256: list[str] = []
    for raw_path in raw_images:
        image = Path(str(raw_path)).resolve(strict=True)
        if not image.is_file() or not image.is_relative_to(source_root):
            raise ExportValidationError(
                f"{part}: image is outside the DesignWins training root"
            )
        digest = hashlib.sha256()
        with image.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        image_uris.append(
            f"dataset://designwins/{image.relative_to(source_root).as_posix()}"
        )
        image_sha256.append(digest.hexdigest())
    provenance = _native_designwins_provenance(
        row,
        source,
        part=part,
        modality="vision",
        license_name=license_name,
    )
    return VisionPair(
        pair_id=_stable_id("designwins-vision", part, response, *image_sha256),
        prompt=prompt,
        response=response,
        provenance=provenance,
        image_uris=tuple(image_uris),
        image_sha256=tuple(image_sha256),
        metadata={"part": part, "modality": "vision"},
    )


def extract_git_candidates(source: RecordSource) -> list[GitCandidate]:
    """Validate generic git examples and retain them in quarantine."""

    output: list[GitCandidate] = []
    for row in _records(source):
        provenance = _provenance(
            row,
            source_kind=SourceKind.GIT,
            data_use=DataUse.QUARANTINE,
        )
        if not provenance.revision:
            raise ExportValidationError("git provenance requires a revision")
        problem = str(_required(row, "problem", "issue", "prompt")).strip()
        patch = str(_required(row, "patch", "diff")).strip()
        assert_no_secrets(problem, field="problem")
        assert_no_secrets(patch, field="patch")
        raw_tests = _required(row, "tests", "test_results")
        if not isinstance(raw_tests, list) or not raw_tests:
            raise ExportValidationError("git candidate requires test evidence")
        try:
            tests = tuple(TestEvidence.model_validate(test) for test in raw_tests)
            candidate = GitCandidate(
                candidate_id=str(
                    row.get("candidate_id")
                    or row.get("id")
                    or _stable_id(
                        "git", provenance.source_record_id, problem, patch
                    )
                ),
                problem=problem,
                patch=patch,
                tests=tests,
                provenance=provenance,
                quarantine_reason=str(
                    row.get("quarantine_reason")
                    or "generic git candidate requires human provenance and license review"
                ),
            )
        except ValidationError as exc:
            raise ExportValidationError("invalid git candidate") from exc
        output.append(candidate)
    return output


# Export-oriented names for callers that already know the input file names.
export_designwins_text_pairs = load_designwins_text_pairs
export_designwins_vision_pairs = load_designwins_vision_pairs
extract_harness_pass_pairs = export_harness_pass_pairs
