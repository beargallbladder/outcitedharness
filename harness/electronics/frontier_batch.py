"""Durable Anthropic Message Batch workflow for local-model teaching data."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal, Mapping, Sequence

import httpx
from pydantic import Field, field_validator, model_validator

from harness.electronics.claims import canonical_json, stable_id
from harness.electronics.models import (
    ModelIdentity,
    PairCapability,
    PairDisposition,
    PairModality,
    PreferenceTrainingPairCandidate,
    Sha256,
    StrictModel,
    TrainingPairCandidate,
)


PREPARED_SCHEMA = "harness.electronics-frontier-batch.v1"
SUBMISSION_SCHEMA = "harness.electronics-frontier-submission.v1"
RESULT_SCHEMA = "harness.electronics-frontier-result.v1"
RECONCILIATION_SCHEMA = "harness.electronics-frontier-reconciliation.v1"
TERMINAL_LOCAL_STATUSES = {
    "no_answer",
    "schema_failed",
    "low_confidence",
    "cross_source_disagreement",
}
COMPLETED_LOCAL_STATUSES = TERMINAL_LOCAL_STATUSES | {"passed_evidence_gate"}


class LocalAttempt(StrictModel):
    provider: Literal["local"]
    model: Annotated[str, Field(min_length=1)]
    status: Literal[
        "passed_evidence_gate",
        "no_answer",
        "schema_failed",
        "low_confidence",
        "cross_source_disagreement",
    ]
    receipt_sha256: Sha256
    output_sha256: Sha256 | None = None
    reason: Annotated[str, Field(min_length=1)]


class FrontierEvidence(StrictModel):
    path: Path
    sha256: Sha256
    media_type: Literal[
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/webp",
    ]
    page_1based: Annotated[int | None, Field(gt=0)] = None
    bbox: tuple[float, float, float, float] | None = None

    @field_validator("bbox")
    @classmethod
    def valid_bbox(
        cls,
        value: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float, float] | None:
        if value is not None:
            x0, y0, x1, y1 = value
            if not all(math.isfinite(item) for item in value):
                raise ValueError("bbox contains non-finite coordinates")
            if x1 <= x0 or y1 <= y0:
                raise ValueError("bbox coordinates are not increasing")
        return value

    @model_validator(mode="after")
    def region_is_page_bound(self) -> FrontierEvidence:
        if self.bbox is not None and self.page_1based is None:
            raise ValueError("bbox evidence requires page_1based")
        return self


class FrontierCandidate(StrictModel):
    schema_name: Literal[
        "harness.electronics-frontier-candidate.v1"
    ] = Field(
        default="harness.electronics-frontier-candidate.v1",
        alias="schema",
        serialization_alias="schema",
    )
    candidate_id: Annotated[str, Field(pattern=r"^candidate-[0-9a-f]{32}$")]
    purpose: Literal["local_training_pair_generation"] = (
        "local_training_pair_generation"
    )
    capability: PairCapability
    document_sha256: Sha256
    entity_hint: Annotated[str, Field(min_length=1)]
    prompt: Annotated[str, Field(min_length=1)]
    response_schema: dict[str, Any]
    evidence: tuple[FrontierEvidence, ...]
    local_attempts: tuple[LocalAttempt, ...]
    estimated_input_tokens: Annotated[int, Field(gt=0)]
    max_output_tokens: Annotated[int, Field(gt=0, le=16384)] = 4096

    @field_validator("response_schema")
    @classmethod
    def object_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") != "object":
            raise ValueError("response_schema must describe a JSON object")
        json.dumps(value, ensure_ascii=False, allow_nan=False)
        return value

    @model_validator(mode="after")
    def local_first_and_evidence_backed(self) -> FrontierCandidate:
        if not self.evidence:
            raise ValueError("frontier escalation requires source evidence")
        if not any(
            evidence.media_type.startswith("image/")
            for evidence in self.evidence
        ):
            raise ValueError(
                "frontier extraction requires rendered page-image evidence"
            )
        if any(
            marker in self.prompt
            for marker in (
                "PyMuPDF evidence:",
                "Extracted page evidence:",
            )
        ):
            raise ValueError(
                "frontier prompt cannot contain PDF-extracted page text"
            )
        if not self.local_attempts:
            raise ValueError("frontier escalation requires local attempts")
        if not all(
            attempt.status in COMPLETED_LOCAL_STATUSES
            for attempt in self.local_attempts
        ):
            raise ValueError("local attempts must have a completed model status")
        return self


class FrontierTeacherVerification(StrictModel):
    schema_name: Literal[
        "harness.electronics-frontier-teacher-verification.v1"
    ] = Field(
        default="harness.electronics-frontier-teacher-verification.v1",
        alias="schema",
        serialization_alias="schema",
    )
    verification_id: Annotated[
        str,
        Field(pattern=r"^teacher-verify-[0-9a-f]{32}$"),
    ]
    candidate_id: Annotated[str, Field(pattern=r"^candidate-[0-9a-f]{32}$")]
    response_sha256: Sha256
    status: Literal["passed", "failed", "quarantined"]
    verifier: Literal["source_evidence_rule", "local_consensus", "human"]
    checks: tuple[Annotated[str, Field(min_length=1)], ...]
    claim_ids: tuple[
        Annotated[str, Field(pattern=r"^claim-[0-9a-f]{32}$")], ...
    ] = ()
    evidence_sha256: tuple[Sha256, ...]
    verified_response: dict[str, Any] | None = None
    verified_response_sha256: Sha256 | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def passed_teacher_has_claims(self) -> FrontierTeacherVerification:
        if not self.checks or not self.evidence_sha256:
            raise ValueError("teacher verification requires checks and evidence")
        if (self.verified_response is None) != (
            self.verified_response_sha256 is None
        ):
            raise ValueError(
                "verified response and its SHA-256 must be supplied together"
            )
        if self.verified_response is not None:
            if self.status != "passed":
                raise ValueError(
                    "only a passed teacher may carry a verified response"
                )
            expected = hashlib.sha256(
                canonical_json(self.verified_response)
            ).hexdigest()
            if self.verified_response_sha256 != expected:
                raise ValueError("verified response SHA-256 mismatch")
        if self.status == "passed":
            if not self.claim_ids:
                raise ValueError("passed teacher verification requires claim IDs")
            if self.reason is not None:
                raise ValueError("passed teacher verification cannot have a reason")
        elif not self.reason:
            raise ValueError("failed teacher verification requires a reason")
        return self


def candidate_identity_payload(candidate: FrontierCandidate) -> dict[str, Any]:
    value = candidate.model_dump(mode="json", by_alias=True)
    value.pop("candidate_id", None)
    return value


def verify_candidate_identity(candidate: FrontierCandidate) -> None:
    expected = "candidate-" + hashlib.sha256(
        canonical_json(candidate_identity_payload(candidate))
    ).hexdigest()[:32]
    if candidate.candidate_id != expected:
        raise ValueError(f"candidate identity mismatch: {candidate.candidate_id}")


def candidate_id(value: Mapping[str, Any]) -> str:
    return "candidate-" + hashlib.sha256(canonical_json(value)).hexdigest()[:32]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_evidence(
    evidence: FrontierEvidence,
    allowed_roots: Sequence[Path],
) -> tuple[Path, bytes]:
    path = evidence.path.expanduser().resolve(strict=True)
    roots = [root.expanduser().resolve(strict=True) for root in allowed_roots]
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evidence is not a regular file: {path}")
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError(f"evidence is outside allowed roots: {path}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != evidence.sha256:
        raise ValueError(f"evidence SHA-256 mismatch: {path}")
    return path, payload


def _request(
    candidate: FrontierCandidate,
    *,
    model: str,
    allowed_roots: Sequence[Path],
) -> dict[str, Any]:
    verify_candidate_identity(candidate)
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"{candidate.prompt}\n\nReturn only JSON matching this schema:\n"
                + json.dumps(
                    candidate.response_schema,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        }
    ]
    evidence_receipts: list[dict[str, Any]] = []
    for evidence in candidate.evidence:
        path, payload = _resolved_evidence(evidence, allowed_roots)
        source = {
            "type": "base64",
            "media_type": evidence.media_type,
            "data": base64.b64encode(payload).decode("ascii"),
        }
        if evidence.media_type == "application/pdf":
            content.append({"type": "document", "source": source})
        else:
            content.append({"type": "image", "source": source})
        evidence_receipts.append(
            {
                "path": str(path),
                "sha256": evidence.sha256,
                "media_type": evidence.media_type,
                "page_1based": evidence.page_1based,
                "bbox": evidence.bbox,
            }
        )
    custom_id = stable_id("pair", candidate_identity_payload(candidate)).replace(
        "pair-", "teach-"
    )
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": candidate.max_output_tokens,
            "thinking": {"type": "disabled"},
            "system": (
                "You extract electronics facts from supplied source evidence. "
                "Do not infer an absent value. Preserve printed units and package "
                "identity. Your output is teacher data for a local model and will "
                "be independently verified before admission."
                + (
                    " For pin semantics, omit any row whose selected-package "
                    "pin or ball cell is blank, -, —, or N/A. Use dir only for "
                    "a printed direction or pin-category value such as I, O, "
                    "I/O, P, or S. Use type only for a printed electrical I/O "
                    "structure or level. Use supply_domain only for a value "
                    "under an explicit supply-domain header. Put Description "
                    "and Function column values in functions. Never copy a "
                    "description or supply domain into type. Every semantic "
                    "value must come from the matching identity row and its "
                    "corresponding semantic column."
                    if candidate.capability == PairCapability.PIN_SEMANTICS
                    else ""
                )
                + (
                    " For parametrics, omit blank cells and non-values such as "
                    "-, —, or N/A. Copy field exactly from one printed Parameter "
                    "or Symbol cell; never construct a field by joining cells. "
                    "Put printed size, test, and operating qualifiers in "
                    "conditions. Copy every condition value verbatim; do not "
                    "paraphrase, expand an acronym, interpret a label, or combine "
                    "qualifiers from separate cells. Omit any fact that requires "
                    "such a transformation. Emit only facts whose field, value, "
                    "role, unit, and conditions can be traced to the same table "
                    "row and its headers."
                    if candidate.capability == PairCapability.PARAMETRICS
                    else ""
                )
                + (
                    " For series summaries, copy each characteristic and "
                    "application nearly verbatim from visible source text. "
                    "Do not add competitors, recommendations, positioning, or "
                    "facts from outside the supplied page."
                    if candidate.capability == PairCapability.SERIES_SUMMARY
                    else ""
                )
                + (
                    " For OPN decoding, emit only segments, suffix codes, and "
                    "meanings explicitly printed in the supplied ordering or "
                    "nomenclature evidence. Use JSON null for unstated scalar "
                    "segments and never infer a code meaning."
                    if candidate.capability == PairCapability.OPN_DECODER
                    else ""
                )
            ),
            "messages": [{"role": "user", "content": content}],
        },
        "_harness": {
            "candidate_id": candidate.candidate_id,
            "purpose": candidate.purpose,
            "capability": candidate.capability.value,
            "document_sha256": candidate.document_sha256,
            "evidence": evidence_receipts,
            "local_attempt_receipts": [
                attempt.receipt_sha256 for attempt in candidate.local_attempts
            ],
        },
    }


def _write(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)


def prepare_batch_bundle(
    destination: Path,
    candidates: Iterable[FrontierCandidate],
    *,
    model: str,
    allowed_roots: Sequence[Path],
    input_price_per_million: float,
    output_price_per_million: float,
    batch_discount: float,
    spend_cap_usd: float,
    created_at: datetime,
) -> dict[str, Any]:
    if not model.strip():
        raise ValueError("model cannot be blank")
    for value, name in (
        (input_price_per_million, "input_price_per_million"),
        (output_price_per_million, "output_price_per_million"),
        (spend_cap_usd, "spend_cap_usd"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(batch_discount) or not 0 < batch_discount <= 1:
        raise ValueError("batch_discount must be within (0, 1]")
    output = destination.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"batch bundle already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        candidate_path = temporary / "candidates.jsonl"
        requests_path = temporary / "requests.jsonl"
        seen_candidates: set[str] = set()
        seen_custom_ids: set[str] = set()
        total_input = 0
        total_output = 0
        capability_counts: Counter[str] = Counter()
        with candidate_path.open("xb") as candidate_handle, requests_path.open(
            "xb"
        ) as request_handle:
            for candidate in candidates:
                verify_candidate_identity(candidate)
                if candidate.candidate_id in seen_candidates:
                    raise ValueError(
                        f"duplicate candidate: {candidate.candidate_id}"
                    )
                request = _request(
                    candidate,
                    model=model,
                    allowed_roots=allowed_roots,
                )
                if request["custom_id"] in seen_custom_ids:
                    raise ValueError(
                        f"duplicate custom_id: {request['custom_id']}"
                    )
                seen_candidates.add(candidate.candidate_id)
                seen_custom_ids.add(request["custom_id"])
                total_input += candidate.estimated_input_tokens
                total_output += candidate.max_output_tokens
                capability_counts[candidate.capability.value] += 1
                candidate_handle.write(
                    canonical_json(
                        candidate.model_dump(mode="json", by_alias=True)
                    )
                    + b"\n"
                )
                # _harness is retained in the sealed source bundle but removed
                # from the Anthropic payload at submission.
                request_handle.write(canonical_json(request) + b"\n")
            for handle in (candidate_handle, request_handle):
                handle.flush()
                os.fsync(handle.fileno())
        estimated_cost = batch_discount * (
            total_input * input_price_per_million / 1_000_000
            + total_output * output_price_per_million / 1_000_000
        )
        if not seen_candidates:
            raise ValueError("cannot prepare an empty frontier batch")
        if estimated_cost > spend_cap_usd:
            raise ValueError(
                f"estimated batch cost ${estimated_cost:.6f} exceeds "
                f"${spend_cap_usd:.6f} cap"
            )
        core = {
            "schema": PREPARED_SCHEMA,
            "purpose": "local_training_pair_generation",
            "model": model,
            "pricing": {
                "input_per_million_usd": input_price_per_million,
                "output_per_million_usd": output_price_per_million,
                "batch_discount_multiplier": batch_discount,
                "spend_cap_usd": spend_cap_usd,
                "estimated_maximum_usd": estimated_cost,
            },
            "token_budget": {
                "estimated_input_tokens": total_input,
                "maximum_output_tokens": total_output,
            },
            "counts": {
                "requests": len(seen_candidates),
                "capabilities": dict(sorted(capability_counts.items())),
            },
            "artifacts": {
                "candidates.jsonl": {
                    "sha256": _sha256(candidate_path),
                    "bytes": candidate_path.stat().st_size,
                },
                "requests.jsonl": {
                    "sha256": _sha256(requests_path),
                    "bytes": requests_path.stat().st_size,
                },
            },
        }
        core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
        manifest = {"created_at": created_at.isoformat(), **core}
        _write(
            temporary / "manifest.json",
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n",
        )
        for path in (candidate_path, requests_path):
            os.chmod(path, 0o444)
        os.chmod(temporary, 0o555)
        os.rename(temporary, output)
        return manifest
    except BaseException:
        try:
            os.chmod(temporary, 0o755)
        except OSError:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_prepared_bundle(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = path.expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != PREPARED_SCHEMA:
        raise ValueError("prepared batch schema is not supported")
    for name, receipt in manifest.get("artifacts", {}).items():
        artifact = root / name
        if _sha256(artifact) != receipt.get("sha256"):
            raise ValueError(f"prepared batch artifact hash mismatch: {name}")
        if artifact.stat().st_size != receipt.get("bytes"):
            raise ValueError(f"prepared batch artifact size mismatch: {name}")
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_at", "evidence_sha256"}
    }
    if manifest.get("evidence_sha256") != hashlib.sha256(
        canonical_json(core)
    ).hexdigest():
        raise ValueError("prepared batch evidence digest mismatch")
    requests: list[dict[str, Any]] = []
    with (root / "requests.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict) or "custom_id" not in value:
                raise ValueError(f"invalid request at line {line_number}")
            requests.append(value)
    if len(requests) != manifest.get("counts", {}).get("requests"):
        raise ValueError("prepared request count mismatch")
    return manifest, requests


def request_chunks(
    requests: Sequence[Mapping[str, Any]],
    *,
    maximum_bytes: int,
    maximum_requests: int,
) -> list[list[dict[str, Any]]]:
    if maximum_bytes < 1024 or maximum_requests < 1:
        raise ValueError("invalid batch chunk limits")
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = len(b'{"requests":[]}')
    for source in requests:
        request = {
            key: value for key, value in source.items() if key != "_harness"
        }
        request_bytes = len(canonical_json(request)) + 1
        if request_bytes + len(b'{"requests":[]}') > maximum_bytes:
            raise ValueError(
                f"request {request.get('custom_id')} exceeds batch byte limit"
            )
        if current and (
            len(current) >= maximum_requests
            or current_bytes + request_bytes > maximum_bytes
        ):
            chunks.append(current)
            current = []
            current_bytes = len(b'{"requests":[]}')
        current.append(request)
        current_bytes += request_bytes
    if current:
        chunks.append(current)
    return chunks


class AnthropicBatchClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        timeout_s: float = 900,
        transport: httpx.BaseTransport | None = None,
    ):
        if not api_key:
            raise ValueError("Anthropic API key is required")
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": api_key,
        }
        self.timeout_s = timeout_s
        self.transport = transport

    def submit(self, requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        with httpx.Client(
            timeout=self.timeout_s,
            transport=self.transport,
        ) as client:
            response = client.post(
                f"{self.base_url}/v1/messages/batches",
                headers=self.headers,
                json={"requests": list(requests)},
            )
            response.raise_for_status()
            value = response.json()
        if not isinstance(value, dict) or not value.get("id"):
            raise ValueError("Anthropic batch response is missing an ID")
        return value

    def status(self, batch_id: str) -> dict[str, Any]:
        with httpx.Client(
            timeout=self.timeout_s,
            transport=self.transport,
        ) as client:
            response = client.get(
                f"{self.base_url}/v1/messages/batches/{batch_id}",
                headers=self.headers,
            )
            response.raise_for_status()
            value = response.json()
        if not isinstance(value, dict) or value.get("id") != batch_id:
            raise ValueError("Anthropic returned the wrong batch identity")
        return value

    def results(self, results_url: str) -> bytes:
        with httpx.Client(
            timeout=self.timeout_s,
            transport=self.transport,
        ) as client:
            response = client.get(results_url, headers=self.headers)
            response.raise_for_status()
            return response.content


def extract_json_response(message: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    content = message.get("content")
    if not isinstance(content, list):
        raise ValueError("frontier message content is not a list")
    text = "".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, Mapping) and block.get("type") == "text"
    ).strip()
    if text.startswith("```"):
        text = re_fenced_json(text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("frontier response must be a JSON object")
    return value, text


def re_fenced_json(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1] == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _matches_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    expected = schema.get("type")
    checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "number": lambda item: isinstance(item, (int, float))
        and not isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if isinstance(expected, str):
        check = checks.get(expected)
        if check is None:
            raise ValueError(f"{path}: unsupported schema type {expected!r}")
        if not check(value):
            raise ValueError(f"{path}: expected {expected}")
    elif isinstance(expected, list):
        if not any(
            schema_type in checks and checks[schema_type](value)
            for schema_type in expected
        ):
            raise ValueError(f"{path}: value does not match allowed types")
    if isinstance(value, dict):
        required = schema.get("required") or []
        if not isinstance(required, list) or not all(
            isinstance(name, str) for name in required
        ):
            raise ValueError(f"{path}: invalid required declaration")
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(f"{path}: missing required fields {missing}")
        properties = schema.get("properties") or {}
        if not isinstance(properties, Mapping):
            raise ValueError(f"{path}: properties is not an object")
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(properties)
            if extras:
                raise ValueError(f"{path}: unexpected fields {sorted(extras)}")
        for name, child in value.items():
            child_schema = properties.get(name)
            if isinstance(child_schema, Mapping):
                _matches_schema(child, child_schema, f"{path}.{name}")
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, child in enumerate(value):
            _matches_schema(child, schema["items"], f"{path}[{index}]")


def reconcile_results(
    prepared_bundle: Path,
    raw_results: Iterable[tuple[str, bytes]],
    *,
    input_price_per_million: float,
    output_price_per_million: float,
    batch_discount: float,
) -> dict[str, Any]:
    manifest, requests = load_prepared_bundle(prepared_bundle)
    expected = {request["custom_id"]: request for request in requests}
    candidates: dict[str, dict[str, Any]] = {}
    candidate_path = Path(prepared_bundle) / "candidates.jsonl"
    with candidate_path.open(encoding="utf-8") as handle:
        for line in handle:
            candidate = json.loads(line)
            candidates[candidate["candidate_id"]] = candidate
    seen: set[str] = set()
    duplicate: list[str] = []
    unexpected: list[str] = []
    outcomes: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    raw_receipts: list[dict[str, Any]] = []
    for batch_id, payload in raw_results:
        raw_receipts.append(
            {
                "batch_id": batch_id,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
        for line_number, line in enumerate(payload.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{batch_id}: invalid result line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{batch_id}: result line is not an object")
            custom_id = str(row.get("custom_id") or "")
            if custom_id not in expected:
                unexpected.append(custom_id)
                continue
            if custom_id in seen:
                duplicate.append(custom_id)
                continue
            seen.add(custom_id)
            request = expected[custom_id]
            result = row.get("result")
            outcome: dict[str, Any] = {
                "custom_id": custom_id,
                "candidate_id": request["_harness"]["candidate_id"],
                "batch_id": batch_id,
            }
            if not isinstance(result, Mapping):
                outcome.update(
                    status="malformed_result",
                    reason="result is not an object",
                )
                outcomes.append(outcome)
                continue
            if result.get("type") != "succeeded":
                outcome.update(
                    status=str(result.get("type") or "unknown_failure"),
                    reason=json.dumps(
                        result.get("error") or {},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
                outcomes.append(outcome)
                continue
            message = result.get("message")
            if not isinstance(message, Mapping):
                outcome.update(
                    status="malformed_result",
                    reason="succeeded result is missing message",
                )
                outcomes.append(outcome)
                continue
            usage = message.get("usage")
            if isinstance(usage, Mapping):
                input_tokens += int(usage.get("input_tokens") or 0)
                output_tokens += int(usage.get("output_tokens") or 0)
            try:
                response, response_text = extract_json_response(message)
                candidate_schema = candidates[
                    request["_harness"]["candidate_id"]
                ]["response_schema"]
                _matches_schema(response, candidate_schema)
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                outcome.update(
                    status="schema_failed",
                    reason=str(exc),
                )
            else:
                outcome.update(
                    status="ready_for_claim_verification",
                    response=response,
                    response_sha256=hashlib.sha256(
                        response_text.encode("utf-8")
                    ).hexdigest(),
                    message_id=message.get("id"),
                )
            outcomes.append(outcome)
    missing = sorted(set(expected) - seen)
    status_counts = Counter(outcome["status"] for outcome in outcomes)
    actual_cost = batch_discount * (
        input_tokens * input_price_per_million / 1_000_000
        + output_tokens * output_price_per_million / 1_000_000
    )
    core = {
        "schema": RECONCILIATION_SCHEMA,
        "purpose": "local_training_pair_generation",
        "prepared_evidence_sha256": manifest["evidence_sha256"],
        "raw_results": sorted(raw_receipts, key=lambda item: item["batch_id"]),
        "counts": {
            "expected": len(expected),
            "seen": len(seen),
            "missing": len(missing),
            "duplicate": len(duplicate),
            "unexpected": len(unexpected),
            "statuses": dict(sorted(status_counts.items())),
        },
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "actual_batch_cost_usd": actual_cost,
        },
        "missing_custom_ids": missing,
        "duplicate_custom_ids": sorted(duplicate),
        "unexpected_custom_ids": sorted(unexpected),
        "outcomes": sorted(outcomes, key=lambda item: item["custom_id"]),
        "complete": not missing and not duplicate and not unexpected,
        "admitted_to_training": False,
        "next_gate": "claim_level_verification",
    }
    core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
    return core


def finalize_training_pairs(
    prepared_bundle: Path,
    reconciliation: Mapping[str, Any],
    verifications: Iterable[FrontierTeacherVerification],
) -> tuple[dict[str, Any], list[TrainingPairCandidate]]:
    if reconciliation.get("schema") != RECONCILIATION_SCHEMA:
        raise ValueError("reconciliation schema is not supported")
    reconciliation_core = {
        key: value
        for key, value in reconciliation.items()
        if key not in {"created_at", "evidence_sha256"}
    }
    if reconciliation.get("evidence_sha256") != hashlib.sha256(
        canonical_json(reconciliation_core)
    ).hexdigest():
        raise ValueError("reconciliation evidence digest mismatch")
    manifest, requests = load_prepared_bundle(prepared_bundle)
    request_by_candidate = {
        request["_harness"]["candidate_id"]: request for request in requests
    }
    candidates: dict[str, FrontierCandidate] = {}
    with (Path(prepared_bundle) / "candidates.jsonl").open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            candidate = FrontierCandidate.model_validate_json(line)
            candidates[candidate.candidate_id] = candidate
    outcomes = {
        outcome["candidate_id"]: outcome
        for outcome in reconciliation.get("outcomes") or []
        if outcome.get("status") == "ready_for_claim_verification"
    }
    verification_by_candidate: dict[str, FrontierTeacherVerification] = {}
    for verification in verifications:
        if verification.candidate_id in verification_by_candidate:
            raise ValueError(
                f"duplicate verification: {verification.candidate_id}"
            )
        verification_by_candidate[verification.candidate_id] = verification

    pairs: list[TrainingPairCandidate] = []
    dispositions: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    for candidate_id_value, candidate in sorted(candidates.items()):
        outcome = outcomes.get(candidate_id_value)
        verification = verification_by_candidate.get(candidate_id_value)
        if outcome is None:
            dispositions["no_valid_response"] += 1
            details.append(
                {
                    "candidate_id": candidate_id_value,
                    "status": "no_valid_response",
                }
            )
            continue
        if verification is None:
            dispositions["awaiting_verification"] += 1
            details.append(
                {
                    "candidate_id": candidate_id_value,
                    "status": "awaiting_verification",
                }
            )
            continue
        if verification.response_sha256 != outcome["response_sha256"]:
            raise ValueError(
                f"verification response mismatch: {candidate_id_value}"
            )
        if verification.status != "passed":
            dispositions[verification.status] += 1
            details.append(
                {
                    "candidate_id": candidate_id_value,
                    "status": verification.status,
                    "reason": verification.reason,
                }
            )
            continue
        if verification.verified_response is None:
            dispositions["verified_response_required"] += 1
            details.append(
                {
                    "candidate_id": candidate_id_value,
                    "status": "verified_response_required",
                    "reason": (
                        "training admission requires a source-rebuilt "
                        "teacher response"
                    ),
                }
            )
            continue
        image_evidence = [
            evidence
            for evidence in candidate.evidence
            if evidence.media_type.startswith("image/")
        ]
        if not image_evidence:
            dispositions["render_required"] += 1
            details.append(
                {
                    "candidate_id": candidate_id_value,
                    "status": "render_required",
                    "reason": (
                        "PDF teacher responses require a sealed rendered-page "
                        "artifact before local vision training"
                    ),
                }
            )
            continue
        image_uris: list[str] = []
        image_hashes: list[str] = []
        for evidence in image_evidence:
            path = evidence.path.expanduser().resolve(strict=True)
            if _sha256(path) != evidence.sha256:
                raise ValueError(f"training image changed: {path}")
            image_uris.append(path.as_uri())
            image_hashes.append(evidence.sha256)
        request = request_by_candidate[candidate_id_value]
        teacher = ModelIdentity(
            provider="anthropic",
            model=manifest["model"],
            request_sha256=hashlib.sha256(
                canonical_json(request["params"])
            ).hexdigest(),
            response_id=outcome.get("message_id"),
            batch_id=outcome["batch_id"],
        )
        verified_response = verification.verified_response
        pair_core = {
            "purpose": "local_training_pair_generation",
            "capability": candidate.capability.value,
            "modality": PairModality.VISION.value,
            "prompt": candidate.prompt,
            "response": json.dumps(
                verified_response,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "source_claim_ids": list(verification.claim_ids),
            "lineage_ids": [candidate.document_sha256],
            "image_uris": image_uris,
            "image_sha256": image_hashes,
            "teacher": teacher.model_dump(mode="json", by_alias=True),
            "disposition": PairDisposition.ADMITTED.value,
            "quarantine_reason": None,
        }
        pair = TrainingPairCandidate(
            pair_id=stable_id("pair", pair_core),
            capability=candidate.capability,
            modality=PairModality.VISION,
            prompt=candidate.prompt,
            response=pair_core["response"],
            source_claim_ids=verification.claim_ids,
            lineage_ids=(candidate.document_sha256,),
            image_uris=tuple(image_uris),
            image_sha256=tuple(image_hashes),
            teacher=teacher,
            disposition=PairDisposition.ADMITTED,
        )
        pairs.append(pair)
        dispositions["admitted"] += 1
        details.append(
            {
                "candidate_id": candidate_id_value,
                "status": "admitted",
                "pair_id": pair.pair_id,
                "verification_id": verification.verification_id,
            }
        )
    core = {
        "schema": "harness.electronics-frontier-finalization.v1",
        "purpose": "local_training_pair_generation",
        "prepared_evidence_sha256": manifest["evidence_sha256"],
        "reconciliation_evidence_sha256": reconciliation["evidence_sha256"],
        "counts": {
            "candidates": len(candidates),
            "training_pairs": len(pairs),
            "dispositions": dict(sorted(dispositions.items())),
        },
        "details": details,
        "next_gate": "frozen_local_model_evaluation",
    }
    core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
    return core, pairs


def build_preference_training_pairs(
    prepared_bundle: Path,
    reconciliation: Mapping[str, Any],
    verifications: Iterable[FrontierTeacherVerification],
    sft_pairs: Iterable[TrainingPairCandidate],
    local_results: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[PreferenceTrainingPairCandidate]]:
    """Pair each local attempt with its verified frontier correction."""

    manifest, _requests = load_prepared_bundle(prepared_bundle)
    candidates: dict[str, FrontierCandidate] = {}
    with (Path(prepared_bundle) / "candidates.jsonl").open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            candidate = FrontierCandidate.model_validate_json(line)
            candidates[candidate.candidate_id] = candidate
    outcomes = {
        outcome["candidate_id"]: outcome
        for outcome in reconciliation.get("outcomes") or []
        if outcome.get("status") == "ready_for_claim_verification"
    }
    verified = {
        value.candidate_id: value
        for value in verifications
        if value.status == "passed"
    }
    sft_by_key = {
        (pair.lineage_ids[0], pair.prompt): pair
        for pair in sft_pairs
    }
    local_by_work = {
        str(value.get("work_id")): value
        for value in local_results
        if value.get("work_id")
    }
    pairs: list[PreferenceTrainingPairCandidate] = []
    dispositions: Counter[str] = Counter()
    for candidate_id_value, candidate in sorted(candidates.items()):
        verification = verified.get(candidate_id_value)
        outcome = outcomes.get(candidate_id_value)
        sft = sft_by_key.get((candidate.document_sha256, candidate.prompt))
        local = local_by_work.get(candidate.entity_hint)
        if verification is None or outcome is None or sft is None:
            dispositions["not_teacher_admitted"] += 1
            continue
        if sft.teacher is None:
            raise ValueError(
                f"SFT pair has no teacher identity: {candidate_id_value}"
            )
        if local is None or not isinstance(local.get("result"), Mapping):
            dispositions["local_response_unavailable"] += 1
            continue
        if local.get("local_pillar_stage") != "focused_local_vision":
            dispositions["nonvision_local_response"] += 1
            continue
        local_source_sha = str(local.get("response_sha256") or "")
        if not any(
            attempt.output_sha256 == local_source_sha
            for attempt in candidate.local_attempts
        ):
            raise ValueError(
                f"local response hash mismatch: {candidate_id_value}"
            )
        rejected = json.dumps(
            local["result"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        chosen = sft.response
        chosen_source_sha256 = (
            verification.verified_response_sha256
            or outcome["response_sha256"]
        )
        if chosen == rejected:
            dispositions["teacher_agreed_exactly"] += 1
            continue
        local_model = ModelIdentity(
            provider="local",
            model=str(local["model"]),
            request_sha256=str(local["request_sha256"]),
        )
        core = {
            "purpose": "local_training_pair_generation",
            "training_format": "vision_dpo",
            "capability": candidate.capability.value,
            "prompt": candidate.prompt,
            "chosen_response": chosen,
            "rejected_response": rejected,
            "chosen_source_sha256": chosen_source_sha256,
            "rejected_source_sha256": local_source_sha,
            "source_claim_ids": list(verification.claim_ids),
            "lineage_ids": list(sft.lineage_ids),
            "image_uris": list(sft.image_uris),
            "image_sha256": list(sft.image_sha256),
            "local_model": local_model.model_dump(mode="json"),
            "teacher": sft.teacher.model_dump(mode="json"),
            "disposition": PairDisposition.ADMITTED.value,
        }
        pairs.append(
            PreferenceTrainingPairCandidate(
                pair_id=stable_id("pair", core),
                capability=candidate.capability,
                prompt=candidate.prompt,
                chosen_response=chosen,
                rejected_response=rejected,
                chosen_source_sha256=chosen_source_sha256,
                rejected_source_sha256=local_source_sha,
                source_claim_ids=verification.claim_ids,
                lineage_ids=sft.lineage_ids,
                image_uris=sft.image_uris,
                image_sha256=sft.image_sha256,
                local_model=local_model,
                teacher=sft.teacher,
            )
        )
        dispositions["admitted"] += 1
    core = {
        "schema": "harness.electronics-frontier-preference-finalization.v1",
        "purpose": "local_training_pair_generation",
        "prepared_evidence_sha256": manifest["evidence_sha256"],
        "reconciliation_evidence_sha256": reconciliation[
            "evidence_sha256"
        ],
        "counts": {
            "preference_pairs": len(pairs),
            "dispositions": dict(sorted(dispositions.items())),
        },
    }
    core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
    return core, pairs


__all__ = [
    "AnthropicBatchClient",
    "FrontierCandidate",
    "FrontierEvidence",
    "FrontierTeacherVerification",
    "LocalAttempt",
    "build_preference_training_pairs",
    "candidate_id",
    "load_prepared_bundle",
    "finalize_training_pairs",
    "prepare_batch_bundle",
    "reconcile_results",
    "request_chunks",
    "verify_candidate_identity",
]
