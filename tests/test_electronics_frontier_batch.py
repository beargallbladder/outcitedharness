from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from harness.electronics.corpus import sha256_file
from harness.electronics.frontier_batch import (
    FrontierCandidate,
    FrontierEvidence,
    FrontierTeacherVerification,
    LocalAttempt,
    build_preference_training_pairs,
    candidate_id,
    candidate_identity_payload,
    load_prepared_bundle,
    finalize_training_pairs,
    prepare_batch_bundle,
    reconcile_results,
    request_chunks,
)
from harness.electronics.models import PairCapability


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "datasheet_frontier_batch_cli",
    ROOT / "scripts" / "datasheet_frontier_batch.py",
)
assert SPEC is not None and SPEC.loader is not None
batch_cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch_cli)


def _candidate(
    evidence: Path,
    capability: PairCapability = PairCapability.PIN_SEMANTICS,
) -> FrontierCandidate:
    draft = FrontierCandidate(
        candidate_id="candidate-" + ("0" * 32),
        capability=capability,
        document_sha256="a" * 64,
        entity_hint="acme:ATOM1:LQFP2:pin1",
        prompt="Extract type and direction for pin 1.",
        response_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "direction"],
            "properties": {
                "type": {"type": "string"},
                "direction": {"type": "string"},
            },
        },
        evidence=(
            FrontierEvidence(
                path=evidence,
                sha256=sha256_file(evidence),
                media_type="image/png",
                page_1based=3,
                bbox=(1.0, 2.0, 3.0, 4.0),
            ),
        ),
        local_attempts=(
            LocalAttempt(
                provider="local",
                model="qwen3-vl-8b",
                status="low_confidence",
                receipt_sha256="b" * 64,
                output_sha256="c" * 64,
                reason="confidence below frozen threshold",
            ),
        ),
        estimated_input_tokens=1000,
        max_output_tokens=200,
    )
    return draft.model_copy(
        update={"candidate_id": candidate_id(candidate_identity_payload(draft))}
    )


def test_frontier_candidate_cannot_skip_local_attempts(tmp_path: Path):
    evidence = tmp_path / "row.png"
    evidence.write_bytes(b"png")
    with pytest.raises(ValidationError, match="requires local attempts"):
        FrontierCandidate(
            candidate_id="candidate-" + ("0" * 32),
            capability=PairCapability.PIN_SEMANTICS,
            document_sha256="a" * 64,
            entity_hint="pin1",
            prompt="Extract pin semantics.",
            response_schema={"type": "object"},
            evidence=(
                FrontierEvidence(
                    path=evidence,
                    sha256=sha256_file(evidence),
                    media_type="image/png",
                ),
            ),
            local_attempts=(),
            estimated_input_tokens=100,
        )


def test_frontier_candidate_rejects_pdf_extracted_prompt_text(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "row.png"
    evidence.write_bytes(b"png")
    with pytest.raises(
        ValidationError,
        match="cannot contain PDF-extracted page text",
    ):
        FrontierCandidate(
            candidate_id="candidate-" + ("0" * 32),
            capability=PairCapability.PIN_SEMANTICS,
            document_sha256="a" * 64,
            entity_hint="pin1",
            prompt="Extract pins.\n\nPyMuPDF evidence:\ncorrupted table text",
            response_schema={"type": "object"},
            evidence=(
                FrontierEvidence(
                    path=evidence,
                    sha256=sha256_file(evidence),
                    media_type="image/png",
                ),
            ),
            local_attempts=(
                LocalAttempt(
                    provider="local",
                    model="qwen3-vl-30b",
                    status="low_confidence",
                    receipt_sha256="b" * 64,
                    output_sha256="c" * 64,
                    reason="source gate failed",
                ),
            ),
            estimated_input_tokens=100,
        )


def test_frontier_candidate_accepts_grounded_local_output(tmp_path: Path) -> None:
    evidence = tmp_path / "row.png"
    evidence.write_bytes(b"png")
    candidate = _candidate(evidence).model_copy(
        update={
            "local_attempts": (
                LocalAttempt(
                    provider="local",
                    model="qwen3-vl-30b",
                    status="passed_evidence_gate",
                    receipt_sha256="b" * 64,
                    output_sha256="c" * 64,
                    reason="local output passed the source-evidence gate",
                ),
            )
        }
    )
    candidate = candidate.model_copy(
        update={"candidate_id": candidate_id(candidate_identity_payload(candidate))}
    )
    candidate = FrontierCandidate.model_validate(
        candidate.model_dump(mode="json", by_alias=True)
    )

    prepare_batch_bundle(
        tmp_path / "prepared",
        [candidate],
        model="claude-sonnet",
        allowed_roots=(tmp_path,),
        input_price_per_million=3,
        output_price_per_million=15,
        batch_discount=0.5,
        spend_cap_usd=1,
        created_at=datetime.now(timezone.utc),
    )


def test_batch_bundle_enforces_cost_cap_and_strips_private_metadata(
    tmp_path: Path,
):
    evidence = tmp_path / "row.png"
    evidence.write_bytes(b"png")
    candidate = _candidate(evidence)
    too_small = tmp_path / "too-small"
    with pytest.raises(ValueError, match="exceeds"):
        prepare_batch_bundle(
            too_small,
            [candidate],
            model="claude-sonnet",
            allowed_roots=(tmp_path,),
            input_price_per_million=3,
            output_price_per_million=15,
            batch_discount=0.5,
            spend_cap_usd=0.0001,
            created_at=datetime.now(timezone.utc),
        )
    output = tmp_path / "prepared"
    manifest = prepare_batch_bundle(
        output,
        [candidate],
        model="claude-sonnet",
        allowed_roots=(tmp_path,),
        input_price_per_million=3,
        output_price_per_million=15,
        batch_discount=0.5,
        spend_cap_usd=1,
        created_at=datetime.now(timezone.utc),
    )
    _, requests = load_prepared_bundle(output)
    chunks = request_chunks(
        requests,
        maximum_bytes=1024 * 1024,
        maximum_requests=100,
    )

    assert manifest["purpose"] == "local_training_pair_generation"
    assert manifest["pricing"]["batch_discount_multiplier"] == 0.5
    assert len(chunks) == 1
    assert "temperature" not in requests[0]["params"]
    assert requests[0]["params"]["thinking"] == {"type": "disabled"}
    assert "_harness" not in chunks[0][0]


def test_parametric_teacher_prompt_rejects_nonvalues(tmp_path: Path) -> None:
    evidence = tmp_path / "table.png"
    evidence.write_bytes(b"png")
    output = tmp_path / "prepared"
    prepare_batch_bundle(
        output,
        [_candidate(evidence, PairCapability.PARAMETRICS)],
        model="claude-sonnet",
        allowed_roots=(tmp_path,),
        input_price_per_million=3,
        output_price_per_million=15,
        batch_discount=0.5,
        spend_cap_usd=1,
        created_at=datetime.now(timezone.utc),
    )

    _, requests = load_prepared_bundle(output)
    system = requests[0]["params"]["system"]
    assert "omit blank cells and non-values" in system
    assert "never construct a field by joining cells" in system


def _submit_args(
    bundle: Path,
    state: Path,
    *,
    resume: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        bundle=bundle,
        state_directory=state,
        approved_spend_cap_usd=1.0,
        maximum_batch_bytes=1_000_000,
        maximum_batch_requests=100,
        resume=resume,
        api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.test",
        timeout_seconds=30,
    )


def test_submit_resume_reuses_receipt_without_second_network_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "row.png"
    evidence.write_bytes(b"png")
    bundle = tmp_path / "prepared"
    prepare_batch_bundle(
        bundle,
        [_candidate(evidence)],
        model="claude-sonnet",
        allowed_roots=(tmp_path,),
        input_price_per_million=3,
        output_price_per_million=15,
        batch_discount=0.5,
        spend_cap_usd=1,
        created_at=datetime.now(timezone.utc),
    )
    calls = []

    class Client:
        def submit(self, requests):
            calls.append(requests)
            return {"id": "msgbatch_once", "processing_status": "in_progress"}

    monkeypatch.setattr(batch_cli, "_client", lambda _args: Client())
    state = tmp_path / "submission"
    first = batch_cli._submit(_submit_args(bundle, state, resume=False))
    second = batch_cli._submit(_submit_args(bundle, state, resume=True))

    assert first["network_submissions"] == 1
    assert second["status"] == "resumed"
    assert second["network_submissions"] == 0
    assert second["batches"] == ["msgbatch_once"]
    assert len(calls) == 1


def test_submit_resume_blocks_ambiguous_attempt_without_resubmitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "row.png"
    evidence.write_bytes(b"png")
    bundle = tmp_path / "prepared"
    prepare_batch_bundle(
        bundle,
        [_candidate(evidence)],
        model="claude-sonnet",
        allowed_roots=(tmp_path,),
        input_price_per_million=3,
        output_price_per_million=15,
        batch_discount=0.5,
        spend_cap_usd=1,
        created_at=datetime.now(timezone.utc),
    )
    calls = 0

    class FailingClient:
        def submit(self, _requests):
            nonlocal calls
            calls += 1
            raise RuntimeError("connection ended after request write")

    monkeypatch.setattr(batch_cli, "_client", lambda _args: FailingClient())
    state = tmp_path / "submission"
    with pytest.raises(RuntimeError, match="connection ended"):
        batch_cli._submit(_submit_args(bundle, state, resume=False))
    assert (state / "chunk-0001.attempted.json").is_file()
    assert not (state / "chunk-0001.submitted.json").exists()

    with pytest.raises(ValueError, match="outcome is ambiguous"):
        batch_cli._submit(_submit_args(bundle, state, resume=True))
    assert calls == 1


def test_reconciliation_still_requires_claim_verification(tmp_path: Path):
    evidence = tmp_path / "row.png"
    evidence.write_bytes(b"png")
    candidate = _candidate(evidence)
    bundle = tmp_path / "prepared"
    prepare_batch_bundle(
        bundle,
        [candidate],
        model="claude-sonnet",
        allowed_roots=(tmp_path,),
        input_price_per_million=3,
        output_price_per_million=15,
        batch_discount=0.5,
        spend_cap_usd=1,
        created_at=datetime.now(timezone.utc),
    )
    _, requests = load_prepared_bundle(bundle)
    raw = (
        json.dumps(
            {
                "custom_id": requests[0]["custom_id"],
                "result": {
                    "type": "succeeded",
                    "message": {
                        "id": "msg_1",
                        "content": [
                            {
                                "type": "text",
                                "text": '{"type":"gpio","direction":"I/O"}',
                            }
                        ],
                        "usage": {
                            "input_tokens": 900,
                            "output_tokens": 20,
                        },
                    },
                },
            }
        ).encode()
        + b"\n"
    )

    report = reconcile_results(
        bundle,
        [("msgbatch_1", raw)],
        input_price_per_million=3,
        output_price_per_million=15,
        batch_discount=0.5,
    )

    assert report["complete"] is True
    assert report["admitted_to_training"] is False
    assert report["next_gate"] == "claim_level_verification"
    assert report["counts"]["statuses"] == {
        "ready_for_claim_verification": 1
    }

    response_sha = report["outcomes"][0]["response_sha256"]
    unreconstructed = FrontierTeacherVerification(
        verification_id="teacher-verify-" + ("a" * 32),
        candidate_id=candidate.candidate_id,
        response_sha256=response_sha,
        status="passed",
        verifier="source_evidence_rule",
        checks=("direction_matches_visible_table_cell",),
        claim_ids=("claim-" + ("b" * 32),),
        evidence_sha256=(sha256_file(evidence),),
    )
    unreconstructed_report, unreconstructed_pairs = finalize_training_pairs(
        bundle,
        report,
        [unreconstructed],
    )
    assert unreconstructed_pairs == []
    assert unreconstructed_report["counts"]["dispositions"] == {
        "verified_response_required": 1
    }

    verified_response = {"type": "gpio"}
    verified_response_sha = hashlib.sha256(
        json.dumps(
            verified_response,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    verification = FrontierTeacherVerification(
        verification_id="teacher-verify-" + ("d" * 32),
        candidate_id=candidate.candidate_id,
        response_sha256=response_sha,
        status="passed",
        verifier="source_evidence_rule",
        checks=("direction_matches_visible_table_cell",),
        claim_ids=("claim-" + ("e" * 32),),
        evidence_sha256=(sha256_file(evidence),),
        verified_response=verified_response,
        verified_response_sha256=verified_response_sha,
    )
    finalization, pairs = finalize_training_pairs(
        bundle,
        report,
        [verification],
    )

    assert finalization["counts"]["training_pairs"] == 1
    assert finalization["next_gate"] == "frozen_local_model_evaluation"
    assert pairs[0].purpose == "local_training_pair_generation"
    assert pairs[0].teacher is not None
    assert pairs[0].teacher.batch_id == "msgbatch_1"
    assert json.loads(pairs[0].response) == verified_response

    preference_report, preference_pairs = build_preference_training_pairs(
        bundle,
        report,
        [verification],
        pairs,
        [
            {
                "work_id": candidate.entity_hint,
                "model": "qwen3-vl-8b",
                "request_sha256": "f" * 64,
                "response_sha256": "c" * 64,
                "local_pillar_stage": "focused_local_vision",
                "result": {"type": "gpio", "direction": "input"},
            }
        ],
    )
    assert preference_report["counts"]["preference_pairs"] == 1
    assert preference_pairs[0].training_format == "vision_dpo"
    assert preference_pairs[0].chosen_source_sha256 == verified_response_sha
    assert preference_pairs[0].rejected_response != (
        preference_pairs[0].chosen_response
    )

    nonvision_report, nonvision_pairs = build_preference_training_pairs(
        bundle,
        report,
        [verification],
        pairs,
        [
            {
                "work_id": candidate.entity_hint,
                "model": "pymupdf-parametric-normalizer-v1",
                "request_sha256": "f" * 64,
                "response_sha256": "c" * 64,
                "local_pillar_stage": "deterministic_structure",
                "result": {"type": "gpio", "direction": "input"},
            }
        ],
    )
    assert nonvision_pairs == []
    assert nonvision_report["counts"]["dispositions"] == {
        "nonvision_local_response": 1
    }
