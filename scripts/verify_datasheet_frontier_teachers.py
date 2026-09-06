#!/usr/bin/env python3
"""Ground frontier teachers in source pages before training admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.electronics.claims import (
    canonical_json,
    make_claim,
    seal_claim_bundle,
)
from harness.electronics.corpus import sha256_file
from harness.electronics.frontier_batch import (
    FrontierCandidate,
    FrontierTeacherVerification,
    load_prepared_bundle,
)
from harness.electronics.ground_truth import (
    load_ground_truth_records,
)
from harness.electronics.local_verification import (
    grounded_pin_rows,
    parametric_fact_identity,
    quoted_parametric_evidence,
    verify_opn_decoder,
    verify_parametrics,
    verify_pin_or_ball,
    verify_pin_semantics,
    verify_series_summary,
)
from harness.electronics.local_model import focused_page_context
from harness.electronics.models import (
    ClaimClass,
    EntityGrain,
    EntityReference,
    EvidenceKind,
    EvidenceReference,
    ModelIdentity,
    is_valid_claim_json,
)
from harness.electronics.regions import structural_pin_regions


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _ground_truth_context(
    corpus: dict[str, Any],
    ground_truth_root: Path,
) -> dict[str, dict[str, Any]]:
    records_by_document = load_ground_truth_records(
        corpus,
        ground_truth_root,
    )
    output: dict[str, dict[str, Any]] = {}
    for document in corpus.get("documents") or []:
        records = records_by_document.get(document["document_sha256"], [])
        packages = {
            package
            for record in records
            for package in record["packages"]
        }
        output[document["document_sha256"]] = {
            "records": records,
            "packages": packages,
            "vendor": next(
                (
                    record["vendor"]
                    for record in records
                    if record.get("vendor")
                ),
                next(iter(document.get("vendors") or []), None),
            ),
        }
    return output


def _package(
    item: dict[str, Any],
    page: dict[str, Any],
    context: dict[str, Any],
) -> tuple[str | None, str]:
    structural = item.get("structural_evidence")
    scope = (
        structural.get("package_scope")
        if isinstance(structural, dict)
        else None
    )
    if isinstance(scope, dict) and scope.get("package"):
        return str(scope["package"]), str(scope.get("source") or "structural")
    packages = sorted(context.get("packages") or [])
    if len(packages) == 1:
        return packages[0], "single_owned_ground_truth_package"
    headers = {
        header
        for region in structural_pin_regions(page)
        for header in region["package_headers"]
    }
    if len(headers) == 1:
        return next(iter(headers)), "single_structural_table_package"
    return None, "package_scope_unresolved"


def _entity_id(document_sha256: str, package: str, pin_no: Any) -> str:
    suffix = re.sub(r"[^A-Za-z0-9._+-]+", "-", str(pin_no)).strip("-")
    package_key = re.sub(r"[^A-Za-z0-9._+-]+", "-", package).strip("-")
    return f"pin:{document_sha256[:16]}:{package_key[:80]}:{suffix[:40]}"


def _quoted_source_row(
    page: dict[str, Any],
    capability: str,
    pin: dict[str, Any],
) -> str:
    context = focused_page_context(capability, page)
    number = re.sub(r"[^A-Z0-9]+", "", str(pin["pin_no"]).upper())
    name = re.sub(r"[^A-Z0-9]+", "", str(pin["name"]).upper())
    for table in context.get("tables") or []:
        for row in table.get("rows") or []:
            if not isinstance(row, list):
                continue
            cells = [
                " ".join(_string(cell).split())
                for cell in row
            ]
            normalized = {
                re.sub(r"[^A-Z0-9]+", "", cell.upper())
                for cell in cells
                if cell
            }
            physical_tokens = {
                re.sub(r"[^A-Z0-9]+", "", part.upper())
                for cell in cells
                for part in re.split(r"[,;\s]+", cell)
                if part
            }
            if name in normalized and (
                number in normalized or number in physical_tokens
            ):
                return " | ".join(cells)
    for segment in _source_segments(page, capability):
        normalized = _normalized(segment)
        physical_tokens = {
            _normalized(part)
            for part in re.split(r"[,;|\s]+", segment)
            if part
        }
        if name in normalized and (
            number in normalized or number in physical_tokens
        ):
            return " ".join(segment.split())
    raise ValueError("verified teacher pin has no quotable source row")


def _string(value: Any) -> str:
    return str("" if value is None else value)


def _normalized(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", _string(value).upper())


def _source_segments(page: dict[str, Any], capability: str) -> list[str]:
    context = focused_page_context(capability, page)
    segments = [
        str(block.get("text") or "")
        for block in context.get("blocks") or []
        if isinstance(block, dict)
    ]
    for table in context.get("tables") or []:
        if not isinstance(table, dict):
            continue
        segments.extend(
            " | ".join(" ".join(_string(cell).split()) for cell in row)
            for row in table.get("rows") or []
            if isinstance(row, list)
        )
    digital_text = context.get("digital_text")
    if isinstance(digital_text, dict):
        segments.extend(str(digital_text.get("text") or "").splitlines())
    return [segment for segment in segments if segment.strip()]


def _quoted_source_span(
    page: dict[str, Any],
    capability: str,
    *values: Any,
) -> str:
    wanted = [_normalized(value) for value in values if _normalized(value)]
    for segment in _source_segments(page, capability):
        normalized = _normalized(segment)
        if wanted and all(value in normalized for value in wanted):
            return " ".join(segment.split())
    raise ValueError("verified teacher fact has no quotable source span")


def _document_entity(
    document_sha256: str,
    vendor: str | None,
) -> EntityReference:
    return EntityReference(
        entity_id=f"document:{document_sha256}",
        grain=EntityGrain.DOCUMENT,
        canonical_id=document_sha256,
        vendor=vendor,
    )


def _opn_entity(
    document_sha256: str,
    response: dict[str, Any],
    vendor: str | None,
) -> EntityReference:
    canonical = (
        response.get("base_part")
        or response.get("series")
        or document_sha256
    )
    grain = (
        EntityGrain.BASE_PART
        if response.get("base_part")
        else (
            EntityGrain.SERIES
            if response.get("series")
            else EntityGrain.DOCUMENT
        )
    )
    suffix = re.sub(r"[^A-Za-z0-9._+-]+", "-", str(canonical)).strip("-")
    return EntityReference(
        entity_id=f"{grain.value}:{document_sha256[:16]}:{suffix[:120]}",
        grain=grain,
        canonical_id=str(canonical),
        vendor=vendor,
    )


def _write_verifications(
    path: Path,
    values: list[FrontierTeacherVerification],
) -> dict[str, Any]:
    output = path.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        canonical_json(value.model_dump(mode="json", by_alias=True)) + b"\n"
        for value in values
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument(
        "--work-queue",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--page-evidence", type=Path, required=True)
    parser.add_argument("--pillar-evidence", type=Path)
    parser.add_argument("--corpus-registry", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--verifications-output", type=Path, required=True)
    parser.add_argument("--claims-output", type=Path, required=True)
    args = parser.parse_args()

    bundle = args.bundle.expanduser().resolve(strict=True)
    manifest, requests = load_prepared_bundle(bundle)
    reconciliation_path = args.reconciliation.expanduser().resolve(strict=True)
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    work = {}
    for path in args.work_queue:
        value = json.loads(path.expanduser().resolve(strict=True).read_text())
        work.update({item["work_id"]: item for item in value["work"]})
    candidate_by_id = {
        value.candidate_id: value
        for value in (
            FrontierCandidate.model_validate(item)
            for item in _jsonl(bundle / "candidates.jsonl")
        )
    }
    request_by_candidate = {
        item["_harness"]["candidate_id"]: item for item in requests
    }
    wanted = {
        (
            candidate.document_sha256,
            int(work[candidate.entity_hint]["page_1based"]),
        )
        for candidate in candidate_by_id.values()
        if candidate.entity_hint in work
    }
    page_root = args.page_evidence.expanduser().resolve(strict=True)
    pages = {
        (value["document_sha256"], int(value["page_1based"])): value
        for value in _jsonl(page_root / "page-evidence.jsonl")
        if (value["document_sha256"], int(value["page_1based"])) in wanted
    }
    pillar_pages = {}
    if args.pillar_evidence is not None:
        pillar_path = args.pillar_evidence.expanduser().resolve(strict=True)
        pillar_manifest = json.loads(
            (pillar_path.parent / "manifest.json").read_text(encoding="utf-8")
        )
        pillar_receipt = pillar_manifest.get("artifacts", {}).get(
            pillar_path.name,
            {},
        )
        if (
            pillar_receipt.get("sha256") != sha256_file(pillar_path)
            or pillar_receipt.get("bytes") != pillar_path.stat().st_size
        ):
            raise ValueError("pillar evidence differs from its manifest")
        pillar_pages = {
            value["work_id"]: value["page"]
            for value in _jsonl(pillar_path)
        }
    corpus = json.loads(
        args.corpus_registry.expanduser().resolve(strict=True).read_text()
    )
    context = _ground_truth_context(
        corpus,
        args.ground_truth_root.expanduser().resolve(strict=True),
    )
    now = datetime.now(timezone.utc)
    claims = []
    verifications = []
    counts: Counter[str] = Counter()
    for outcome in reconciliation.get("outcomes") or []:
        if outcome.get("status") != "ready_for_claim_verification":
            counts[f"outcome:{outcome.get('status', 'unknown')}"] += 1
            continue
        candidate = candidate_by_id[outcome["candidate_id"]]
        item = work.get(candidate.entity_hint)
        if item is None:
            raise ValueError(f"teacher candidate has unknown work: {candidate.entity_hint}")
        page = pillar_pages.get(candidate.entity_hint) or pages[
            (candidate.document_sha256, int(item["page_1based"]))
        ]
        document_context = context.get(candidate.document_sha256, {})
        capability = candidate.capability.value
        package = None
        package_source = None
        verified_response = None
        if capability in {"pin_or_ball", "pin_semantics"}:
            package, package_source = _package(
                item,
                page,
                document_context,
            )
            verifier = (
                verify_pin_or_ball
                if capability == "pin_or_ball"
                else verify_pin_semantics
            )
            raw_pins = outcome["response"].get("pins")
            raw_pins = raw_pins if isinstance(raw_pins, list) else []
            counts["pin_rows:seen"] += len(raw_pins)
            grounded_pins = grounded_pin_rows(
                {"result": outcome["response"]},
                page,
                require_semantics=capability == "pin_semantics",
            )
            counts["pin_rows:grounded"] += len(grounded_pins)
            counts["pin_rows:quarantined"] += (
                len(raw_pins) - len(grounded_pins)
            )
            if grounded_pins:
                verified_response = {"pins": grounded_pins}
                verdict = verifier(
                    {"result": verified_response},
                    page,
                )
                passed = verdict.passed and package is not None
                if passed:
                    counts["pin_rows:admitted"] += len(grounded_pins)
                    if len(grounded_pins) == len(raw_pins):
                        counts["pin_pages:fully_grounded"] += 1
                    else:
                        counts["pin_pages:partially_salvaged"] += 1
                else:
                    counts["pin_rows:aggregate_quarantined"] += len(
                        grounded_pins
                    )
                    verified_response = None
            else:
                verdict = verifier(
                    {"result": outcome["response"]},
                    page,
                )
                passed = False
        elif capability == "series_summary":
            # Per-fact salvage, mirroring the parametric lane: one
            # paraphrased characteristic used to quarantine the whole
            # response, discarding applications that ARE printed verbatim
            # (the GD probe lost all 52 application strings this way).
            # Each fact is grounded independently; only printed facts are
            # admitted, and the reassembled response must still pass the
            # aggregate verifier.
            raw_characteristics = [
                str(value).strip()
                for value in outcome["response"].get("characteristics", [])
                if isinstance(value, str) and value.strip()
            ]
            raw_applications = [
                str(value).strip()
                for value in outcome["response"].get("applications", [])
                if isinstance(value, str) and value.strip()
            ]
            counts["summary_facts:seen"] += len(raw_characteristics) + len(
                raw_applications
            )
            grounded_characteristics: list[str] = []
            grounded_applications: list[str] = []
            seen_facts: set[str] = set()
            for value, admitted in (
                *((v, grounded_characteristics) for v in raw_characteristics),
                *((v, grounded_applications) for v in raw_applications),
            ):
                probe = {
                    "summary": value,
                    "characteristics": (
                        [value] if admitted is grounded_characteristics else []
                    ),
                    "applications": (
                        [value] if admitted is grounded_applications else []
                    ),
                }
                fact_verdict = verify_series_summary(
                    {"result": probe},
                    page,
                )
                if fact_verdict.passed and value not in seen_facts:
                    admitted.append(value)
                    seen_facts.add(value)
                else:
                    counts["summary_facts:quarantined"] += 1
            counts["summary_facts:admitted"] += len(
                grounded_characteristics
            ) + len(grounded_applications)
            if grounded_characteristics or grounded_applications:
                verified_response = {
                    "summary": " ".join(
                        grounded_characteristics or grounded_applications
                    ),
                    "characteristics": grounded_characteristics,
                    "applications": grounded_applications,
                }
                verdict = verify_series_summary(
                    {"result": verified_response},
                    page,
                )
                if not verdict.passed:
                    raise ValueError(
                        "individually grounded summary facts failed "
                        "aggregate verification"
                    )
                passed = True
                if len(grounded_characteristics) + len(
                    grounded_applications
                ) == len(raw_characteristics) + len(raw_applications):
                    counts["summary_pages:fully_grounded"] += 1
                else:
                    counts["summary_pages:partially_salvaged"] += 1
            else:
                verdict = verify_series_summary(
                    {"result": outcome["response"]},
                    page,
                )
                passed = False
                verified_response = None
        elif capability == "opn_decoder":
            verdict = verify_opn_decoder(
                {"result": outcome["response"]},
                page,
            )
            passed = verdict.passed
            if passed:
                verified_response = {
                    key: outcome["response"].get(key)
                    for key in ("series", "base_part", "package_code")
                }
                verified_response["suffixes"] = [
                    {
                        "code": item["code"],
                        "meaning": item.get("meaning"),
                    }
                    for item in outcome["response"].get("suffixes", [])
                ]
        elif capability == "parametrics":
            raw_facts = outcome["response"].get("facts")
            grounded_facts = []
            # Dedup key must match the aggregate duplicate gate inside
            # verify_parametrics exactly; hashing raw JSON let normalized
            # twins (case or numeric-type variants of one printed fact)
            # through, and the aggregate re-check then failed the whole page.
            grounded_identities = set()
            if isinstance(raw_facts, list):
                counts["parametric_facts:seen"] += len(raw_facts)
                for fact in raw_facts:
                    # Claim-contract JSON gate first: a fact with, say, an
                    # empty conditions key can pass grounding but would abort
                    # the whole run when minted into a FactClaim.
                    if not is_valid_claim_json(fact):
                        counts["parametric_facts:quarantined"] += 1
                        continue
                    fact_verdict = verify_parametrics(
                        {"result": {"facts": [fact]}},
                        page,
                    )
                    identity = (
                        parametric_fact_identity(fact)
                        if fact_verdict.passed
                        else None
                    )
                    if (
                        fact_verdict.passed
                        and identity not in grounded_identities
                    ):
                        grounded_facts.append(fact)
                        grounded_identities.add(identity)
                    else:
                        counts["parametric_facts:quarantined"] += 1
            counts["parametric_facts:admitted"] += len(grounded_facts)
            if grounded_facts:
                verified_response = {"facts": grounded_facts}
                verdict = verify_parametrics(
                    {"result": verified_response},
                    page,
                )
                if not verdict.passed:
                    raise ValueError(
                        "individually grounded parametric facts failed "
                        "aggregate verification"
                    )
                passed = True
                if len(grounded_facts) == len(raw_facts):
                    counts["parametric_pages:fully_grounded"] += 1
                else:
                    counts["parametric_pages:partially_salvaged"] += 1
            else:
                verdict = verify_parametrics(
                    {"result": outcome["response"]},
                    page,
                )
                passed = False
        else:
            raise ValueError(
                f"unsupported teacher verification lane: {capability}"
            )
        reason = verdict.reason
        if capability == "parametrics" and not passed:
            reason = "teacher contains no independently grounded parametric facts"
        if capability == "series_summary" and not passed:
            reason = (
                "teacher contains no independently grounded summary facts"
            )
        if capability in {"pin_or_ball", "pin_semantics"} and not passed:
            reason = (
                "teacher contains no aggregate-safe set of independently "
                "grounded pin rows"
                if package is not None
                else "package scope cannot be isolated"
            )
        if (
            capability in {"pin_or_ball", "pin_semantics"}
            and verdict.passed
            and package is None
        ):
            reason = "package scope cannot be isolated"

        claim_ids = []
        if passed:
            request = request_by_candidate[candidate.candidate_id]
            teacher = ModelIdentity(
                provider="anthropic",
                model=manifest["model"],
                request_sha256=hashlib.sha256(
                    canonical_json(request["params"])
                ).hexdigest(),
                response_id=outcome.get("message_id"),
                batch_id=outcome["batch_id"],
            )
            evidence_item = candidate.evidence[0]
            page_size = page["page_size"]
            bbox = evidence_item.bbox or (
                0.0,
                0.0,
                float(page_size["width"]),
                float(page_size["height"]),
            )
            vendor = item.get("vendor") or document_context.get("vendor")

            def evidence_for(quoted_text: str) -> EvidenceReference:
                return EvidenceReference(
                    kind=EvidenceKind.IMAGE_REGION,
                    document_sha256=candidate.document_sha256,
                    source_uri=(
                        evidence_item.path.expanduser().resolve().as_uri()
                    ),
                    artifact_sha256=evidence_item.sha256,
                    page_1based=item["page_1based"],
                    bbox=bbox,
                    quoted_text=quoted_text,
                )

            if capability in {"pin_or_ball", "pin_semantics"}:
                assert package is not None
                assert verified_response is not None
                for pin in verified_response["pins"]:
                    evidence = evidence_for(
                        _quoted_source_row(page, capability, pin)
                    )
                    entity = EntityReference(
                        entity_id=_entity_id(
                            candidate.document_sha256,
                            package,
                            pin["pin_no"],
                        ),
                        grain=EntityGrain.PIN_OR_BALL,
                        canonical_id=str(pin["pin_no"]),
                        vendor=vendor,
                        package=package,
                    )
                    claim = make_claim(
                        entity=entity,
                        field=(
                            "pin.identity_record"
                            if capability == "pin_or_ball"
                            else "pin.semantic_record"
                        ),
                        value=pin,
                        claim_class=(
                            ClaimClass.VISIBLE_FACT
                            if capability == "pin_or_ball"
                            else ClaimClass.SEMANTIC_LABEL
                        ),
                        extraction_method="frontier_teacher_source_grounded",
                        evidence=(evidence,),
                        conditions={
                            "package": package,
                            "package_scope_source": package_source,
                        },
                        model=teacher,
                        created_at=now,
                    )
                    claims.append(claim)
                    claim_ids.append(claim.claim_id)
            elif capability == "parametrics":
                entity = _document_entity(
                    candidate.document_sha256,
                    vendor,
                )
                assert verified_response is not None
                for fact in verified_response["facts"]:
                    quoted_text = quoted_parametric_evidence(fact, page)
                    if quoted_text is None:
                        raise ValueError(
                            "verified parametric fact has no table-cell evidence"
                        )
                    claim = make_claim(
                        entity=entity,
                        field="parametric.fact",
                        value=fact,
                        claim_class=ClaimClass.VISIBLE_FACT,
                        extraction_method="frontier_teacher_source_grounded",
                        evidence=(
                            evidence_for(quoted_text),
                        ),
                        conditions={
                            "value_role": fact.get("value_role"),
                            "source_conditions": fact.get("conditions") or {},
                        },
                        model=teacher,
                        created_at=now,
                    )
                    claims.append(claim)
                    claim_ids.append(claim.claim_id)
            elif capability == "series_summary":
                entity = _document_entity(
                    candidate.document_sha256,
                    vendor,
                )
                assert verified_response is not None
                for key, field in (
                    ("characteristics", "summary.characteristic"),
                    ("applications", "summary.application"),
                ):
                    for value in verified_response[key]:
                        claim = make_claim(
                            entity=entity,
                            field=field,
                            value=value,
                            claim_class=ClaimClass.VISIBLE_FACT,
                            extraction_method=(
                                "frontier_teacher_source_grounded"
                            ),
                            evidence=(
                                evidence_for(
                                    _quoted_source_span(
                                        page,
                                        capability,
                                        value,
                                    )
                                ),
                            ),
                            conditions={"source_page_role": key},
                            model=teacher,
                            created_at=now,
                        )
                        claims.append(claim)
                        claim_ids.append(claim.claim_id)
            elif capability == "opn_decoder":
                assert verified_response is not None
                response = verified_response
                entity = _opn_entity(
                    candidate.document_sha256,
                    response,
                    vendor,
                )
                for key, field in (
                    ("series", "product.series"),
                    ("base_part", "product.base_part"),
                    ("package_code", "opn.package_code"),
                ):
                    value = response.get(key)
                    if value is None:
                        continue
                    claim = make_claim(
                        entity=entity,
                        field=field,
                        value=value,
                        claim_class=ClaimClass.VISIBLE_FACT,
                        extraction_method="frontier_teacher_source_grounded",
                        evidence=(
                            evidence_for(
                                _quoted_source_span(
                                    page,
                                    capability,
                                    value,
                                )
                            ),
                        ),
                        conditions={"decoder_field": key},
                        model=teacher,
                        created_at=now,
                    )
                    claims.append(claim)
                    claim_ids.append(claim.claim_id)
                for suffix in response["suffixes"]:
                    quote_values = [suffix["code"]]
                    if suffix.get("meaning"):
                        quote_values.append(suffix["meaning"])
                    claim = make_claim(
                        entity=entity,
                        field="opn.suffix",
                        value=suffix,
                        claim_class=ClaimClass.VISIBLE_FACT,
                        extraction_method="frontier_teacher_source_grounded",
                        evidence=(
                            evidence_for(
                                _quoted_source_span(
                                    page,
                                    capability,
                                    *quote_values,
                                )
                            ),
                        ),
                        conditions={"decoder_field": "suffixes"},
                        model=teacher,
                        created_at=now,
                    )
                    claims.append(claim)
                    claim_ids.append(claim.claim_id)
        verification_core = {
            "candidate_id": candidate.candidate_id,
            "response_sha256": outcome["response_sha256"],
            "status": "passed" if passed else "quarantined",
            "checks": [
                *verdict.checks,
                *(
                    ["package_scope_isolated"]
                    if capability in {"pin_or_ball", "pin_semantics"}
                    else []
                ),
                *(
                    ["independent_parametric_fact_grounding"]
                    if capability == "parametrics"
                    else []
                ),
            ],
            "claim_ids": claim_ids,
            "evidence_sha256": [
                evidence.sha256 for evidence in candidate.evidence
            ],
            "verified_response": verified_response,
            "verified_response_sha256": (
                hashlib.sha256(canonical_json(verified_response)).hexdigest()
                if verified_response is not None
                else None
            ),
            "reason": None if passed else reason,
        }
        verification_id = (
            "teacher-verify-"
            + hashlib.sha256(canonical_json(verification_core)).hexdigest()[:32]
        )
        verifications.append(
            FrontierTeacherVerification(
                verification_id=verification_id,
                candidate_id=candidate.candidate_id,
                response_sha256=outcome["response_sha256"],
                status="passed" if passed else "quarantined",
                verifier="source_evidence_rule",
                checks=tuple(verification_core["checks"]),
                # A datasheet can print the same summary line twice (for
                # example one bullet repeated under USART and UART); both
                # mint the identical content-addressed claim, and the
                # verification references it once.
                claim_ids=tuple(dict.fromkeys(claim_ids)),
                evidence_sha256=tuple(verification_core["evidence_sha256"]),
                verified_response=verified_response,
                verified_response_sha256=verification_core[
                    "verified_response_sha256"
                ],
                reason=None if passed else reason,
            )
        )
        counts["passed" if passed else "quarantined"] += 1

    verification_receipt = _write_verifications(
        args.verifications_output,
        verifications,
    )
    # Claim IDs are content-derived, so two overlapping work items that
    # ground the identical fact with identical evidence mint the same
    # claim. The bundle stores each claim once; verification records keep
    # referencing the shared claim_id.
    unique_claims = list(
        {claim.claim_id: claim for claim in claims}.values()
    )
    counts["claims:duplicate_identity_merged"] = len(claims) - len(
        unique_claims
    )
    claims_manifest = seal_claim_bundle(
        args.claims_output,
        unique_claims,
        source_receipts={
            "prepared_evidence_sha256": manifest["evidence_sha256"],
            "reconciliation_evidence_sha256": reconciliation[
                "evidence_sha256"
            ],
            "verifications_sha256": verification_receipt["sha256"],
        },
        created_at=now,
    )
    summary = {
        "counts": {
            **dict(sorted(counts.items())),
            "claims": len(unique_claims),
            "verifications": len(verifications),
        },
        "verifications": verification_receipt,
        "claims_evidence_sha256": claims_manifest["evidence_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
