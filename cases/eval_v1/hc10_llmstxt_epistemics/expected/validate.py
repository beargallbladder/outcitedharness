#!/usr/bin/env python3
"""Correctness = named the right verdict + the two concepts.

A later English use of 'confirmed' is format noise, not a wrong verdict.
"""
import os
import re
import sys
from pathlib import Path

text = Path(os.environ["HARNESS_ANSWER_PATH"]).read_text()
upper = text.upper()
low = text.lower()
failures = []
if "CONSISTENT_BUT_UNPROVEN" not in upper:
    failures.append("missing verdict CONSISTENT_BUT_UNPROVEN")
if not any(
    k in low
    for k in [
        "omission",
        "omit",
        "silence",
        "silent",
        "curated",
        "proves nothing",
        "does not prove",
        "fail to prove",
        "editorial",
    ]
):
    failures.append("did not treat curated-index silence as non-proof")
if not any(
    k in low
    for k in [
        "orderable",
        "parametric",
        "feed",
        "product-status",
        "product status",
        "known_fact",
        "known fact",
    ]
):
    failures.append("did not name settling evidence")
if failures:
    print("; ".join(failures))
    raise SystemExit(1)

# Format-only note for logs. Do not fail the case on it.
verdicts = re.findall(r"\b(CONFIRMED|CONSISTENT_BUT_UNPROVEN|REFUTED)\b", upper)
if verdicts and verdicts[-1] != "CONSISTENT_BUT_UNPROVEN":
    print("ok (format_chatty: last verdict-shaped word was %s)" % verdicts[-1])
else:
    print("ok")
