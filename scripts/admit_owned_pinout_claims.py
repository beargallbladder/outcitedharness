#!/usr/bin/env python3
"""Verify owned pinout claims and seal training/eval/CR admissions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from harness.electronics.admission import build_admission_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-bundle", type=Path, required=True)
    parser.add_argument("--training-authorization", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_admission_bundle(
        args.claim_bundle,
        args.output_directory,
        authorization_path=args.training_authorization,
        created_at=datetime.now(timezone.utc),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
