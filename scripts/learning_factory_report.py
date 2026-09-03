#!/usr/bin/env python3
"""Write paid-replacement and learning-factory operating metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from harness.economics import write_learning_factory_report
from harness.storage.db import Store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = write_learning_factory_report(Store(args.database), args.output)
    print(args.output)
    return 0 if report["schema"] == "harness.learning-factory-metrics.v1" else 1


if __name__ == "__main__":
    raise SystemExit(main())
