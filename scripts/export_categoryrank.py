#!/usr/bin/env python3
"""Export a sentinel-filtered CategoryRank snapshot from Render PostgreSQL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RENDER_API = "https://api.render.com/v1"
TABLE = "category_mentions_v2"
SENTINELS = ("__unknown__", "n", "-unknown-", "unknown")
WEEK_PATTERN = re.compile(r"^\d{4}-W(?:0[1-9]|[1-4]\d|5[0-3])$")


def _render_json(path: str) -> Any:
    token = os.environ.get("RENDER_API_KEY")
    if not token:
        raise ValueError("RENDER_API_KEY is required")
    request = urllib.request.Request(
        f"{RENDER_API}{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _connection_environment(postgres_id: str) -> dict[str, str]:
    info = _render_json(f"/postgres/{postgres_id}/connection-info")
    url = urllib.parse.urlsplit(str(info["externalConnectionString"]))
    if url.scheme not in {"postgres", "postgresql"} or not url.hostname:
        raise ValueError("Render returned an invalid PostgreSQL connection string")
    environment = os.environ.copy()
    environment.update(
        {
            "PGHOST": url.hostname,
            "PGPORT": str(url.port or 5432),
            "PGUSER": urllib.parse.unquote(url.username or ""),
            "PGPASSWORD": urllib.parse.unquote(url.password or ""),
            "PGDATABASE": urllib.parse.unquote(url.path.lstrip("/")),
            "PGSSLMODE": "require",
        }
    )
    return environment


def _psql(environment: dict[str, str], query: str) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("psql")
    if not executable:
        raise ValueError("psql is required")
    return subprocess.run(
        [executable, "-X", "-q", "-t", "-A", "-v", "ON_ERROR_STOP=1", "-c", query],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=1800,
    )


def _psql_to_path(
    environment: dict[str, str],
    query: str,
    path: Path,
) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("psql")
    if not executable:
        raise ValueError("psql is required")
    with path.open("w", encoding="utf-8") as output:
        return subprocess.run(
            [
                executable,
                "-X",
                "-q",
                "-t",
                "-A",
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                query,
            ],
            stdout=output,
            stderr=subprocess.PIPE,
            check=False,
            env=environment,
            text=True,
            timeout=1800,
        )


def discover() -> list[dict[str, str]]:
    rows = _render_json("/postgres?limit=100")
    found: list[dict[str, str]] = []
    for envelope in rows:
        postgres = envelope.get("postgres", envelope)
        if postgres.get("status") != "available":
            continue
        postgres_id = str(postgres["id"])
        result = _psql(
            _connection_environment(postgres_id),
            f"SELECT to_regclass('public.{TABLE}') IS NOT NULL;",
        )
        found.append(
            {
                "id": postgres_id,
                "name": str(postgres["name"]),
                "has_category_mentions_v2": (
                    "yes" if result.returncode == 0 and result.stdout.strip() == "t" else "no"
                ),
            }
        )
    return found


def _query(layer: str, through_week: str) -> str:
    if not WEEK_PATTERN.fullmatch(through_week):
        raise ValueError("week must use ISO YYYY-Www format")
    cohort = ""
    if layer == "electronics":
        cohort = """
        WHERE b.primary_vertical = 'electronics'
          AND b.verification_tier IS DISTINCT FROM 'flagged_noise'
        """
    return f"""
WITH cohort AS (
  SELECT
    b.id AS brand_id,
    COALESCE(successor.to_domain, b.domain) AS canonical_domain
  FROM brands b
  LEFT JOIN LATERAL (
    SELECT sm.to_domain
    FROM cr_brand_canonical_successor_map sm
    WHERE sm.from_domain = b.domain
      AND sm.grounding_status LIKE 'verified_%'
      AND sm.to_domain IS NOT NULL
    ORDER BY
      CASE sm.grounding_status
        WHEN 'verified_multi_source' THEN 0
        ELSE 1
      END,
      sm.updated_at DESC,
      sm.edge_id DESC
    LIMIT 1
  ) successor ON TRUE
  {cohort}
),
base AS (
  SELECT
    cm.brand_id,
    cm.kim_category_id AS kim_slug,
    cm.model AS model_id,
    cm.time_window,
    COUNT(*)::bigint AS n_mentions,
    SUM(cm.strength)::double precision AS strength_sum,
    COUNT(cm.strength)::bigint AS strength_count,
    SUM(cm.rank)::double precision AS rank_sum,
    COUNT(cm.rank)::bigint AS rank_count
  FROM {TABLE} cm
  JOIN cohort ON cohort.brand_id = cm.brand_id
  WHERE lower(trim(cm.category)) NOT IN {SENTINELS}
    AND cm.kim_category_id IS NOT NULL
    AND cm.time_window ~ '^\\d{{4}}-W\\d{{2}}$'
    AND to_char(
      to_date(cm.time_window || '-1', 'IYYY-"W"IW-ID'),
      'IYYY-"W"IW'
    ) = cm.time_window
    AND cm.time_window <= '{through_week}'
  GROUP BY
    cm.brand_id, cm.kim_category_id, cm.model, cm.time_window
),
stitched AS (
  SELECT
    cohort.canonical_domain AS brand_domain,
    base.kim_slug,
    base.model_id,
    base.time_window,
    SUM(base.n_mentions)::bigint AS n_mentions,
    SUM(base.strength_sum) / NULLIF(SUM(base.strength_count), 0) AS avg_strength,
    SUM(base.rank_sum) / NULLIF(SUM(base.rank_count), 0) AS avg_rank
  FROM base
  JOIN cohort ON cohort.brand_id = base.brand_id
  GROUP BY
    cohort.canonical_domain, base.kim_slug, base.model_id, base.time_window
)
  SELECT jsonb_build_object(
    'brand_domain', brand_domain,
    'kim_slug', kim_slug,
    'model_id', model_id,
    'time_window', time_window,
    'n_mentions', n_mentions,
    'avg_strength', avg_strength,
    'avg_rank', avg_rank
  )::text
  FROM stitched
  ORDER BY brand_domain, kim_slug, model_id, time_window;
"""


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def export(postgres_id: str, destination: Path, *, week: str, layer: str) -> None:
    destination = destination.resolve(strict=False)
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing export: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_descriptor, raw_name = tempfile.mkstemp(
        prefix=f".{destination.name}.raw.",
        dir=destination.parent,
    )
    os.close(raw_descriptor)
    raw = Path(raw_name)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    count = 0
    try:
        result = _psql_to_path(
            _connection_environment(postgres_id),
            _query(layer, week),
            raw,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"CategoryRank query failed: {result.stderr.strip()[-1000:]}"
            )
        with (
            raw.open(encoding="utf-8") as raw_handle,
            os.fdopen(descriptor, "w", encoding="utf-8") as handle,
        ):
            for line in raw_handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        raw.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    manifest = {
        "schema": "harness.categoryrank.snapshot.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_week": week,
        "source": {
            "provider": "render-postgres",
            "postgres_id": postgres_id,
            "table": TABLE,
            "layer": layer,
        },
        "filters": {
            "sentinels_excluded": list(SENTINELS),
            "requires_kim_category_id": True,
            "requires_valid_iso_week": True,
            "through_week": week,
            "identity_policy": "verified-successor-one-hop",
            "mutable_facts": True,
            "data_use": "retrieval_only",
        },
        "artifact": {
            "path": destination.name,
            "rows": count,
            "bytes": destination.stat().st_size,
            "sha256": _digest(destination),
        },
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(destination, 0o600)
    os.chmod(manifest_path, 0o600)
    print(
        json.dumps(
            {
                "path": str(destination),
                "rows": count,
                "sha256": manifest["artifact"]["sha256"],
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--postgres-id")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--week", default="2026-W34")
    parser.add_argument("--layer", choices=("all", "electronics"), default="electronics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.discover:
            print(json.dumps(discover(), indent=2, sort_keys=True))
            return 0
        if not args.postgres_id or args.destination is None:
            raise ValueError("--postgres-id and --destination are required")
        export(
            args.postgres_id,
            args.destination,
            week=args.week,
            layer=args.layer,
        )
        return 0
    except Exception as error:
        print(f"CategoryRank export failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
