#!/usr/bin/env python3
"""Build the live-file decision pack from the sandbox.

Fleet frame this pack scores:
  1 box  — existing 27B stays (volume / SQL / FE / review). Also the
           embeddings+training spare.
  3 box  — only a model that cannot fit on one Spark. GLM-5.2 and
           MiniMax M3 are the 3-node recipes. Hy3/Flash are 2-node
           controls: they must beat GLM by enough to justify wasting
           a node, or they lose the slot.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

DEST = Path("cases/eval_live")
SANDBOX = Path("/Volumes/M5_4TB/repos/outcited-ai-sandbox-20260822")


def dump_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def write_case(
    case_id: str,
    title: str,
    category: str,
    prompt: str,
    files: dict[str, str],
    evaluation: dict,
    notes: str,
    expected: dict | None = None,
    validator: str | None = None,
) -> None:
    root = DEST / case_id
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "expected").mkdir(parents=True, exist_ok=True)
    (root / "prompt.md").write_text(prompt.rstrip() + "\n")
    for name, text in files.items():
        (root / "inputs" / name).write_text(text.rstrip() + "\n")
    (root / "notes.md").write_text(notes.rstrip() + "\n")
    spec = {
        "id": case_id,
        "title": title,
        "category": category,
        "tags": ["live_sandbox", "three_node_slot"],
        "input_files": [f"inputs/{name}" for name in files],
        "evaluation": evaluation,
        "historical": {"notes": notes[:400]},
    }
    if expected is not None:
        (root / "expected" / "answer.json").write_text(
            json.dumps(expected, indent=2) + "\n"
        )
        spec["reference_answer"] = {"file": "expected/answer.json"}
    if validator:
        (root / "expected" / "validate.py").write_text(validator)
    dump_yaml(root / "case.yaml", spec)


HOUSE = """# House rules that bind this review (from AGENTS.md + constitution)

6. As-of carryforward, always. Priors age; they never zero on week rollover.
   `week <= target` then latest. A target BEFORE the earliest artifact
   returns null, not the earliest week.

8. Coverage sentinels are not data. category = '__unknown__' (legacy 'n')
   is a coverage row, never a mention. Every consumer of
   category_mentions_v2.category must filter them explicitly.

10. sha-check, never presence-check, on artifact ingest. kind+week already
    in cr_artifacts is not a skip. Compare on-disk sha256 vs the DB row.

Tri-state is constitutional. A summary/interface list PROVES PRESENCE of
what it names and NEVER proves absence. Explicit zero counts prove
absence. Silence must stay UNKNOWN (key absent), not False.
"""

def lines(rel: str, start: int, end: int) -> str:
    rows = (SANDBOX / rel).read_text().splitlines()
    return "\n".join(rows[start - 1 : end])


AUTH = lines("lib/api/v1/auth.ts", 156, 171)
CALLER = """// lib/api/v1/with-auth.ts
if (slug !== null && !slugInWallet(user, slug)) {
  return 403
}

// lib/designwins/facade/auth.ts
if (aisle && !slugInWallet(user, aisle)) {
  return designWinsProblem(403, 'forbidden', 'Aisle is not in this wallet', {...})
}
"""

SENTINEL_OK = """// lib/insights/recall-gap-loader.ts (does filter)
AND cm.kim_category_id IS NOT NULL
AND cm.category NOT IN ('__unknown__','n')
"""

SENTINEL_NAKED = """-- hypothetical new consumer copied from an analyst notebook
SELECT COUNT(*) AS n_mentions, COUNT(DISTINCT model) AS n_models
FROM category_mentions_v2
WHERE brand_id = (SELECT id FROM brands WHERE domain = 'examplebrand.com');
"""

ASOF = lines("lib/db/cr-artifacts.ts", 97, 108)

INGEST = """// scripts/audit/ingest-core-ai-index-and-below-floor-artifacts.ts
const fullCheck = await db.query(
  `SELECT COUNT(*)::text AS n FROM cr_artifacts WHERE artifact_kind = $1 AND week = $2`,
  [fullKind, WEEK],
)
if (Number(fullCheck.rows[0].n) === 0) {
  // upload
} else {
  console.log('already in cr_artifacts — not re-ingesting')
}
"""

SERIES = """// lib/designwins/facade/catalogue-series-count.ts
// Isolated so family/board routes do not pull the 109 MB corpus
// into their serverless function graph.
import { readFileSync } from 'fs'
const path = verifyReleaseArtifact(loaded, recordsPath(loaded, aisle), pin.records_sha256)
records = readFileSync(path, 'utf8')  // ENOENT in prod if only the .gz twin shipped
"""

BLOCK_DIFF = """@@ -12,9 +12,13 @@ def bake_row(r):
     comm = (r.get('Communication interface') or '').lower()
     can_count = parse_count(r.get('CAN (#)'))
     if can_count is not None:
         row['has_can'] = can_count > 0
         row['n_can'] = can_count
-        if 'can fd' in comm or 'can-fd' in (r.get('CAN (#)') or '').lower():
-            row['has_can_fd'] = True
-        elif can_count == 0:
-            row['has_can_fd'] = False
+        # simplify: the comm string is the vendor's own summary — if FD
+        # were present they would list it. Absence in the summary is a no.
+        row['has_can_fd'] = ('can fd' in comm or 'can-fd' in comm)
     elif 'can' in comm:
         row['has_can'] = True
"""

APPROVE_DIFF = """@@ -44,6 +44,17 @@
+_JUNK_MPN_PATTERNS = (
+    re.compile(r'copyright|all rights reserved|\\u00a9'),
+    re.compile(r'^\\d[\\d,]* results? found$'),
+    re.compile(r'^(products? compare|show more|load more)$', re.I),
+)
+
+def is_junk_mpn(s: str) -> bool:
+    t = (s or '').strip().lower()
+    if not t or len(t) < 4 or len(t) > 40:
+        return True
+    return any(p.search(t) for p in _JUNK_MPN_PATTERNS)
+
@@ -61,7 +72,7 @@ def read_rows(path):
     for r in rows[1:]:
         pn = cell(r, 'Part Number')
-        if not pn:
+        if not pn or is_junk_mpn(pn):
             continue
"""

HC27 = r'''#!/usr/bin/env python3
import os, re, sys
from pathlib import Path
text = Path(os.environ["HARNESS_ANSWER_PATH"]).read_text()
upper = text.upper()
low = text.lower()
failures = []
verdicts = re.findall(r"\b(APPROVE|BLOCK)\b", upper)
if not verdicts:
    failures.append("missing APPROVE or BLOCK")
elif verdicts[-1] != "BLOCK":
    failures.append(f"final verdict {verdicts[-1]}, must BLOCK")
if not any(k in low for k in ["silence", "omission", "omits", "absent", "absence", "never proves", "does not prove", "unknown"]):
    failures.append("did not treat summary silence as non-proof")
if not any(k in low for k in ["false", "explicit denial", "has_can_fd=false", "has_can_fd = false", "writes false"]):
    failures.append("did not name the False-from-silence write")
if failures:
    print("; ".join(failures))
    raise SystemExit(1)
print("ok")
'''

HC28 = r'''#!/usr/bin/env python3
import os, re, sys
from pathlib import Path
text = Path(os.environ["HARNESS_ANSWER_PATH"]).read_text()
verdicts = re.findall(r"\b(APPROVE|BLOCK)\b", text.upper())
if not verdicts:
    print("missing APPROVE or BLOCK")
    raise SystemExit(1)
if verdicts[-1] != "APPROVE":
    print(f"final verdict {verdicts[-1]}, must APPROVE")
    raise SystemExit(1)
print("ok")
'''


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)

    write_case(
        "lv01_authz_wallet",
        "Live auth.ts wallet hole",
        "middle_tier_review",
        """These are LIVE files from our sandbox. The 27B already sits on one Spark
and stays there. You are auditioning for the 3-Spark monster slot.

Review lib/api/v1/auth.ts slugInWallet and its callers.
Q1: api_tier='custom', wallet_slugs=['microcontrollers'] — which aisles can it read?
Q2: why is that dangerous in THIS paid-API business (ops sets wallets, unpublished aisles exist)?
Q3: write the minimal patch that keeps a genuine all-access path for internal keys.

Score is conceptual. Name the hole, the ops/money trap, and an explicit
all-access mechanism (internal flag/tier/wildcard) — not the exact memo words.""",
        {
            "HOUSE_RULES.md": HOUSE,
            "auth.ts": AUTH,
            "callers.ts": CALLER,
        },
        {
            "type": "keyword_rubric",
            "groups": [
                ["every aisle", "all aisles", "everything", "unrestricted", "return true", "bypasses the wallet", "wallet_slugs is never"],
                ["wallet is ignored", "wallet_slugs is decorative", "dead data", "ops", "unpublished", "paid for", "paying for", "never consulted", "revenue"],
                ["is_internal", "internal", "explicit all-access", "wildcard", "all_access", "separate flag", "wallet_slugs for custom", "custom now"],
            ],
        },
        "Live slugInWallet still short-circuits custom to true.",
    )

    write_case(
        "lv02_sentinel_sql",
        "Live sentinel filter",
        "sql_data",
        """LIVE sandbox. House rule 8 is attached.

One consumer already filters sentinels. A new query does not.
Find the flaw in the naked query, write the corrected query, and state
what changes if those rows get aggregated or embedded.""",
        {
            "HOUSE_RULES.md": HOUSE,
            "good_consumer.sql": SENTINEL_OK,
            "naked_query.sql": SENTINEL_NAKED,
        },
        {
            "type": "keyword_rubric",
            "groups": [
                ["__unknown__", "sentinel", "'n'"],
                ["not in", "exclude", "filter", "category !=", "<>"],
                ["inflate", "overcount", "phantom", "not mentions", "not data", "coverage row"],
            ],
        },
        "Real rule #8. The good consumer already does this; the naked one does not.",
    )

    write_case(
        "lv03_asof",
        "Live findArtifactWeekAsOf",
        "sql_data",
        """LIVE function from lib/db/cr-artifacts.ts. House rule 6.

Rows: hero_rollup 2026-W24, 2026-W28, 2026-W31; recall_pack 2026-W30.
Return EXACT JSON: {"a": ..., "b": ..., "c": ..., "d": ...}
a = findArtifactWeekAsOf('hero_rollup', '2026-W30')
b = findArtifactWeekAsOf('hero_rollup', '2026-W31')
c = findArtifactWeekAsOf('hero_rollup', '2026-W23')
d = findArtifactWeekAsOf('recall_pack', '2026-W52')
Use week strings or null. JSON only.""",
        {"HOUSE_RULES.md": HOUSE, "cr-artifacts.ts": ASOF},
        {"type": "exact_json", "ignore_order": False},
        "Verbatim production as-of. Trap is (c): before earliest => null.",
        expected={"a": "2026-W28", "b": "2026-W31", "c": None, "d": "2026-W30"},
    )

    write_case(
        "lv04_artifact_integrity",
        "Live sha-vs-presence + gz twin",
        "middle_tier_bug",
        """Two LIVE defects, same house rule 10.

1) The ingest skips when kind+week is already in cr_artifacts.
2) catalogue-series-count.ts readFileSyncs the path verifyReleaseArtifact
   returned. records.jsonl is 109 MB; serverless bundles only the .gz twin.
   Local disk has both files.

Diagnose both. Name the correct checks (sha, not presence; prefer raw then
gz). State why prod 503s and local works.""",
        {
            "HOUSE_RULES.md": HOUSE,
            "ingest.ts": INGEST,
            "catalogue-series-count.ts": SERIES,
        },
        {
            "type": "keyword_rubric",
            "groups": [
                ["sha", "sha256", "presence", "already in cr_artifacts", "kind+week"],
                ["gz", "gunzip", "109", "bundle", "not in the bundle", "excluded"],
                ["local", "dev", "both files", "on disk", "prod", "enoent", "503"],
            ],
        },
        "Combined hc15+hc25 against the real files.",
    )

    write_case(
        "lv05_diff_block",
        "Live must-BLOCK silence-to-false",
        "diff_review",
        """Review this PR against the attached tri-state constitution.
Verdict word required: APPROVE or BLOCK, then reasons.""",
        {"HOUSE_RULES.md": HOUSE, "diff.patch": BLOCK_DIFF},
        {"type": "command", "command": ["python3", "expected/validate.py"]},
        "Silence in a summary must not become has_can_fd=False.",
        validator=HC27,
    )

    write_case(
        "lv06_diff_approve",
        "Live must-APPROVE junk MPN",
        "diff_review",
        """Review this PR. A vendor footer once shipped as a part number.
Verdict word required: APPROVE or BLOCK. Blocking a correct change fails.""",
        {"HOUSE_RULES.md": HOUSE, "diff.patch": APPROVE_DIFF},
        {"type": "command", "command": ["python3", "expected/validate.py"]},
        "Approve-with-notes. Discrimination case.",
        validator=HC28,
    )
    print(f"wrote {len(list(DEST.glob('*/case.yaml')))} cases into {DEST}")


if __name__ == "__main__":
    main()
