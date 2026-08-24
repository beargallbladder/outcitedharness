# House rules that bind this review (from AGENTS.md + constitution)

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
