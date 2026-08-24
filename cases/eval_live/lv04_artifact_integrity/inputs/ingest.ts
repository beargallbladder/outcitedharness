// scripts/audit/ingest-core-ai-index-and-below-floor-artifacts.ts
const fullCheck = await db.query(
  `SELECT COUNT(*)::text AS n FROM cr_artifacts WHERE artifact_kind = $1 AND week = $2`,
  [fullKind, WEEK],
)
if (Number(fullCheck.rows[0].n) === 0) {
  // upload
} else {
  console.log('already in cr_artifacts — not re-ingesting')
}
