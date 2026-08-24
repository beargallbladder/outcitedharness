export async function findArtifactWeekAsOf(
  kind: string,
  targetWeek: string
): Promise<string | null> {
  const r = await db.query<{ week: string }>(
    `SELECT week FROM cr_artifacts
       WHERE artifact_kind = $1 AND week <= $2
       ORDER BY week DESC LIMIT 1`,
    [kind, targetWeek]
  )
  return r.rows[0]?.week ?? null
}
