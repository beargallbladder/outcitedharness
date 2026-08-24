/*
 * Ingest the Core AI Index + Below-Floor Mentions artifacts from
 * /Volumes/shared/exports/cr/ into cr_artifacts.
 *
 * Tapes ship 2026-06-08T05:28Z (in_reply_to
 * core-ai-index-plus-below-floor-mentions-001):
 *
 *   1. cr_per_kim_slug_competing_set_core_ai_index_aliasmap_v3_layer2
 *      sha16 19ee00ca8aa522ed
 *      226 slugs, 933 brands, median_density=29
 *
 *   2. cr_per_brand_below_floor_mentions_aliasmap_v3_layer2
 *      sha16 70efc8f55691875e
 *      1,937 brands with below-floor mentions, 88,653 brand-slug
 *      appearances
 *
 * Sam-lock 2026-06-08: Core AI Index becomes the public default for
 * /categories and /categories/[slug]. Full Substrate (existing
 * aliasmap_v3 layer2 artifact) stays available for paid/admin via
 * the `panel: 'full'` loader parameter. Below-floor mentions surface
 * on the L4 paid brand-owner page only (never public).
 */
import 'dotenv/config'
import { resolve } from 'path'
import dotenv from 'dotenv'
dotenv.config({ path: resolve(process.cwd(), '.env.local'), override: true })
import * as fs from 'fs'
import { uploadArtifact } from '../../lib/db/cr-artifacts'
import { db } from '../../lib/db'

const WEEK = '2026-W23'
const EXPORTS_DIR = '/Volumes/shared/exports/cr'

const PACK = [
  {
    kind: 'cr_per_kim_slug_competing_set_core_ai_index_aliasmap_v3_layer2',
    file: `${EXPORTS_DIR}/cr_per_kim_slug_competing_set_core_ai_index_aliasmap_v3_layer2_2026-W23.json`,
  },
  {
    kind: 'cr_per_brand_below_floor_mentions_aliasmap_v3_layer2',
    file: `${EXPORTS_DIR}/cr_per_brand_below_floor_mentions_aliasmap_v3_layer2_2026-W23.json`,
  },
]

async function main(): Promise<void> {
  console.log(`Ingesting ${PACK.length} Core AI Index + Below-Floor artifacts → cr_artifacts (week=${WEEK})`)
  console.log('')
  for (const spec of PACK) {
    if (!fs.existsSync(spec.file)) {
      console.error(`  ✗ missing: ${spec.file}`)
      continue
    }
    const blob = fs.readFileSync(spec.file)
    const result = await uploadArtifact({
      kind: spec.kind,
      week: WEEK,
      blob,
      generated_at: new Date(),
      source_path: spec.file,
    })
    console.log(`  ✓ ${spec.kind.padEnd(70)} size=${(blob.length / 1024).toFixed(0)}KB sha16=${result.sha256.slice(0, 16)}`)
  }

  console.log('')
  console.log('Verifying in DB:')
  const verify = await db.query<{ artifact_kind: string; week: string; size_bytes: string }>(
    `SELECT artifact_kind, week, size_bytes::text
       FROM cr_artifacts
      WHERE week = $1 AND artifact_kind = ANY($2::text[])
      ORDER BY artifact_kind`,
    [WEEK, PACK.map(p => p.kind)],
  )
  for (const r of verify.rows) {
    console.log(`  ${r.artifact_kind.padEnd(70)} week=${r.week}  size=${(Number(r.size_bytes) / 1024).toFixed(0)}KB`)
  }

  // Also ingest the Layer2 v3 artifact (Full Substrate) if it's not
  // already in cr_artifacts — the loader will need this as the `full`
  // panel.
  const fullKind = 'cr_per_kim_slug_competing_set_aliasmap_v3_layer2_electronics'
  const fullFile = `${EXPORTS_DIR}/cr_per_kim_slug_competing_set_aliasmap_v3_layer2_electronics_2026-W23.json`
  const fullCheck = await db.query<{ n: string }>(
    `SELECT COUNT(*)::text AS n FROM cr_artifacts WHERE artifact_kind = $1 AND week = $2`,
    [fullKind, WEEK],
  )
  if (Number(fullCheck.rows[0].n) === 0) {
    if (fs.existsSync(fullFile)) {
      const blob = fs.readFileSync(fullFile)
      const result = await uploadArtifact({
        kind: fullKind,
        week: WEEK,
        blob,
        generated_at: new Date(),
        source_path: fullFile,
      })
      console.log('')
      console.log(`  ✓ Full Substrate (v3 layer2) also ingested: size=${(blob.length / 1024).toFixed(0)}KB sha16=${result.sha256.slice(0, 16)}`)
    } else {
      console.log('')
      console.log(`  ?  Full Substrate file not found at ${fullFile} — already in DB? skipping`)
    }
  } else {
    console.log('')
    console.log(`  ✓ Full Substrate (v3 layer2) already in cr_artifacts — not re-ingesting`)
  }

  await db.end()
}

main().catch((e: unknown) => { console.error(e); process.exit(1) })
