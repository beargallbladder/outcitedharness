/**
 * Week Status (DB-based) — the truth-teller for weekly collection.
 *
 * Single source of truth for "what's the state of W20 collection?"
 * Always queries the DB, never reads docs or scripts. The DB is the
 * truth; docs and orchestrator scripts drift.
 *
 * What this answers in one command:
 *   - Active lanes for the target week (rows in category_mentions_v2)
 *   - Per-lane status (realtime done / batch submitted / failed / stale)
 *   - Missing lanes vs the previous week's actual lineup
 *   - batch_jobs table snapshot (provider × status)
 *   - Pipeline manifest stage status
 *   - Drift detection: backfill-week.sh's lane list vs reference week
 *
 * Usage:
 *   npx tsx scripts/ops/week-status.ts 2026-W20
 *   npx tsx scripts/ops/week-status.ts 2026-W20 --reference=2026-W18
 *
 * Built 2026-05-11 after a Monday morning where the orchestrator script
 * (backfill-week.sh) had drifted from the actual W19 lineup and nobody
 * noticed for a week. This command makes that drift impossible to miss.
 */
import 'dotenv/config'
import { resolve } from 'path'
import dotenv from 'dotenv'
dotenv.config({ path: resolve(process.cwd(), '.env.local'), override: true })

import { db } from '@/lib/db'
import * as fs from 'fs'
import * as path from 'path'

// ────────────────────────────────────────────────────────────────────────
// arg parsing
// ────────────────────────────────────────────────────────────────────────

const ARGS = process.argv.slice(2)
function getArg(name: string): string | undefined {
  const a = ARGS.find(x => x.startsWith(`--${name}=`))
  return a ? a.split('=')[1] : undefined
}

const week =
  getArg('week') ||
  getArg('timeWindow') ||
  ARGS.find(a => /^\d{4}-W\d{2}$/.test(a))
if (!week) {
  console.error('Usage: npx tsx scripts/ops/week-status.ts <YYYY-Www> [--reference=YYYY-Www]')
  process.exit(2)
}

// Default reference = previous ISO week
function prevWeek(w: string): string {
  const m = w.match(/^(\d{4})-W(\d{2})$/)
  if (!m) return w
  const yr = parseInt(m[1], 10)
  const wk = parseInt(m[2], 10)
  if (wk > 1) return `${yr}-W${String(wk - 1).padStart(2, '0')}`
  return `${yr - 1}-W52` // good enough; W53 edge cases can pass --reference
}

const reference = getArg('reference') || prevWeek(week)

// ────────────────────────────────────────────────────────────────────────
// formatting helpers
// ────────────────────────────────────────────────────────────────────────

function pad(s: string | number, w: number, align: 'left' | 'right' = 'left'): string {
  const str = String(s)
  if (str.length >= w) return str.slice(0, w)
  const padding = ' '.repeat(w - str.length)
  return align === 'left' ? str + padding : padding + str
}

function fmtN(n: number | string | null | undefined): string {
  const v = typeof n === 'number' ? n : parseInt(String(n ?? 0), 10)
  if (!Number.isFinite(v) || v === 0) return '0'
  return v.toLocaleString('en-US')
}

function ago(iso: string | null | undefined): string {
  if (!iso) return '—'
  // Postgres `timestamp::text` drops the timezone suffix. If the string
  // has no `Z` or `+00`, append `Z` to force UTC parsing — created_at is
  // stored as UTC in our schema.
  const isoNorm =
    /Z$|[+-]\d{2}:\d{2}$/.test(iso) ? iso : iso.replace(' ', 'T') + 'Z'
  const t = new Date(isoNorm).getTime()
  if (!Number.isFinite(t)) return '—'
  const sec = Math.floor((Date.now() - t) / 1000)
  if (sec < 0) return 'just now'
  if (sec < 60) return `${sec}s ago`
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`
  return `${Math.floor(sec / 86400)}d ago`
}

function laneStatus(row: { brands: number; last_write_iso: string | null }, cohortSize: number): string {
  const fullCohort = cohortSize > 0 ? row.brands / cohortSize : 0
  const last_t = row.last_write_iso ? new Date(row.last_write_iso).getTime() : 0
  const secAgo = last_t ? (Date.now() - last_t) / 1000 : Infinity
  if (fullCohort >= 0.95) return '✓ done'
  if (secAgo < 30 * 60) return '⚡ writing'
  if (fullCohort >= 0.5) return '◐ partial (idle)'
  if (fullCohort >= 0.05) return '⚠ stalled'
  return '✗ thin'
}

function rule(s = '─', n = 78): string {
  return s.repeat(n)
}

// ────────────────────────────────────────────────────────────────────────
// queries
// ────────────────────────────────────────────────────────────────────────

interface LaneRow {
  model: string
  prompt_version: string
  brands: number
  rows: number
  last_write_iso: string | null
}

async function activeLanes(weekId: string): Promise<LaneRow[]> {
  const r = await db.query<LaneRow>(`
    SELECT
      model,
      prompt_version,
      COUNT(DISTINCT brand_id)::int AS brands,
      COUNT(*)::int AS rows,
      MAX(created_at)::text AS last_write_iso
    FROM category_mentions_v2
    WHERE time_window = $1 AND category <> '__unknown__'
    GROUP BY model, prompt_version
    ORDER BY brands DESC, rows DESC
  `, [weekId])
  return r.rows
}

interface BatchJobRow {
  provider: string
  status: string
  n: number
}

async function batchJobs(weekId: string): Promise<BatchJobRow[]> {
  const r = await db.query<BatchJobRow>(`
    SELECT provider, status, COUNT(*)::int AS n
    FROM batch_jobs
    WHERE time_window = $1
    GROUP BY provider, status
    ORDER BY provider, status
  `, [weekId])
  return r.rows
}

interface ManifestRow {
  time_window: string
  status: string | null
  started_at: string | null
  completed_at: string | null
  stages: Record<string, { status?: string; counts?: Record<string, unknown> }> | null
}

async function pipelineManifest(weekId: string): Promise<ManifestRow | null> {
  const r = await db.query<ManifestRow>(`
    SELECT time_window, status, started_at::text, completed_at::text, stages
    FROM week_pipeline_manifest
    WHERE time_window = $1
  `, [weekId])
  return r.rows[0] ?? null
}

async function cohortSize(weekId: string): Promise<number> {
  // The "true cohort" is the union of brands seen across all lanes for the
  // week (more honest than a static brand count, because the cohort can
  // grow as we add brands).
  const r = await db.query<{ n: number }>(`
    SELECT COUNT(DISTINCT brand_id)::int AS n
    FROM category_mentions_v2
    WHERE time_window = $1 AND category <> '__unknown__'
  `, [weekId])
  return r.rows[0]?.n ?? 0
}

// ────────────────────────────────────────────────────────────────────────
// drift detection: backfill-week.sh's lane list vs reference week
// ────────────────────────────────────────────────────────────────────────

function parseBackfillScriptLanes(): Set<string> {
  const scriptPath = path.join(process.cwd(), 'scripts/week/backfill-week.sh')
  const lanes = new Set<string>()
  try {
    const text = fs.readFileSync(scriptPath, 'utf8')
    // Match `maybe_submit "MODEL_LABEL"` invocations. The DB label is the
    // first quoted token after maybe_submit. There are also script-name
    // labels (openai-mini, anthropic-haiku) but those map to DB labels
    // via the surrounding script invocations — we parse the first token.
    const re = /maybe_submit\s+"([^"]+)"/g
    let m: RegExpExecArray | null
    while ((m = re.exec(text)) !== null) {
      lanes.add(m[1])
    }
  } catch {
    // No backfill script or unreadable; return empty set so drift section
    // just prints a warning.
  }
  return lanes
}

// ────────────────────────────────────────────────────────────────────────
// main
// ────────────────────────────────────────────────────────────────────────

async function main() {
  console.log(rule('═'))
  console.log(`WEEK STATUS · ${week}  (reference: ${reference})`)
  console.log(`queried at ${new Date().toISOString()}`)
  console.log(rule('═'))

  // ─── Active lanes ───────────────────────────────────────────────────
  const [currentLanes, referenceLanes, currentCohort, referenceCohort] = await Promise.all([
    activeLanes(week),
    activeLanes(reference),
    cohortSize(week),
    cohortSize(reference),
  ])

  console.log(`\nACTIVE LANES IN DB · ${week}    (cohort union: ${fmtN(currentCohort)} brands)`)
  console.log(rule())
  if (currentLanes.length === 0) {
    console.log('  ✗ no rows yet for this week')
  } else {
    console.log(
      pad('  status', 18) +
        pad('model', 24) +
        pad('brands', 9, 'right') +
        pad('rows', 12, 'right') +
        pad('  prompt_ver', 22) +
        '  last write'
    )
    for (const lane of currentLanes) {
      const status = laneStatus(lane, currentCohort || 1)
      console.log(
        pad('  ' + status, 18) +
          pad(lane.model, 24) +
          pad(fmtN(lane.brands), 9, 'right') +
          pad(fmtN(lane.rows), 12, 'right') +
          pad('  ' + lane.prompt_version, 22) +
          '  ' +
          ago(lane.last_write_iso)
      )
    }
  }

  // ─── batch_jobs (read early — we need to distinguish in-flight from missing) ──
  const bj = await batchJobs(week)
  // Provider → model mapping so we can identify which "missing" lanes are actually in flight
  const PROVIDER_OF_MODEL: Record<string, string> = {
    'gpt-4o-mini': 'openai',
    'gpt-4o': 'openai',
    'gpt-5': 'openai',
    'anthropic': 'anthropic',
    'anthropic-sonnet': 'anthropic',
    'claude-opus-4-7': 'anthropic',
    'mistral': 'mistral',
    'mistral-large': 'mistral',
    'xai': 'xai',
    'gemini-flash': 'gemini',
    'gemini-pro': 'gemini',
    'deepseek-r1': 'together',
    'deepseek-v3.1': 'together',
    'qwen': 'together',
    'llama-3.3-70b': 'together',
    'groq-llama': 'groq',
    'cohere': 'cohere',
  }
  const providersWithBatch = new Set(bj.filter(r => r.status === 'submitted' || r.status === 'running' || r.status === 'pending').map(r => r.provider))

  // ─── Missing vs reference ───────────────────────────────────────────
  const currentLaneNames = new Set(currentLanes.map(l => l.model))
  const referenceLaneNames = new Set(referenceLanes.map(l => l.model))
  const missingVsReference = [...currentLaneNames].filter(m => !referenceLaneNames.has(m))

  console.log(`\nMISSING VS ${reference}    (lanes in ref that are absent in ${week})`)
  console.log(rule())
  if (missingVsReference.length === 0) {
    console.log('  ✓ no missing lanes — current week meets or exceeds reference lineup')
  } else {
    for (const m of missingVsReference) {
      const refRow = referenceLanes.find(r => r.model === m)
      const provider = PROVIDER_OF_MODEL[m]
      const inFlight = provider ? providersWithBatch.has(provider) : false
      const inFlightTag = inFlight
        ? ' [batch in flight, awaiting import]'
        : ' [NEVER SUBMITTED]'
      console.log(
        '  ✗ ' +
          pad(m, 24) +
          ' (ref ' + reference + ' had ' + fmtN(refRow?.brands ?? 0) + ' brands)' +
          inFlightTag
      )
    }
  }

  console.log(`\nbatch_jobs · ${week}`)
  console.log(rule())
  if (bj.length === 0) {
    console.log('  (no batch_jobs rows for this week)')
  } else {
    const byProvider = new Map<string, Map<string, number>>()
    for (const row of bj) {
      if (!byProvider.has(row.provider)) byProvider.set(row.provider, new Map())
      byProvider.get(row.provider)!.set(row.status, row.n)
    }
    for (const [provider, statuses] of byProvider) {
      const parts = [...statuses].map(([s, n]) => `${s}=${n}`).join('  ')
      console.log('  ' + pad(provider, 14) + '  ' + parts)
    }
  }

  // ─── STALL WATCH (Sam-lock 2026-06-10 reliability layer) ────────────
  // Flag any batch still in_progress/submitted past 3h. A 9h Together
  // stall on 2026-W24 went undetected because nothing surfaced this.
  // The truth-teller now screams about it.
  const STALL_HOURS = 3
  const stalls = await db.query<{ provider: string; batch_id: string; hours: number }>(`
    SELECT provider, batch_id,
           ROUND((EXTRACT(EPOCH FROM (now() - submitted_at)) / 3600.0)::numeric, 1) AS hours
    FROM batch_jobs
    WHERE time_window = $1
      AND status IN ('submitted','in_progress','pending','running')
      AND submitted_at < now() - ($2 || ' hours')::interval
    ORDER BY hours DESC
  `, [week, String(STALL_HOURS)])
  if (stalls.rows.length > 0) {
    console.log(`\n⚠ STALL WATCH · batches in_progress > ${STALL_HOURS}h`)
    console.log(rule())
    for (const s of stalls.rows) {
      console.log(`  ⚠ ${pad(s.provider, 12)} ${s.batch_id.slice(0, 24)}…  ${s.hours}h — provider-side stall; cancel + resubmit or finalize without it`)
    }
  }

  // ─── Pipeline manifest ──────────────────────────────────────────────
  const manifest = await pipelineManifest(week)
  console.log(`\nPIPELINE MANIFEST · ${week}`)
  console.log(rule())
  if (!manifest) {
    console.log(`  (no week_pipeline_manifest row for ${week})`)
  } else {
    console.log(
      `  status=${manifest.status ?? '—'}  started=${(manifest.started_at ?? '—').slice(0, 19)}  completed=${(manifest.completed_at ?? '—').slice(0, 19)}`
    )
    const stages = manifest.stages ?? {}
    const stageOrder = [
      'collect', 'import', 'embed', 'kim-tag', 'tensors', 'centroids',
      'drift', 'alignment', 'worldview', 'validate', 'publish',
    ]
    for (const s of stageOrder) {
      const v = (stages as Record<string, { status?: string }>)[s]
      if (!v) continue
      console.log('  ' + pad(s, 12) + '  ' + (v.status ?? '—'))
    }
  }

  // ─── Orchestrator drift ─────────────────────────────────────────────
  const scriptLanes = parseBackfillScriptLanes()
  console.log(`\nORCHESTRATOR DRIFT · backfill-week.sh vs ${reference} actual lineup`)
  console.log(rule())
  if (scriptLanes.size === 0) {
    console.log('  ⚠ could not parse scripts/week/backfill-week.sh — manual audit recommended')
  } else {
    const scriptOnly = [...scriptLanes].filter(s => !referenceLaneNames.has(s))
    const referenceOnly = [...referenceLaneNames].filter(s => !scriptLanes.has(s))
    console.log(`  script submits ${scriptLanes.size} lanes; reference week has ${referenceLaneNames.size}`)
    if (scriptOnly.length === 0 && referenceOnly.length === 0) {
      console.log('  ✓ in sync — orchestrator matches reference week lineup')
    } else {
      if (scriptOnly.length) {
        console.log('  ✗ script tries to submit but reference does not have:')
        for (const m of scriptOnly) console.log(`      ${m}   (likely retired)`)
      }
      if (referenceOnly.length) {
        console.log('  ✗ reference has but script does not submit:')
        for (const m of referenceOnly) console.log(`      ${m}   (orchestrator needs sync)`)
      }
    }
  }

  console.log()
  console.log(rule('═'))
  console.log('truth-teller done. DB is the source of truth; trust this over any doc.')
  console.log(rule('═'))
}

main()
  .then(() => process.exit(0))
  .catch(err => {
    console.error('week-status: error:', err)
    process.exit(1)
  })
