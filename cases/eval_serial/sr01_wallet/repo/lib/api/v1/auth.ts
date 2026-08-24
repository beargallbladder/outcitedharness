/**
 * API authentication — reads Clerk session, returns the canonical
 * ApiUser shape used by every /v1/* route.
 *
 * Auth flow (no API keys yet; Clerk session JWT is the auth):
 *
 *   1. Request arrives at /v1/<endpoint>.
 *   2. Clerk's middleware (configured in middleware.ts) attaches the
 *      session to the request. If absent → 401 outside this function.
 *   3. We read publicMetadata from the Clerk session:
 *        - api_tier       'free' | 'pro' | 'enterprise' | 'custom'
 *        - trial_ends_at  ISO date when 30-day free trial expires
 *        - wallet_slugs   array of slugs paid users can query
 *        - free_slug      single preassigned slug for free-tier users
 *        - tracked_brand  the customer's primary brand domain
 *   4. We compute trial_expired by comparing trial_ends_at to now.
 *   5. We return the ApiUser. Caller decides what to do with it.
 *
 * If a user has no publicMetadata yet (just-signed-up, before our webhook
 * has fired), they get default-free with trial_ends_at = now + 30d
 * INFERRED (we don't write it back; the webhook will). This prevents
 * race conditions where a brand-new user gets locked out.
 *
 * API key flow (LIVE 2026-07-17 — Sam directive: API is the product):
 * routes accept EITHER a Clerk session JWT OR `Authorization: Bearer
 * crk_live_…`. Keys live in the api_keys table (sha256 hash only) with
 * their own tier + wallet, minted via scripts/api/mint-api-key.ts.
 * Bearer is checked FIRST so server-to-server consumers (Octopart)
 * never touch Clerk.
 */
import 'server-only'

import type { NextRequest } from 'next/server'
import { auth, currentUser } from '@clerk/nextjs/server'
import { parseApiTier, DEFAULT_FREE_SLUG, type ApiTier } from './tiers'
import { resolveApiKey } from './api-keys'

export interface ApiUser {
  userId: string
  email: string | null
  api_tier: ApiTier
  /** Wallet of slugs the user can query (3 for pro, 5 for enterprise). */
  wallet_slugs: string[]
  /** For free-tier users only: the single preassigned slug. */
  free_slug: string | null
  /** The customer's primary brand domain (e.g. "acme.com"). */
  tracked_brand: string | null
  /** ISO timestamp when the 30-day free trial expires. */
  trial_ends_at: string | null
  /** True iff trial_ends_at is in the past. */
  trial_expired: boolean
}

const TRIAL_DURATION_MS = 30 * 24 * 60 * 60 * 1000

/**
 * Read the current API user. Returns null if neither a bearer key nor a
 * signed-in Clerk session is present.
 *
 * NOTE: This is a CR-side abstraction; we don't fail with 401 here.
 * Callers (typically `withApiAuth`) decide what to do with null.
 */
export async function getApiUser(req?: NextRequest): Promise<ApiUser | null> {
  // Bearer key first — server-to-server consumers have no Clerk session.
  const authz = req?.headers.get('authorization')
  if (authz?.toLowerCase().startsWith('bearer ')) {
    const key = authz.slice(7).trim()
    const identity = await resolveApiKey(key)
    if (identity) {
      return {
        userId: `key:${identity.key_id}`,
        email: null,
        api_tier: identity.api_tier,
        wallet_slugs: identity.wallet_slugs,
        free_slug: null,
        tracked_brand: identity.tracked_brand,
        trial_ends_at: null,
        trial_expired: false,
      }
    }
    // A malformed/revoked bearer key is a hard auth failure — do NOT
    // fall through to the Clerk session (that would mask revocation).
    if (key.startsWith('crk_')) return null
  }

  const { userId } = await auth()
  if (!userId) return null

  const user = await currentUser()
  if (!user) return null

  const meta = user.publicMetadata as Record<string, unknown> | null

  // Tier — default to free for any user without explicit tier set.
  const tier = parseApiTier(meta?.api_tier) ?? 'free'

  // Trial expiration — only relevant for free-tier users.
  let trial_ends_at: string | null = null
  let trial_expired = false
  if (tier === 'free') {
    const raw = meta?.trial_ends_at
    if (typeof raw === 'string') {
      trial_ends_at = raw
      try {
        trial_expired = new Date(raw).getTime() < Date.now()
      } catch {
        trial_expired = false
      }
    } else {
      // No trial_ends_at yet (webhook hasn't fired or user predates the
      // tier system). Infer 30 days from createdAt to avoid locking out
      // brand-new users mid-flight. Webhook will write the real value
      // shortly.
      const created = user.createdAt ? new Date(user.createdAt) : new Date()
      const ends = new Date(created.getTime() + TRIAL_DURATION_MS)
      trial_ends_at = ends.toISOString()
      trial_expired = ends.getTime() < Date.now()
    }
  }

  // Wallet of slugs the user can query.
  const wallet_slugs = Array.isArray(meta?.wallet_slugs)
    ? (meta!.wallet_slugs as unknown[]).filter((s): s is string => typeof s === 'string')
    : []

  // Free-tier preassigned slug.
  const free_slug =
    typeof meta?.free_slug === 'string'
      ? (meta.free_slug as string)
      : tier === 'free'
        ? DEFAULT_FREE_SLUG
        : null

  // Customer's primary brand domain.
  const tracked_brand =
    typeof meta?.tracked_brand === 'string' ? (meta.tracked_brand as string).toLowerCase() : null

  // Primary email.
  const email =
    user.primaryEmailAddress?.emailAddress ??
    user.emailAddresses?.[0]?.emailAddress ??
    null

  return {
    userId,
    email,
    api_tier: tier,
    wallet_slugs,
    free_slug,
    tracked_brand,
    trial_ends_at,
    trial_expired,
  }
}

/**
 * For routes that need to know if a slug is in the user's accessible set.
 *
 *   - Free users: only `free_slug` is accessible.
 *   - Pro/Enterprise: only slugs in `wallet_slugs` are accessible.
 *   - Custom: unlimited.
 *
 * Used by both endpoint gates and wallet enforcement.
 */
export function slugInWallet(user: ApiUser, slug: string): boolean {
  if (user.api_tier === 'custom') return true
  if (user.api_tier === 'free') {
    return user.free_slug === slug
  }
  return user.wallet_slugs.includes(slug)
}
