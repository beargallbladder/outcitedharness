/**
 * Auth + tier + rate-limit wrapper for /v1/* route handlers.
 *
 * Usage:
 *
 *   import { withApiAuth } from '@/lib/api/v1/with-auth'
 *
 *   export const GET = withApiAuth(
 *     { endpoint: 'category.leaderboard', extract_slug: (req) => req.nextUrl.pathname.match(/category\/([^/]+)/)?.[1] },
 *     async (req, ctx) => {
 *       // ctx.user, ctx.tier_config, ctx.slug
 *       return NextResponse.json({ ... })
 *     }
 *   )
 *
 * Pipeline:
 *   1. Read Clerk session → ApiUser (or 401 if absent).
 *   2. Resolve user's tier_config from API_TIERS.
 *   3. Check trial expiration → 402 Payment Required if free trial ended.
 *   4. Check endpoint is in user's tier.allowed_endpoints → 403 if not.
 *   5. Extract slug param + check wallet → 403 if slug not in wallet.
 *   6. Check rate limit → 429 if exceeded.
 *   7. Invoke handler with augmented ctx.
 *   8. Decorate response with rate-limit headers + Cache-Control hint.
 */
import 'server-only'

import { NextRequest, NextResponse } from 'next/server'
import {
  API_TIERS,
  endpointGate,
  type ApiEndpoint,
  type ApiTierConfig,
} from './tiers'
import { getApiUser, slugInWallet, type ApiUser } from './auth'
import { checkAndIncrement, rateLimitHeaders } from './rate-limit'

export interface ApiContext<P = Record<string, string>> {
  user: ApiUser
  tier_config: ApiTierConfig
  /** Slug parameter extracted from the URL/query, if applicable. */
  slug: string | null
  /** Resolved Next.js route params (e.g. { slug: 'power-supplies' }). */
  params: P
}

export interface WithApiAuthOptions<P = Record<string, string>> {
  /** Logical endpoint id (matches ApiEndpoint enum). Used for tier gating + upgrade hints. */
  endpoint: ApiEndpoint
  /** Extract the slug parameter from the request, if this endpoint is slug-scoped. */
  extract_slug?: (req: NextRequest, params: P) => string | null | Promise<string | null>
  /** Cache-Control max-age in seconds. Default: 3600 (1h) for weekly-refresh data. */
  cache_max_age_seconds?: number
}

type ApiHandler<P = Record<string, string>> = (
  req: NextRequest,
  ctx: ApiContext<P>
) => Promise<NextResponse>

/** Next.js v15 route-handler signature (params is a Promise). */
interface NextRouteCtx<P> {
  params: Promise<P>
}

const NEXT_REFRESH_HINT_HOURS = 1

function upgradePromptForTier(
  user: ApiUser,
  required_tier: string | null
): Record<string, unknown> {
  return {
    current_tier: user.api_tier,
    required_tier,
    upgrade_url: 'https://www.categoryrank.ai/pricing',
    contact_url: 'https://www.categoryrank.ai/contact',
  }
}

export function withApiAuth<P = Record<string, string>>(
  opts: WithApiAuthOptions<P>,
  handler: ApiHandler<P>
) {
  return async function wrapped(
    req: NextRequest,
    routeCtx?: NextRouteCtx<P>
  ): Promise<NextResponse> {
    const t0 = Date.now()
    const params = routeCtx ? await routeCtx.params : ({} as P)

    // STEP 1: Read user (bearer API key first, then Clerk session).
    const user = await getApiUser(req)
    if (!user) {
      return NextResponse.json(
        {
          error: 'unauthenticated',
          message: 'Sign in at https://www.categoryrank.ai/signin to use the API.',
          _meta: {
            endpoint_version: 'v1',
            signup_url: 'https://www.categoryrank.ai/signup',
            pricing_url: 'https://www.categoryrank.ai/pricing',
          },
        },
        { status: 401, headers: { 'WWW-Authenticate': 'Bearer realm="categoryrank-v1"' } }
      )
    }

    const tier_config = API_TIERS[user.api_tier]

    // STEP 2: Trial expiration (free tier only).
    if (user.api_tier === 'free' && user.trial_expired) {
      return NextResponse.json(
        {
          error: 'trial_expired',
          message: `Your 30-day free trial expired on ${user.trial_ends_at}. Upgrade to Pro to continue.`,
          _meta: {
            endpoint_version: 'v1',
            trial_ends_at: user.trial_ends_at,
            ...upgradePromptForTier(user, 'pro'),
          },
        },
        { status: 402 }
      )
    }

    // STEP 3: Endpoint gate.
    const gate = endpointGate(tier_config, opts.endpoint)
    if (!gate.allowed) {
      return NextResponse.json(
        {
          error: 'tier_required',
          message: `This endpoint requires the ${gate.required_tier} tier or higher.`,
          _meta: {
            endpoint_version: 'v1',
            endpoint: opts.endpoint,
            ...upgradePromptForTier(user, gate.required_tier),
          },
        },
        { status: 403 }
      )
    }

    // STEP 4: Slug wallet gate.
    let slug: string | null = null
    if (opts.extract_slug) {
      const extracted = await opts.extract_slug(req, params)
      slug = typeof extracted === 'string' ? extracted : null
    }
    if (slug !== null && !slugInWallet(user, slug)) {
      const slug_allowed =
        user.api_tier === 'free'
          ? user.free_slug
            ? [user.free_slug]
            : []
          : user.wallet_slugs
      const reason =
        user.api_tier === 'free'
          ? `Your free trial is preassigned to '${user.free_slug}'. Upgrade to Pro to add up to 3 slugs to your wallet.`
          : opts.endpoint.startsWith('family.')
            ? `'${slug}' is not in your Brand Families wallet. Entitled aisle(s): ${
                slug_allowed.length ? slug_allowed.join(', ') : '(none)'
              }.`
            : `'${slug}' is not in your wallet. Your ${tier_config.display_name} tier allows ${tier_config.wallet_size} slugs.`
      return NextResponse.json(
        {
          error: 'slug_not_in_wallet',
          /** Stable family:read alias — aisle exists but outside wallet. */
          code: opts.endpoint.startsWith('family.')
            ? 'aisle_not_entitled'
            : 'slug_not_in_wallet',
          message: reason,
          _meta: {
            endpoint_version: 'v1',
            slug_requested: slug,
            slug_allowed,
            ...upgradePromptForTier(user, tier_config.upgrade_to),
          },
        },
        { status: 403 }
      )
    }

    // STEP 5: Rate limit.
    const rate = checkAndIncrement(user.userId, tier_config, { slug })
    if (!rate.allowed) {
      const reason_msg = {
        burst: 'Per-minute burst limit reached.',
        day: 'Daily request limit reached.',
        month: 'Monthly request limit reached.',
        slug_day: 'Per-slug daily limit reached.',
        ok: 'unreachable',
      }[rate.reason]
      return NextResponse.json(
        {
          error: 'rate_limited',
          message: `${reason_msg} Retry in ${rate.retry_after_seconds}s or upgrade.`,
          _meta: {
            endpoint_version: 'v1',
            reason: rate.reason,
            reset_at: rate.reset_at?.toISOString() ?? null,
            ...upgradePromptForTier(user, tier_config.upgrade_to),
          },
        },
        {
          status: 429,
          headers: {
            'Retry-After': String(rate.retry_after_seconds),
            ...rateLimitHeaders(rate),
          },
        }
      )
    }

    // STEP 6: Invoke handler.
    let response: NextResponse
    try {
      response = await handler(req, { user, tier_config, slug, params })
    } catch (err: unknown) {
      console.error(`[v1:${opts.endpoint}]`, err)
      return NextResponse.json(
        {
          error: 'internal_error',
          message: 'An unexpected error occurred. Try again in a moment.',
          _meta: { endpoint_version: 'v1', endpoint: opts.endpoint },
        },
        { status: 500 }
      )
    }

    // STEP 7: Decorate response with rate-limit + cache headers.
    const cache_seconds = opts.cache_max_age_seconds ?? 3600
    const headers = new Headers(response.headers)
    for (const [k, v] of Object.entries(rateLimitHeaders(rate))) {
      headers.set(k, v)
    }
    headers.set('Cache-Control', `private, max-age=${cache_seconds}`)
    headers.set('X-Request-Duration-Ms', String(Date.now() - t0))
    headers.set('X-API-Tier', user.api_tier)
    headers.set('X-Next-Refresh-Hint', `~${NEXT_REFRESH_HINT_HOURS}h`)

    // Re-wrap with augmented headers
    return new NextResponse(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    })
  }
}

/**
 * Helper: attach the canonical `_meta` block to a successful response body.
 * Routes can call this just before NextResponse.json() to keep envelope
 * conventions consistent across endpoints.
 */
export function decorateMeta<P = Record<string, string>>(
  body: Record<string, unknown>,
  ctx: ApiContext<P>,
  extra: Record<string, unknown> = {}
): Record<string, unknown> {
  const meta = (body._meta as Record<string, unknown>) ?? {}
  body._meta = {
    endpoint_version: 'v1',
    tier: ctx.user.api_tier,
    wallet_size: ctx.tier_config.wallet_size,
    upgrade_url:
      ctx.tier_config.upgrade_to !== null
        ? 'https://www.categoryrank.ai/pricing'
        : null,
    upgrade_to: ctx.tier_config.upgrade_to,
    ...meta,
    ...extra,
  }
  return body
}
