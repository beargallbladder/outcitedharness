import 'server-only'

import { NextRequest, NextResponse } from 'next/server'
import { getApiUser, slugInWallet, type ApiUser } from '@/lib/api/v1/auth'
import { API_TIERS, type ApiTierConfig } from '@/lib/api/v1/tiers'
import { checkAndIncrement, rateLimitHeaders } from '@/lib/api/v1/rate-limit'
import {
  DesignWinsReleaseError,
  loadDesignWinsRelease,
  type LoadedDesignWinsRelease,
} from '@/lib/designwins/facade/release'
import { designWinsProblem } from '@/lib/designwins/facade/problem'
import {
  EDGE_AUTH_HEADER,
  edgeAuthSecret,
  verifyEdgeContext,
  type EdgeAuthContext,
} from '@/lib/designwins/facade/edge-auth'

export type DesignWinsContext<P = Record<string, string>> = {
  user: ApiUser
  tier: ApiTierConfig
  params: P
  aisle: string | null
  release: LoadedDesignWinsRelease
}

type RouteContext<P> = {
  params: Promise<P>
}

type Handler<P> = (
  request: NextRequest,
  context: DesignWinsContext<P>,
) => Promise<NextResponse>

type Options<P> = {
  aisle?: (_request: NextRequest, params: P) => string | null
  requireSameOrigin?: boolean
}

function privateTiers(): Set<string> {
  return new Set(
    (process.env.DESIGNWINS_PRIVATE_TIERS ?? 'enterprise,custom')
      .split(',')
      .map((value) => value.trim())
      .filter(Boolean),
  )
}

function allowedOrigins(request: NextRequest): string[] {
  const configured = process.env.DESIGNWINS_ALLOWED_ORIGIN?.trim()
  if (!configured) return [request.nextUrl.origin]
  // Comma-separated Preview origins (www + apex + frontend candidates).
  return configured
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean)
}

function sameOrigin(request: NextRequest, edgeAuthed: boolean): boolean {
  const origin = request.headers.get('origin')
  if (!origin) {
    // A verified edge context means the original request carried a bearer
    // key — middleware stripped the Authorization header to make the
    // response CDN-cacheable (see edge-auth.ts).
    if (edgeAuthed) return true
    return request.headers
      .get('authorization')
      ?.toLowerCase()
      .startsWith('bearer ') === true
  }
  try {
    const requestOrigin = new URL(origin).origin
    return allowedOrigins(request).some((expected) => {
      try {
        return requestOrigin === new URL(expected).origin
      } catch {
        return false
      }
    })
  } catch {
    return false
  }
}

export function withDesignWinsAuth<P = Record<string, string>>(
  options: Options<P>,
  handler: Handler<P>,
) {
  return async function designWinsAuthenticatedRoute(
    request: NextRequest,
    routeContext?: RouteContext<P>,
  ): Promise<NextResponse> {
    const params = routeContext
      ? await routeContext.params
      : ({} as P)
    // Edge-authed path (FACADE-PERF-V1): middleware already resolved the
    // bearer key against the DB, enforced the wallet, stripped the
    // Authorization header (Vercel's CDN never caches authed requests) and
    // signed the identity into x-designwins-edge-auth. A valid HMAC here
    // reconstructs the ApiUser without a DB round-trip and makes the 200
    // response edge-cacheable. An invalid/forged header is ignored and the
    // request falls through to full route-side auth.
    const edgeCtx: EdgeAuthContext | null = verifyEdgeContext(
      request.headers.get(EDGE_AUTH_HEADER),
      edgeAuthSecret(),
    )
    const user: ApiUser | null = edgeCtx
      ? {
          userId: `key:${edgeCtx.key_id}`,
          email: null,
          api_tier: (edgeCtx.api_tier as ApiUser['api_tier']) ?? 'enterprise',
          wallet_slugs: edgeCtx.wallet_slugs,
          free_slug: null,
          tracked_brand: edgeCtx.tracked_brand,
          trial_ends_at: null,
          trial_expired: false,
        }
      : await getApiUser(request)
    if (!user) {
      return designWinsProblem(
        401,
        'unauthenticated',
        'Authentication required',
        {
          detail: 'Sign in or provide a valid CategoryRank API key.',
          headers: {
            'WWW-Authenticate': 'Bearer realm="designwins-v1"',
          },
        },
      )
    }
    if (options.requireSameOrigin && !sameOrigin(request, edgeCtx !== null)) {
      return designWinsProblem(403, 'forbidden', 'Cross-site request rejected')
    }

    let release: LoadedDesignWinsRelease
    try {
      release = loadDesignWinsRelease()
    } catch (error) {
      if (error instanceof DesignWinsReleaseError) {
        return designWinsProblem(error.status, error.code, 'Release unavailable', {
          detail: error.message,
          retryable: error.status === 503,
          releaseId: error.release_id,
        })
      }
      return designWinsProblem(
        503,
        'release_unavailable',
        'Release unavailable',
        { retryable: false },
      )
    }

    if (
      release.manifest.release.approval === 'private' &&
      !privateTiers().has(user.api_tier)
    ) {
      return designWinsProblem(
        403,
        'forbidden',
        'Private DesignWins release is not available for this account',
        { releaseId: release.manifest.release.id },
      )
    }

    const aisle = options.aisle?.(request, params) ?? null
    if (aisle && !slugInWallet(user, aisle)) {
      return designWinsProblem(403, 'forbidden', 'Aisle is not in this wallet', {
        detail: `The authenticated account is not entitled to '${aisle}'.`,
        releaseId: release.manifest.release.id,
      })
    }

    const tier = API_TIERS[user.api_tier]
    const rate = checkAndIncrement(user.userId, tier, { slug: aisle })
    if (!rate.allowed) {
      return designWinsProblem(429, 'rate_limited', 'Rate limit exceeded', {
        retryable: true,
        releaseId: release.manifest.release.id,
        headers: {
          'Retry-After': String(rate.retry_after_seconds),
          ...rateLimitHeaders(rate),
        },
      })
    }

    try {
      const response = await handler(request, {
        user,
        tier,
        params,
        aisle,
        release,
      })
      const headers = new Headers(response.headers)
      if (edgeCtx && request.method === 'GET' && response.status === 200) {
        // Release-scoped responses are immutable within a deployment (the
        // release pin is an env var, a new pin is a new deployment, and
        // Vercel's CDN cache is deployment-scoped), so cache effectively
        // forever at the edge. Vary keys the cache on the signed identity
        // header: requests without a valid signed context (unauthenticated,
        // forged, Clerk-session) can never match a populated slot.
        // Rate-limit headers are intentionally omitted here — they would be
        // frozen into the cached copy; rate limiting still applies on every
        // cache MISS (i.e. every function invocation).
        headers.set(
          'Cache-Control',
          'public, s-maxage=31536000, stale-while-revalidate=86400',
        )
        headers.set('Vary', EDGE_AUTH_HEADER)
      } else {
        headers.set('Cache-Control', 'private, no-store')
        for (const [key, value] of Object.entries(rateLimitHeaders(rate))) {
          headers.set(key, value)
        }
      }
      return new NextResponse(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      })
    } catch (error) {
      if (error instanceof DesignWinsReleaseError) {
        return designWinsProblem(error.status, error.code, 'Release gate failed', {
          detail: error.message,
          retryable: error.status === 503,
          releaseId: error.release_id ?? release.manifest.release.id,
        })
      }
      console.error('[designwins-v1]', error)
      return designWinsProblem(
        503,
        'upstream_unavailable',
        'DesignWins service unavailable',
        {
          retryable: true,
          releaseId: release.manifest.release.id,
        },
      )
    }
  }
}
