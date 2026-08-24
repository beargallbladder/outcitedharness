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
