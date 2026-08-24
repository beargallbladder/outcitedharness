Production checkout (isolated). Task: fix entitlement.

`lib/api/v1/auth.ts` `slugInWallet` returns true for every `api_tier==='custom'`
key, so `wallet_slugs` is decorative. Callers in `with-auth.ts` and
`lib/designwins/facade/auth.ts` trust that function.

Requirements:
1. A custom key with wallet_slugs=['microcontrollers'] must NOT read other aisles.
2. Free tier still uses free_slug.
3. Keep a genuine all-access path for internal keys (is_internal flag or
   api_tier==='internal' or equivalent). Do not make custom mean unlimited.
4. Do not delete the callers. Do not mock the function.

Use `run` until it prints PASS, then finish.
