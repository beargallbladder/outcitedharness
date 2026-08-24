// lib/api/v1/with-auth.ts
if (slug !== null && !slugInWallet(user, slug)) {
  return 403
}

// lib/designwins/facade/auth.ts
if (aisle && !slugInWallet(user, aisle)) {
  return designWinsProblem(403, 'forbidden', 'Aisle is not in this wallet', {...})
}
