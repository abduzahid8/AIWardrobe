# BILLING.md (Phase 7CZ)

Subscription architecture. Code: `store/subscriptionStore.ts`,
`src/services/iapService.ts`, `src/services/growth/entitlements.ts`,
`__tests__/growth/*`.

## Tiers

`free | lite | premium | vip`. Existing tier matrix is wrapped, not
redefined. Free stays genuinely useful (wardrobe, basic outfits, limited
BuyCheck, manual planning).

## Lifecycle (never boolean-only isPremium)

`unknown → free | trial → premium_active`, with `grace_period`,
`billing_issue`, `expired`, `restoring`, `unavailable`. Unknown/offline and
billing-issue states degrade to free features; wardrobe and history are
always available. Downgrade preserves all user data; only expensive
regeneration is disabled.

## Boundaries

- Domain code asks the entitlement matrix; RevenueCat SDK stays in infra
  (`EntitlementProvider` interface, mock for tests).
- Server-authoritative verification for expensive actions
  (`AllowlistServerVerifier` pattern); client entitlement is display-only.
- No user data deletion on expiry; restore always reachable; trial terms
  explicit (start/end/conversion/manage).

## Activation dependency (remote, pending approval)

RevenueCat offerings/products, store pricing, sandbox billing QA scripts
(see `docs/growth/PHASE6_ACTIVATION.md` and `docs/REMOTE_ACTIVATION.md`).
