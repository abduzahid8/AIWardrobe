# Stylist 6 local bridge (development only)

Runs the REAL Stylist 6 fashion ranker (`recommendFashionDeck`) for the iOS
simulator app without deploying anything.

## Start

```sh
# Terminal 1 — ranker (leave running; plain Node, zero npm deps)
node scripts/stylist6/dev_bridge.mjs --port=3127

# Terminal 2 — app (Dev Client on the simulator)
EXPO_PUBLIC_STYLIST6_DEV_BRIDGE=true \
EXPO_PUBLIC_STYLIST_DIAGNOSTICS=true \
npx expo run:ios
```

`.env` already carries the two flags locally (gitignored). Release builds
ignore them: the engine switch requires `__DEV__` AND the opt-in flag.

## Test

1. Open the AI Stylist screen (New clothing / shop mode).
2. Pick `Stylist 6 (local)` in the `Engine (dev only)` section.
3. Set style + occasion, tap Generate outfit.
4. The results header shows `Engine stylist-6-dev-local-…`, each card shows
   the same version, and diagnostics show `visual NOT_RUN` (no NVIDIA).
5. If Stylist 6 fails, the real error is shown — never a Stylist 5 deck.

## Files

- `scripts/stylist6/dev_bridge.mjs` — loopback HTTP bridge (127.0.0.1 only).
  Adapts live `shop_catalog` rows with the real `catalog_adapter`, builds
  cached reference briefs from `benchmarks/stylist6/references/`, ranks with
  the real `recommendFashionDeck`. No visual critic is injected, so no
  network or NVIDIA is needed.
- `src/services/stylist6Dev.ts` — dev-only app client. Throws unless the
  dev gate is open; surfaces bridge failures with their Stylist 6 codes.
- `screens/AIOutfitmaker.tsx` — dev-only `Stylist 5 / Stylist 6 (local)`
  picker, exact-deck pipeline (rendering-contract acceptance, no
  re-diversification, no recovery fill), and the actual-engine banner +
  per-card version labels. Failures show a dev-only `code:` line with the
  real failure key.
- `__tests__/services/stylist6Dev.test.ts` — gate, identity,
  exact-deck acceptance and no-fallback coverage.

## Notes

- Exact-deck acceptance (`acceptStylist6DevDeck`) enforces only the
  rendering contract (3-4 slotted items, unique ids, https photos, shop
  provenance). The shared Stylist 5 taste gates (base-shirt-only tops,
  style-judge repair, closest-match recovery) are skipped for Stylist 6
  dev decks so the ranker's exact output — including fine-knit base
  layers — reaches the screen. Budgets still apply downstream.
- Live discovery rows omit `is_active` (pre-filtered server-side); the
  bridge defaults a missing key to active. Explicit `false` still filters.
- Budgeted shop runs need the deployed `budget-rates` function. If it is
  unreachable, both engines fail with `outfitMaker.budgetRatesUnavailable`
  before any ranking — an environment issue, not a Stylist 6 defect.

## Limits

- Shop/catalog inventory only. Wardrobe-owned pieces are out of scope: the
  ranker consumes catalog rows.
- One currency per request. With a budget set, the pool is filtered to the
  budget currency (same rule as the offline replay); without a budget,
  mixed-currency decks are totalled client-side with the pinned FX snapshot.
- No repeat-tap variety salt: identical re-taps may return the same top
  deck (the bridge owns ranking; history only penalizes recent signatures).
- The `client_rotation_overlap` audit is skipped for Stylist 6 dev decks.
