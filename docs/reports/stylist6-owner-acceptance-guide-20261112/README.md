# Stylist 6 cloud staging — owner acceptance testing guide

**Status before your test:** ENGINEERING READY (staging only). No successful
user-level cloud generation has been demonstrated. Production Stylist 5 is
untouched. Nothing is merged to main; nothing is promoted to production.

This guide is the complete manual test script. You perform every step
yourself in the real app. Engineering never submits authenticated generation
requests on your behalf and never consumes your quota.

---

## 0. Before you start (one-time, ~5 minutes)

1. On this Mac, in the repo root, make sure the dev gate is on — `.env`
   contains `EXPO_PUBLIC_STYLIST6_DEV_BRIDGE=true` (it does as of
   2026-11-12).
2. Start Metro (this is what makes the dev app run the **current** cloud
   implementation — the JS bundle is served live, the installed native
   binary stays as-is):
   ```
   npx expo start --ios
   ```
   If the app red-boxes on launch, report it; do not retry generation.
3. Sign in with the same test account used before (Premium, dedicated).
4. The engine picker is on the outfit-maker screen: select
   **Stylist 6 cloud staging** (`stylist6cloud`). Leave
   **NVIDIA visual review** unchecked for the first pass.

## 1. Quota snapshot (before any generation)

Note the daily `edge:aiOutfits` usage for the account (Profile → usage, or
ask engineering to read it). Expected before the first tap: unchanged from
your last reading. Every successful cloud generation debits exactly **one**
unit of the same production bucket; failures before admission debit nothing.

## 2. The matrix (one tap each — never re-tap the same logical request)

Tap Generate **once** per row and record the outcome. If a request fails,
do NOT retry the same request; change the input (or re-lock an item) to make
it a new logical request. A duplicate of an already-claimed request returns
HTTP 409 `duplicate_request` by design and consumes nothing.

| # | Source | Style / Occasion | Budget | Locked anchor | Expect |
|---|--------|------------------|--------|---------------|--------|
| 1 | New clothing (catalog) | Classic / Work | 25,000,000 UZS | none | 3 cards, UZS prices, spend ≤ cap |
| 2 | New clothing (catalog) | Old Money / Work | none | none | 3 cards |
| 3 | New clothing (catalog) | Minimalism / Work | none | none | 3 cards |
| 4 | New clothing (catalog) | Semi-Classic / Date night | none | none | 3 cards |
| 5 | From wardrobe | Classic / Work | none | none | 3 cards from your own items |
| 6 | New clothing + locked owned item | Classic / Work | none | lock any owned piece from My Closet | anchor appears in all 3 cards, rest is catalog |
| 7 | Your choice | any | small budget in EUR/USD | none | same-currency shop items only |

What to check on every card:

- **Images** are the real product photos (no placeholders, no wrong garment).
- **Prices and currency** match the merchant listing style (original listing
  currency on tiles; totals in the budget currency).
- **Buy** opens the real product URL (catalog mode only; owned pieces have
  no Buy button and no price).
- **Descriptions** reference the actual items on the card.
- **Locked anchor** (row 6) is present in every card.
- **NVIDIA status** (second pass, section 4): the deck reports truthful
  status — `NVIDIA_APPLIED`, `NVIDIA_EVALUATED_NO_CHANGE`, `NVIDIA_ABSTAINED`
  or `NVIDIA_NOT_RUN`. It must never claim provider judgment when the
  ranking was deterministic.

## 3. Recording results

For each row report: pass/fail, screenshots, the exact server error text
shown in the app (the cloud path now surfaces the real server code — e.g.
`anchor_not_found`, `rate_limited`, `feature_limit_reached`,
`stylist6_no_feasible_looks` — not a generic failure), and the post-test
quota delta. Engineering maps each code to a fix without re-running your
request.

## 4. NVIDIA pass (only after rows 1–7 are recorded)

1. Select **Stylist 6 cloud staging**, check **NVIDIA visual review**.
2. Run row 2 (Old Money / Work, no budget) once.
3. The response includes a visual receipt: provider call count, valid
   verdicts, abstentions, token usage. Reference-guided judging stays
   `NOT_RUN` until reference bytes are staged in the private
   `STYLIST6_REFERENCE_BUCKET` — that is honest behavior, not a failure.
   Retailer images are only sent to the provider when their host is on the
   committed allowlist.

## 5. What engineering already fixed in this round (context)

- Catalog discovery: the server boundary now restores the omitted
  `fashion_subtype` (and previously `is_active`) so cloud classification
  matches the verified pilot.
- The app no longer rewrites cloud decks with Stylist 5 taste gates.
- Locked owned anchors in catalog mode now send the mixed pool instead of
  failing.
- Budgeted catalog requests no longer fail on the FX-snapshot check.
- Real server error codes reach the screen.
- Post-claim planning failures are recorded so retries replay honestly.

## 6. Hard rules during testing

- One Generate tap per logical request. No automatic retries.
- Do not run the three-scenario smoke script.
- If the app shows a red box or a 5xx, capture it and stop that row.
- Rate limit is 10 requests/minute per account (cheap bucket) and the daily
  generation quota is the production one — shared with Stylist 5.

## 7. Rollback (engineering action, not yours)

Stop/delete only `aiw-stylist6-node-staging` or revert its staging pin.
Production `generate-outfits` (v209) and the S5 staging proxy (v6) are never
touched. Claim history stays; it is RLS-locked and inert.
