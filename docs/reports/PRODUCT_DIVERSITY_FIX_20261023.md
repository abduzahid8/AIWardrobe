# Product-Uniqueness Fix — Personal Stylist Deck Diversity

**Date:** 2026-10-23
**Device:** iPhone Simulator `AIWardrobe-Login-Audit-20261020` (`A803D6CC-0840-40F3-95EF-70546B2E0ED0`), iOS 26.5
**App:** `com.aiwardrobe` 1.4.3 (56), Debug build + Metro (`localhost:8082`)
**Runner:** Maestro 2.8.0, `.maestro/run_personal_stylist_20_sim.sh`
**Fix validated:** no non-anchor product repeats across the 3 looks when valid alternatives exist.

This report validates the fix for the bug found in
`REAL_SIMULATOR_DIVERSITY_20261023.md`: decks repeated the same actual
products (same SKU / same name / same image) across all three cards.

## Rule

- A **product** is identified by catalog id, normalized catalog name, or
  normalized image URL. Garment family is **not** identity: a black loafer
  and a brown loafer are two products.
- **Invariant:** no non-anchor product may appear in more than one look.
- Same archetype (e.g. three `blazer + top + trousers + loafers` looks) and
  same garment family are explicitly **allowed** when every product differs.
- The anchor/pinned item may repeat by design; `anchorRepetitionCount`
  tracks any accidental extra repetition.
- Uniqueness relaxes only for genuinely thin inventory
  (`insufficient_unique_products`); otherwise the rule never relaxes.

## Gates

| Gate | Result |
|---|---|
| TypeScript (`npx tsc --noEmit`) | pass |
| Jest full suite | **184 suites / 2353 tests pass** |
| Deterministic 20-scenario stylist simulation | **20/20 STRICT**, `nonAnchorProductReuseCount=0` everywhere, worst pairwise similarity **0.40** (soft signal, threshold 0.65) |
| Real simulator 20-scenario suite | **16/20 pass**; all 4 failures are environment/catalog gaps (below), none diversity-related |
| Product telemetry (`[stylist-product-diversity]`) | **16/16 captured decks**: `nonAnchorProductReuseCount=0`, `repeatedCanonicalKeys=[]`, `relaxReason=null` |

## Real simulator results

Every passing scenario produced a 3-look deck where each of the 12 displayed
products is distinct (id + normalized name + normalized image).

| # | Scenario | Result | Product uniqueness |
|---|---|---|---|
| 01 | work, old money | PASS | `reuse=0`, 3 looks / 12 unique products |
| 02 | work, classic, under $250 | **FAIL** | FX provider: "reference exchange rates unavailable" on 6/6 attempts across two runs |
| 03 | job interview, business casual | PASS | `reuse=0` |
| 04 | corporate conference, semi-classic | PASS | `reuse=0` (passed on retry) |
| 05 | date night, old money | PASS | `reuse=0` (passed on retry) |
| 06 | romantic dinner, minimalist | PASS | `reuse=0` |
| 07 | date night, casual | PASS | `reuse=0` (passed on retry) |
| 08 | weekend brunch, classic | PASS | `reuse=0` |
| 09 | everyday casual, minimalist | PASS | `reuse=0` |
| 10 | formal wedding, old money | PASS | `reuse=0` |
| 11 | black tie wedding, classic | **FAIL** | edge `503 ai_generation_unavailable` 6/6 attempts (known provider gap) |
| 12 | formal gala, semi-classic | PASS | `reuse=0` (passed on retry) |
| 13 | gym workout, casual | **FAIL** | edge `422 no_matching_clothing` — no sport catalog (known) |
| 14 | sport run, minimalist | **FAIL** | edge `422 no_matching_clothing` — no sport catalog (known) |
| 15 | travel airport, classic | PASS | `reuse=0` |
| 16 | flight travel, semi-classic | PASS | `reuse=0` |
| 17 | quiet luxury weekend, old money | PASS | `reuse=0` |
| 18 | modern professional office | PASS | `reuse=0` |
| 19 | офис, олд мани | PASS | `reuse=0` |
| 20 | sayohat uchun, oddiy | PASS | `reuse=0` |

All four failures are the previously documented environment/catalog gaps
(`02` FX rates provider, `11` AI provider 503, `13`/`14` no sport catalog).
None is caused by the diversity change — the same scenarios failed before
the fix for the same reasons.

### Representative captured deck (scenario 10 — formal wedding, old money)

```
Look 1: linen shirt | striped cool-wool blazer | striped wide-leg trousers | leather dress shoes
Look 2: wool blazer | cotton crew-neck sweater | wool pleated trousers | rubber-sole loafers
Look 3: cool-wool blazer | check shirt | cool-wool total-look pants | desert boots
```

12 products, 0 repeats, `relaxReason=null`, `supportsUniqueProducts=true`.
The archetype repeats (tailored formulas) — which is allowed — while every
actual product rotates.

## What changed

- `backend/supabase/functions/_shared/outfitDiversity.ts`
  - robust `imageKey` (strips query/fragment/protocol/trailing slash);
  - `ProductIdentity` / `productIdentityOf` / `productIdentityKeys` /
    `canonicalProductKey` / `countDistinctProducts`;
  - `analyzeDeckProductDiversity` with union-find clustering over
    id + name + image keys, anchor-aware;
  - inventory assessment (`uniqueProductsPerSlot`,
    `maxDistinctProducts`, `supportsUniqueProducts`);
  - rewritten `assembleDiverseDeck`: hard product-uniqueness gate
    (`maxSharedProducts=0`), soft archetype/similarity scoring only,
    combination search + fill rotation, relaxation gated on inventory.
- `features/outfit-generator/utils/normalizeSlots.ts` — deck freshness by
  product identity; family demoted to soft ordering; recency fallback
  prefers a recently-shown product over repeating one inside the deck.
- `features/outfit-generator/utils/deckDedupe.ts` — legacy `assembleDeck`
  default aligned to `maxShared: 0`.
- `screens/AIOutfitmaker.tsx`, `src/services/outfitGenerationService.ts`,
  `backend/supabase/functions/generate-outfits/index.ts` — product-key
  plumbing, deck analysis, and always-on dev telemetry:
  `console.log('[stylist-product-diversity]', …)`.
- `scripts/stylist-diversity-report.ts` — reports product keys, reuse,
  anchor repetition, uniqueness support; pass requires `reuse=0`.

## Evidence

Screenshots are committed under `.maestro/verify-shots/sim20_product/`
(`sim20_XX.png` + `_c2/_c3/_c4` scrolls per scenario, `_fail.png` for
failures). Maestro run logs: `run.log` and the two `run_*.log` retries.
Product telemetry: `.expo/dev/logs/start.log`, tag
`[stylist-product-diversity]` (25 deck records across the audit runs).

Key cards:

| Evidence | Path |
|---|---|
| 10 — 12 unique products across 3 looks | `.maestro/verify-shots/sim20_product/10/sim20_10.png` … `_c4.png` |
| 18 — formerly relaxed blazer deck, now unique products | `.maestro/verify-shots/sim20_product/18/sim20_18.png` … `_c4.png` |
| 05 — formerly near-duplicate knitwear deck, now unique | `.maestro/verify-shots/sim20_product/05/sim20_05.png` … `_c4.png` |
| 02 — FX rates failure | `.maestro/verify-shots/sim20_product/02/sim20_02_fail.png` |
| 11 — black-tie 503 | `.maestro/verify-shots/sim20_product/11/sim20_11_fail.png` |
| 13 / 14 — sport 422 | `.maestro/verify-shots/sim20_product/13/sim20_13_fail.png`, `.../14/sim20_14_fail.png` |
| Deterministic report output | `/tmp/stylist-diversity-report-final.md` |

## Reproduction

```bash
npx tsc --noEmit
npx jest --runInBand
npx tsx scripts/stylist-diversity-report.ts

# real simulator (Metro on :8082, debug app installed)
xcrun simctl launch A803D6CC-0840-40F3-95EF-70546B2E0ED0 com.aiwardrobe
OUT=.maestro/verify-shots/sim20_product ./.maestro/run_personal_stylist_20_sim.sh

# product telemetry
grep stylist-product-diversity .expo/dev/logs/start.log
```

## Follow-up

- `02` FX and `11` black-tie are provider-side; retry when the providers
  recover (both passed in earlier audits).
- `13`/`14` need a sport catalog in the shop feed.

## Hardening pass (same day, follow-up)

The first fix removed the greedy cross-look filters but left four structural
defects that a live simulator run reproduced. All four are fixed and verified.

### Defects → fixes

| Defect (reproduced) | Fix |
|---|---|
| The Selector could show **two identical cards**. Live run (scenario 01, shop): card 1 and card 3 carried the same four products (trench coat + sweater + wool pants + denim shoes, identical prices) because every slot was thin, so every repeat was "allowed". | `assembleDiverseDeck` now rejects a pair whose items match in **every** position (same photo or same row id, matched one-to-one). A thin inventory yields a **smaller** deck, never the same card twice. |
| Capacity was judged against the **candidate batch** (`limit: 5`) instead of the **displayed deck** (3 cards), and against **inactive** slots. | The client sends the additive `deckTarget` field (validated in `outfitGenerationRequest.ts`); the edge assesses capacity per **active composition slot** (`layered ? top/outerwear/bottom/shoes : top/bottom/shoes`) and reports the verdict back. |
| The client re-derived "is this slot thin?" from a **truncated pool**, so it could reject a repeat the edge had legitimately forced (and vice versa). | `assembleDiverseDeck` accepted a `capacity` override; the edge returns `diversity: {activeSlots, strictSlots, forcedRepeatSlots}` and the client applies it (unnamed slots keep the local verdict). |
| Provider failure returned a hard `503` even though a deterministic builder existed and was never called. A **second** cross-look product rule also lived in `localFallback`, and its "relaxed pass" reached the target by repeating products. | `localFallback` is now wired into the provider-failure path, gated through the SAME `filterValidOutfits` + `assembleDiverseDeck`. Its own cross-look rule and relaxed pass are gone. If nothing validates, the client still gets the retryable `503 ai_generation_unavailable`. Curated mode keeps failing closed. |

Also removed: the unreachable Gemini provider (dead `callGemini`, key and
model loading), the prompt's contradicting "must differ by ≥2 item ids"
diversity rule, and the ASCII-only product-name normalization (Cyrillic
names collapsed to an empty identity). NVIDIA is the single provider, with
one bounded retry for transient failures (429/5xx/network) that does **not**
consume the format/quality correction budget.

### Verification

| Gate | Result |
|---|---|
| `npx tsc --noEmit` | pass |
| `npx jest` | 186 suites · 2379 tests pass |
| `npx tsx scripts/stylist-diversity-report.ts` | 20/20 STRICT, reuse 0, worst similarity 0.40 |
| Simulator, live edge v120, scenario 01 (shop, old money) | 3 cards; `illegalProductReuseCount=0`; one **forced** repeat (`n:100 cool wool suit blazer`, thin outerwear slot); `deckSize=3`, `deckTarget=3`; `edgeCapacity` present in telemetry (the edge's verdict reached the client) |
| Lint | unchanged (1 pre-existing error, no new errors) |

New regression suites:

- `__tests__/features/stylistDeckSelection.test.ts` — greedy trap, same
  formula, duplicate rows, Cyrillic identity, irrelevant outerwear, anchor
  slot, thin-slot-only repeats, rich never relaxes, duplicate card, deck
  target, authoritative capacity override.
- `__tests__/backend/generateOutfitsFallback.test.js` — `filterValidOutfits`
  is independent per look (a look sharing a product is kept; an exact
  duplicate is not), the fallback no longer judges the deck, and the
  provider policy (no second provider, bounded retry, single routing
  comment).

## Composition-policy stabilization follow-up

The follow-up audit confirmed that the backend and client still had
independent layering decisions. The production contract is now centralized in
`backend/supabase/functions/_shared/outfitComposition.ts` and used by the Edge
Function, client style inference, local generation service, offline builder,
and client diversity selection:

- warm weather and missing weather: `top + bottom + shoes`;
- cold temperature/conditions: `outerwear + top + bottom + shoes`;
- explicit black-tie/tuxedo intent: layered, even if the weather is warm;
- explicit summer/no-jacket intent: non-layered unless formal intent is also
  present;
- style presets alone do not force a layer.

The client now uses the Edge-resolved active slots for capacity and diversity
selection. The request validator rejects `deckTarget > limit`, the server
keeps the requested deck target independent of the number of surviving AI
candidates, and quality telemetry reports the selector's actual
`hasQualitySignal` value. Missing `NVIDIA_API_KEY` is recorded as a provider
failure and proceeds through the existing deterministic fallback path.

Focused composition and request-validation tests were added. In this
environment, the baseline and final Jest/typecheck commands could not run
because `npm ci` could not retrieve the uncached `expo-modules-core` tarball;
the available JavaScript syntax checks and `git diff --check` passed. No
production deployment was performed.

Reproduce the headline case:

```bash
npx jest __tests__/features/stylistDeckSelection.test.ts
OUT=.maestro/verify-shots/sim_product_dup_fix ./.maestro/run_personal_stylist_20_sim.sh
grep stylist-product-diversity /tmp/metro8083.log | tail -1
```
