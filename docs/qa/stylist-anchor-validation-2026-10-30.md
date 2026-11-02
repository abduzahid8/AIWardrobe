# Personal Stylist anchor validation — 30 October 2026

## Result

The Bexford Oxford failure is fixed in active `generate-outfits` version 179. The name-only slot repair interpreted “Oxfords” as a shirt, and the server’s separate identity validator repeated that mistake. Both now share the Oxford-footwear distinction. Explicit Oxford shirts remain tops.

The broad simulator test also exposed a later ranking filter that still applied the piqué-polo work restriction to Date night. It reduced a valid date inventory to two tops and allowed forced top repetition. The date-specific ranking conflict is fixed, with a hosted regression requiring three different compatible piqué polos. The work restriction remains in place.

A replay with the existing 12-deck history then exposed a misfiled “Slim-Straight Leg Metallic Coated Jeans” row counted as outerwear because “coated” matched “coat.” Word-boundary repair and name-first catalog checks now reject it before per-slot counts/page completion. An additional hosted test starts with 100 such misfiled rows and requires the search to reach real layers. The anchored proposal reserve remains bounded and now covers the complete standard six-by-six-by-six companion pool before its final shortlist is trimmed.

All 20 live API scenarios below passed against final deployment 179 with rate-limit-aware retries. Sixteen successful scenarios returned 48 outfits / 192 clothing selections. Four negative scenarios returned the expected rejection with no successful outfits. This validates the listed inventory and requests; it is not a claim that every possible wardrobe, occasion or weather combination can produce an outfit.

## Checks applied to every successful live generation

- Exactly three cards, with exactly one outerwear item, top, bottom and pair of shoes per card.
- The requested owned anchor appears exactly once in every card.
- All non-anchor companions come from the live shop catalog and have image references.
- No non-anchor product repeats within the three-card deck, using ID, normalized name and image identity.
- Returned quality assessments contain no rejection reasons.
- Three identical locked-Oxford requests retain the same shoes and rotate all companion products.
- Old Money → Minimalist → Classic requests share shoe history and use nine different physical pairs.

## Live API matrix

| Scenario | Anchor | Style / occasion | Expected / actual HTTP | Result |
|---|---|---|---|---|
| oxford-date-repeat-1 | Bexford Leather Oxfords - Black | old_money / date | 200 / 200 | PASS |
| oxford-date-repeat-2 | Bexford Leather Oxfords - Black | old_money / date | 200 / 200 | PASS |
| oxford-date-repeat-3 | Bexford Leather Oxfords - Black | old_money / date | 200 / 200 | PASS |
| oxford-work | Bexford Leather Oxfords - Black | old_money / work | 200 / 200 | PASS |
| oxford-classic | Bexford Leather Oxfords - Black | classic / date | 200 / 200 | PASS |
| oxford-minimalist | Bexford Leather Oxfords - Black | minimalist / date | 200 / 200 | PASS |
| oxford-casual | Bexford Leather Oxfords - Black | casual / date | 200 / 200 | PASS |
| polo-work-dresscode | Polo Piquet 1 | old_money / work | 422 / 422 | PASS |
| polo-date | Polo Piquet 1 | old_money / date | 200 / 200 | PASS |
| chino-date | Брюки Harmont & Blaine Chino's Sartoriale | old_money / date | 200 / 200 | PASS |
| chino-minimalist | Брюки Harmont & Blaine Chino's Sartoriale | minimalist / date | 200 / 200 | PASS |
| chino-work | Брюки Harmont & Blaine Chino's Sartoriale | old_money / work | 200 / 200 | PASS |
| performance-shirt-work | Regular Fit Striped Th Performance Shirt | old_money / work | 200 / 200 | PASS |
| mesh-polo-date | The Iconic Classic Fit Mesh Polo Shirt – White/Navy | old_money / date | 200 / 200 | PASS |
| cross-style-old_money | No anchor | old_money / date | 200 / 200 | PASS |
| cross-style-minimalist | No anchor | minimalist / date | 200 / 200 | PASS |
| cross-style-classic | No anchor | classic / date | 200 / 200 | PASS |
| missing-owned-anchor | Nonexistent owned item | old_money / date | 422 / 422 | PASS |
| unauthenticated | No anchor | old_money / date | 401 / 401 | PASS |
| wardrobe-only-no-layer | No anchor | old_money / date | 422 / 422 | PASS |

The live matrix uses the current catalog, explicit style/occasion choices and no optional budget or weather override. Budget, currency, weather and request-boundary cases are covered by the automated suites below.

**Additional saved-history replay: PASS.** The account’s existing 12-deck history and actual style preferences were sent with the Oxford anchor. The final server returned three complete, quality-approved outfits with distinct companions; the same input previously returned `incomplete_outfit_deck`. Evidence: `.maestro/artifacts/anchor-matrix/extended-history-replay.json`.

## Expected limits and rate protection

- **Piqué polo + Work:** rejected by the current work dress code; the same polo succeeds for Date night.
- **Wardrobe only:** this account has no owned outerwear, so a four-slot outfit cannot be completed without enabling New clothing.
- **Missing owned anchor:** returns `anchor_not_found`; the server does not substitute an unrelated product.
- **Unauthenticated request:** returns HTTP 401 and no outfits.
- **Request burst:** the initial run hit the existing 10-per-minute protection (HTTP 429) on its final three cases. The final run used cooldown-aware retries and passed all 20 original assertions. The limit was not disabled or bypassed.

## Automated regression coverage

**398 tests passed across 16 targeted suites.** Coverage includes Oxford shoes versus shirts, client sanitization and slot repair, hosted four-slot and anchor generation, all four anchor slots, repeated/cross-style rotation, physical aliases, sparse inventories, styling/occasion/weather constraints, budgets and currency handling, request bounds, authentication and rate limiting. TypeScript (`tsc --noEmit`), the deployed Edge entrypoint (`deno check`) and `git diff --check` passed.

Commands used:

```sh
./node_modules/.bin/jest --runInBand --silent \
  __tests__/utils/oxfordAnchorClassification.test.ts \
  __tests__/utils/occasionRulesRussianShoes.test.ts \
  __tests__/e2e/aiOccasion.e2e.test.ts \
  __tests__/features/normalizeSlots.test.ts \
  __tests__/features/sanitizeGeneratedOutfit.test.ts \
  __tests__/services/recentOutfitIds.test.ts \
  __tests__/services/shoeRotation.test.ts \
  __tests__/services/outfitChoices.test.ts \
  __tests__/utils/wardrobePoloEvidence.test.ts \
  __tests__/backend/fourSlotHosted.test.js \
  __tests__/services/stylistStrictQuality.test.ts \
  __tests__/services/stylistCanonicalRequest.test.ts \
  __tests__/backend/outfitBudgetWire.test.js \
  __tests__/backend/outfitGenerationRequest.test.js \
  __tests__/backend/rateLimit.test.js \
  __tests__/services/fourSlotRequirement.test.ts
./node_modules/.bin/tsc --noEmit
npx --yes deno check --no-config --node-modules-dir=none \
  --lock=backend/supabase/functions/deno.lock --frozen \
  backend/supabase/functions/generate-outfits/index.ts
```

## Manual checks to repeat

1. Open the owned **Bexford Leather Oxfords – Black** and choose **New clothing → Old Money → Date night**. Generate: expect three complete cards; Oxfords remain in SHOES on every card.
2. Return to setup and generate twice more with the same choices: shoes stay locked; the companion tops, layers and trousers change while valid alternatives exist.
3. Switch to **Minimalist** and generate: the same owned shoe stays locked, with three complete outfits.
4. Start an unanchored shop request, then switch **Old Money → Minimalist → Classic**: expect footwear to rotate across selections, as long as compatible alternatives exist.
5. Test **From wardrobe** without an owned layer: expect an explicit rejection, not a fabricated layer or a successful three-item outfit.
6. Test **Polo Piquet 1** for Date night and Work: Date night succeeds; the current work dress code rejects the piqué polo.

## Simulator verification

**PASS — final run against version 179**, on `AIWardrobe-Payments-20261027` / iOS 26.5 in Expo Go, with New clothing explicitly selected.

| App action | Visible result |
|---|---|
| Owned Bexford Oxfords → Old Money / Date night → Generate | Three complete four-slot cards; Oxford shoes retained |
| Generate again with the same choices | Three complete cards; all nine companion products change |
| Switch to Minimalist / Date night → Generate | Three complete cards; Oxford shoes retained |

Persisted displayed-deck identities were checked independently of the automation assertions. Each deck contains exactly 10 distinct products: one shared owned shoe plus nine different companions, filling 12 selections. The two same-intent generations share exactly the anchor’s three identity keys (ID / normalized name / image), and no companion identity keys. Evidence: `.maestro/artifacts/anchor-matrix/native-history.json`.

Screenshot evidence: [Old Money third card](/Users/v.zahid/AIWardrobe/.maestro/artifacts/anchor-matrix/native-v179/2026-10-30_200734/native/takeScreenshot/oxford-third-card.png) · [Minimalist third card](/Users/v.zahid/AIWardrobe/.maestro/artifacts/anchor-matrix/native-v179/2026-10-30_200734/native/takeScreenshot/oxford-minimalist-result.png).

The initial automation entered wardrobe-only mode and correctly could not fill the missing owned layer. A later extended-history check exposed the piqué ranking and coated-jeans catalog defects described above; those were fixed and the full simulator flow rerun successfully.

An Expo Go Fast Refresh startup failed to load `ExpoAsset` during setup; a clean app relaunch restored the runtime. The final acceptance run uses that clean session. These simulator checks do not replace a production native-build test.

Raw local API evidence: `.maestro/artifacts/anchor-matrix/results.json`; per-case response files and the rate-limited initial results are in the same ignored artifact directory. No authentication tokens are included in this report or response artifacts.
