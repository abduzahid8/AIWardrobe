# Stylist 6 — New Clothing editorial remediation (2026-11-12)

## Scope and release boundary

Branch: `codex/stylist6-visual-first-20261110`, PR #15. Review-only staging changes.
**Do not merge or promote to production** without owner acceptance, reproducible QA,
full release checks and exact deployment SHA verification.

Owner screenshots (16:41–16:42, 2026-11-12) showed the prior
`stylist-6-cloud-staging-1` build, with `NVIDIA_NOT_RUN`.
That engine string alone cannot identify the new editorial-copy commit.
The Vercel READY Preview pinned before this remediation also predates it.

## Observed failures and code-level repairs

| Symptom | Grounded change | Gate |
|---|---|---|
| Three cards titled "Refined Tailoring" | `deck_editorial_copy.mjs` gives unique product-evidence-led titles, preserving the native concept ID | `deck_editorial_copy.test.mjs`, `fashion_deck.test.mjs` |
| Generic claims, no explanation of changes | Each direction now explains observable palette, silhouette, or layering differences from the lead without body, weather or stock inventions | `deck_editorial_copy.test.mjs` |
| Technical leisure jacket in Classic/Old Money smart looks | Narrow catalog-only guard for explicitly sporty leisure outerwear in work/date contexts; ordinary blazers, casual contexts and explicit user intent survive | `fashion_deck.test.mjs` |
| Identical shoes reused across cards | Bounded soft preference for distinct physical shoe identity, using canonical overlap; no hard three-shoes requirement | `creative_directions.test.mjs` |
| "Total unavailable" hides all prices when currencies mix | Show exact subtotal for each listed currency; still refuse a fake exchange rate or single total | `mixedCurrencySpend.test.ts` |
| "Discovery" is just novelty | Already-existing conservative editorial quality floor remains in place; if no worthy third direction exists, do not fabricate one | `editorial_quality.test.mjs`, `creative_directions.test.mjs` |

## Product semantics

- The initial style quiz remains sufficient. No new mandatory questionnaire.
- Signature: highest-quality accepted look for selected style/occasion.
- Refresh: a distinct interpretation within measured quality constraints.
- Discovery: measurable structural or palette difference that **also**
  passes the stricter quality floor.
- A third look labelled `alternative` is honest when Discovery is unavailable.
- Outfit visual judge stays opt-in; `NVIDIA_NOT_RUN` means no NVIDIA review
  was performed. The local editorial score is deterministic, not AI vision.
- SKU provenance, ownership, image evidence, hard budget, locked items,
  style/occasion and auth/quota boundaries must never be bypassed.
- Budgeted requests continue to enforce strict same-currency pricing.
  Without a budget, do not invent an exchange rate, available stock or delivery.
- Changes to the editorial scorer stay catalog-only in this phase.
- Shoe variety is a soft tie-break: one great pair may correctly recur when
  the catalog has no other adequate shoes or an anchor is locked.

## Owner acceptance matrix

1. **Classic / Work / New Clothing**, unknown weather: three quality-valid
   options when inventory supports them, distinct titles and specific reasons;
   no explicitly technical leisure jacket in a quiet smart outfit.
2. **Old Money / Date night / New Clothing**: refined formality;
   avoid sports-training sneakers, resort tees, and weak novelty cards.
3. **Minimalist / Date night**: do not prescribe conspicuous athletic
   or inappropriate footwear merely to create visual difference.
4. **All three directions**: compare actual SKUs, product links and photos.
   Shoe reuse should decrease when equivalently suitable alternatives exist;
   a pinned shoe must remain if specifically required.
5. **Mixed currencies**: exact per-currency subtotals; no unverified converted
   single total. Single-currency or confirmed-budget-FX totals still work.
6. **Sparse catalog and wardrobe/mixed modes**: no invented stock, products,
   body details or additional onboarding; truthful partial results are allowed.
7. **Staging identity**: verify commit-specific Preview URL and
   `diagnostics.deckCopyVersion = s6-grounded-deck-copy-v3` before interpreting
   owner screenshots as evidence of the new algorithm.
8. **Security**: no client inventory trust, no secrets, no unauthenticated
   quota use, no forced NVIDIA evaluation, no production migration.

## Known release gates

- Existing global Jest suites have historical unrelated failures in Stylist 5
  hosted/quality-floor tests; **do not silently suppress or rewrite** these tests
  to claim all CI green.
- Dedicated Stylist 6 Node tests, TypeScript, ESLint, and migration/RLS
  gates must pass on the final SHA. Verify exact run before release.
- Vercel previously rejected a preview with HTTP 402 deployment quota;
  do not bypass this limit via production or a substitute project.
- A new ready Preview must be built from this final SHA and explicitly
  pinned on the isolated Supabase staging function **after** verification.
- Owner simulator screenshots, merchant availability, image-based taste
  judgments and real-world styling acceptance remain pending.

## Expected reviewer report

Provide exact GitHub commit, test run URLs, staging function version,
verified Vercel deployment SHA, 3x scenario screenshots, and item-level notes
about any underdressed/overpriced/repeated selections. **Do not claim
production readiness from deterministic unit tests alone.**

## Owner second screenshot round — 22:32–22:34 (+0500)

**Observed origin:** `Client 5d2d1950c510+dirty`, engine
`stylist-6-dev-local-20261111-r2`, `NVIDIA_NOT_RUN`.
These are local bridge screenshots, not evidence of cloud staging v16 changes.

### Reproductions

- Classic / Work: cards 1 and 3 both display “Refined Tailoring” even
  with different items. Root cause: React Native
  `outfitConceptTitle(nativeConceptId, suppliedEditorialTitle)` always
  retranslated the native concept ID, masking the unique server title.
- Classic / Work: first two looks share blue suede lug-style loafers.
  Footwear rotation is deliberately a *soft* preference; improvement
  remains subject to available better-quality, distinct shoes and visual
  sole evidence.
- Semi-Classic / Work: an oversized shirt, blue trousers and chunky
  comfort-sole footwear are questionable for polished office use.
- Old Money / Date: the first screenshot pairs a white dress-style shirt
  with green jeans and chunky slip-ons — a register mismatch.
- Old Money / Date setup: the comment still reads `work, classic` after
  the user selected Old Money and Date chips. Even when chip authority
  wins, old comment tags may contaminate ranking and styling explanations.
- An unlayered short-sleeve knit polo look is titled “Smart knit layering”,
  without an actual outer layer.
- Mobile mixed-currency breakdown has a cramped two-column layout. Sums
  themselves are correct exact original-currency subtotals, not FX totals.

### Remediation committed for next test

1. Preserve the *S6-specific* server concept title at render time; S5
   retains its localized legacy concept translation.
2. When a chip selection changes, discard only standalone conflicting
   style/occasion tags (e.g. `work, classic`) and preserve other free
   constraints (e.g. `avoid neon`, budget). The generator sanitizes the
   effective prompt as a second safety net.
3. In catalog-only smart refined looks, exclude evidenced casual/chunky
   shoe soles, and exclude unsolicited jeans in Old Money smart dates.
   Unknown soles stay unknown; Casual, Travel, wardrobe and explicitly
   requested styling are unaffected.
4. Explain actual role, volume, texture or palette changes from the lead
   rather than claiming an unspecified silhouette change. Do not claim
   a mix is appropriate without evidence.
5. When there is no outerwear, replace native “layering” headlines with
   real garment roles from the displayed outfit.
6. Display mixed-currency totals vertically on mobile.
7. Add Jest and Node regression tests for titles, chip transitions, dress
   register, grounded descriptions, unlayered headings.

**Release restriction:** The local bridge must restart to load this
GitHub revision; the client must rebuild/reload to see updated title and
comment behavior. The Supabase v16 staging proxy has **not** been repointed
by these commits. Do not attribute an unchanged local engine-version
string to a fresh code deployment. No NVIDIA inference was run.

Follow-up: inspect exact CI results on final SHA and run independent
visual/merchant feasibility review before production.

Version marker for new local proof: `stylist-6-dev-local-20261112-r3`; an R2 banner after pulling means the dev bridge process was not restarted.
