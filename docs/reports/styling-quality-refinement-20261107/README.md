# Styling refinement — November 7

Base: published CPU-bounded v3 `203dfbf25885f17e197b4d50afab6780f22ba5db`. Branch: `codex/styling-quality-refinement-20261107`. No production deployment or catalog/database writes.

## Changes and evidence

- Strict Formal: the shared garment/coherence path now rejects explicit short-sleeve bases using the same sleeve detector as the existing occasion gate. The occasion gate already rejected ordinary short-sleeve woven shirts; the defect was inconsistent coherence validity/scoring, not absence of every rejection. Unknown sleeve length remains unknown. Work/Casual sleeve policy is unchanged.
- Footwear: positive chunky/cup/platform sole evidence prevents a penny/dress label from promoting casual construction to formal. Plain rubber soles remain eligible. The imported Flextride row has only generic studio metadata, so a narrow model-specific evidence rule covers Geox U66MAC/Flextride Plus. [Manufacturer evidence](https://www.geox.com/en-GB/leather_loafers-black-flextride_plus_man-U66MAC0009BC9999.html) confirms cushioning and a synthetic/rubber outsole; calling this less suitable for refined tailoring is our styling judgment. No blanket Geox ban.
- Old Money Work: soft penalties for leisure/utility construction, technical zip/shell layers, sport/bold striped tops and casual-soled shoes. Normal dress pinstripes and performance wool tailoring remain eligible. Multiple casual cues lower whole-look refinement as well as individual garment rank.
- Fit/fabric/color: reward compatible tailoring volume and documented fabric families; softly penalize slim/wide tailoring contrast and competing patterned office uppers. A stale `solid` field no longer erases positive striped-title evidence. Unknown measurements, fabric and colors receive no invented details.
- Value: without an explicit budget, lower verified prices break equal quality/recency ranking ties within the same slot and currency. The provider receives the same preference. Owned, unknown-price and mixed-currency items are not treated as cheap comparisons. No budget ceiling or exchange rate is invented. A full-handler integration check verifies cheaper equal-quality products appear earlier in the actual provider prompt without creating a budget reference.

Fixture comparisons: the Flextride office look scores 0.895 → 0.801; technical leisure plus sport stripes 0.830 → 0.738; matching Formal tailoring 0.875 → 0.932. These are deterministic ranking scores, not user satisfaction or real generated-outfit measurements.

## Catalog investigation

Read-only audit of 40,989 checked-in snapshot rows found 170 title/category disagreements, not 170 confirmed live database defects. Examples include `selected-homme-16101349-bracken-1264359`, a short-sleeve shirt labeled bottoms. Existing title-based slot/structure guards reject the sampled misfiles; no second category engine was introduced. Two Flextride image-alias groups also contain synthetic color-name variants sharing the same product photo. Existing physical-product identity/recency guards remain authoritative. Examples and product references are in `verification.json`.

## Focused verification and limits

- 71 focused tests passed: 51 domain/CPU regressions, 11 selected hosted-handler cases, 8 budget/auth/source cases, and the new no-budget prompt-order integration case. No full suite.
- All nine existing Work/Formal CPU replays completed with three outfits and HTTP 200; 51.670–130.644 ms process CPU under the 500 ms ceiling. No network or production requests. Transpilation is excluded.
- `outfitDiversity.ts`, `outfitRotation.ts`, `validatedQuartets.ts` and the entire post-provider handler suffix are byte-for-byte unchanged from the v3 base. Hashes are recorded in `verification.json`. The Edge changes are the value tie-break and a prompt sentence; authentication/admission and explicit-budget rules are unchanged.
- These are metadata-based refinements. Sparse descriptions can still hide sleeve, sole, fit or fabric details. Soft Work penalties preserve scarce inventory, user anchors and recency priority, so they do not guarantee a dress-only deck. Same fabric/color evidence does not prove two pieces are a matching suit, and inferred fit is not a body-measurement guarantee.
- Value preference is relative, not an affordability promise: expensive inventories and unknown/mixed currencies remain possible without a user budget. The snapshot audit does not establish current live-row status. No new simulator visual signoff or production styling-quality claim is made.

Original v3 artifacts remain unchanged. Deploying these styling changes would require a separately built artifact and explicit authorization.
