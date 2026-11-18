# AIWardrobe Stylist 6 — real-photo-FIRST source coverage, independent source-grounded art fallback

Date: 18 November 2026 · Commit tested on LIVE isolated Render staging: `468c7f21758c15eec46d1457e9018e1d49423949` · Parent `7bf4e2a8adaab4ebf70b779a509b803c0d8bd562` · PR #15, `codex/stylist6-visual-first-20261110`.

**Goal (UNCHANGED):** autonomous fashion stylist that chooses three outstanding visually different wearable four-SKU looks for the actual person's style and situation, with authenticated available real merchant inventory, price correctness, and zero unverifiable/photo-rejected displayed items. Real personal taste is not established by passing unit tests, passing one NVIDIA photo critic, or merely having 3 tabs. Production Stylist 5 must remain unchanged.

## What actual source checks show

Live Supabase `fyqpifmrsftsfqibhwhy` (read-only evidence): `shop_catalog` **8,554 active SKUs**. The separately licensed-photo table `catalog_photo_cache_v1` has **ZERO rows**, hence **ZERO `ready` source-hash-current grants** for cache hosting plus external model inference. This is a claim about records in the connected cache, NOT proof that legal third-party licenses do or do not exist elsewhere. No grant, rights basis, permissions or owner photo were fabricated, and no copyrighted merchant image was rehosted to bootstrap first-pass vision. Existing independent original-photo critic has its own separate controlled image-access path; this new first-artist license gate is stronger, and it remains CLOSED.

The server already had an auditable admin-only ingest `scripts/stylist6/ingest_licensed_product_photos.mjs` which requires a demonstrable SKU-specific legal basis, rights reference, valid grant/expiry, `hostingPermission` and `externalAiInferencePermission`, immutable source-product+image hash, private admin key, proof of current original active shoppable merchant SKU and downloaded JPEG byte checksum. It refuses to infer a photo license from a publicly accessible product thumbnail. The source-verified image board `backend/stylist6/visual_sku_board.mjs` already assembled four photo columns with opaque P-tokens, no merchant identity/source URLs in its picture, and fails CLOSED when too few verified images exist.

## Important architectural source-visibility defect fixed (not an outfit rule)

BEFORE `7bf4e2a`, the server could recognize 3+ granted photos per role somewhere in its discovered inventory yet the 48-token composer menu could choose NONE of those exact images, because its popularity/subtype source-prep shortlist ignored the very legal SKU IDs already verified. Turning on PHOTO-FIRST with good licenses still risked `insufficient_photo_inventory` or a silent fallback to text.

`7bf4e2a` now:

1. Counts only rights-valid original merchant source photos whose items remain eligible under the actual source/current-budget/FX request, never raw rejected/excluded products.
2. Passes a *server-owned*, source-hash-verified set of exact original catalog IDs from the licensed photo cache into the original inventory-first 48 P-token composer (never client flags).
3. Gives genuinely licensed photographed items first opportunity in each real garment role, including when outside an earlier 650-item ranker window, retaining the **same 12 per role and 48 overall**, without manufacturing new products or constraining AI combinations. The model retains freedom to style any permitted P-token, subject to the original full-four-SKU and photo critic gate.
4. With ZERO verified cache records, its source picking remains unchanged. The original licensed-media validator independently revalidates exact JPEG SHA bytes, rights and current merchant URL before the external original-photo AI artist can view them. Expired/missing/corrupt cached photos still ABSTAIN and never get promoted to a permission.

A real `sharp`-generated test JPEG fixture builds the exact four-column visual-contact-sheet with **3 independently licensed-source-designated P-tokens in every role** (12 photos), never actual copyrighted merchant images. All 48 tokens stay real source-eligible test SKUs and no IDs/URLs leak to the model. This validates the code's visual path, **NOT** live readiness with third-party image rights.

## Separate failure of autonomous artistic planning fixed

During real iOS Classic/Date on the immediately previous code (`7bf4e2a`), first 120B Creative Director returned `creativePlanStatus:invalid_plan`, zero accepted art directions, initial AI still composed 6 source SKUs but real photo critic approved **1/5** and displayed ONE outfit. The chosen color/occasion relation remained weak. Instead of writing deterministic garment pairing rules, `468c7f2` added **one bounded independent 30B source-grounded Director attempt** only when the first 120B concept output contains zero valid plans and sufficient staging budget remains. The second model receives the same truthful availability and photo-derived feature examples, person/occasion and output schema; it cannot declare products, prices or photo-approvals, and it does not run for a successfully planned first pass or outside isolated 72s staging.

Tests replay an invalid first 120B output and a valid second 30B full 4-garment-role plan; independently test that no backup or duplicate model call happens if the first artistic director succeeds. Safe server telemetry records whether a real backup was needed; never logs user identity/photos/inventory IDs. This is resilience against lost creative input, NOT automatic improvement of fashion aesthetics.

The later Kimi-FIRST attempt (`0565271`) was tested and **REVERTED** in `adc1e823` after a genuine full-source Date run where Kimi yielded no valid slate before timeout; Nemotron recovered, final remained **2 looks, 2/5 photo-approved**. Split concurrent models (`dccf0be`) and late photo budget patch (`34a8574`) remain REVERTED. No additional made-up fashion rule/ban was introduced to hide these failed tests.

## Final LIVE verified authenticated iOS quality (no licenses yet)

Render `dep-dbunt7lckfvc73cp76r0` at SHA `468c7f2` confirmed **LIVE**, signed authenticated 72s staging request from native iOS 26.5, catalog user context Classic/New, static server-owned discovery seed `stableSourceCohort:true` (a repeatable SAMPLE, not immutable merchant inventory), engine has `skuCompositionPhotoFirst:not_run` as expected because there are no legal cache rows:

| Real iOS user intention | Created source-true exact-SKU proposals | Original-photo approvals | Actually shown FOUR-real-SKU complete looks | Engine / delivery | First art direction |
| --- | ---: | ---: | ---: | --- | --- |
| Classic / Date night | 8 | **3/7** | **2** | 59.1s / 59.6s | `planned`, **3 concepts**, no backup needed |
| Classic / Work | 4 | **2/4** | **2** | 52.3s / 52.9s | `planned`, **3 concepts**, no backup needed |

All displayed physical goods are source-true, four item, no fake fourth, no rejected card. Date source SKU artist Nemotron+independent Kimi initial, original whole-look photo editorial applied; remaining portfolio still 2 since 3 independent good designs not confirmed. Work Nemotron-only plus photo editor. The first-artist fallback did not execute on these runs (`creativePlanBackup:not_needed`); its positive proof is in integration tests only.

**Owner's fashion taste goal remains NO-GO:** native Date card 01 pairs a light blue shirt with dark brown jacket/trousers/shoes, card 02 a pale knit with a brown coat/blazer and similar dark tailoring. Two cards can be attractive yet share too much style construction and lack a third striking direction; AI photo approval alone cannot certify exceptional romance or fashion personality. Work likewise produced only two approved suits, with no statistically established aesthetic uplift. Do not claim 3/3 or human reviewer acceptance.

Verification: **557/557 Stylist 6 automated tests**, **42/42 server/portable build tests** green at exact `468c7f2`; GitHub PR remains unmerged; Stylist 5 production untouched.

## Binding next real milestone (NO micro-rules)

- Obtain **documented per-source SKU photo hosting and external AI inference rights** from the actual photo owner/licensed provider. Minimum first-pilot rights portfolio is 3 suitable independently source-current SKUs per top, meaningful layer, bottoms and shoes (12), preferably broader visual shapes/palettes to empower genuine styling choice. Do not create self-signed merchant rights evidence or present `owner_upload` as ownership of somebody else's picture. Grant manifests and admin keys must remain private.
- Use existing admin `ingest_licensed_product_photos.mjs` dry-run, verify per-SKU match/expiry, then apply only grants verifiably possessed, on the admin environment. Enable BOTH `STYLIST6_PHOTO_FIRST_COMPOSER_ENABLED=true` and `STYLIST6_PHOTO_ASSETS_READY=source-hash-verified` **only after** the connected source cache shows ready photo evidence for all four roles, and after the code supports their current photo/source exactness. Test a real authorized 4-column board from live SKU bytes and evaluate *whole outfit aesthetics before SKU commitment*.
- Compare that genuine first-photo artist vs the existing image-feature-only pipeline over the same fixed source set; audit 3 fully source/disjoint four-item shoppable looks with original photo veto, and run at least **30 held-out S6-versus-S5 paired scenarios, 3 genuinely independent blind human reviewers, >=70% S6 preference**, zero physical/rights/budget/privacy failures. Existing image-model judgments alone are NOT this acceptance gate.

**No production promotion or `main` merge until the owner's actual quality goal is demonstrated.**
