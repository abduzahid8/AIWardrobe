# AIWardrobe Stylist 6 — stop accepting empty creative plans, expose real user taste feedback

**Verified checkpoint:** 18 November 2026. Staging creative-source code `3c8f325dea185745e1d84dd4196b2def0031a700`, Render `dep-dbupgdgae00c73f84g80` **LIVE**; separate React Native screen implementation `451d0074e5a2c2e622ac5658c216f80ece8dbed6` committed but **not verified in a fresh native binary**. All changes on the open PR #15 branch; production Stylist 5 unchanged. **QUALITY / RELEASE NO-GO**.

## Actual user goal: 3 aesthetically excellent four-real-SKU sets, not 3 output tabs

The owner requires an autonomous stylist with genuine taste, three BEAUTIFUL and sufficiently different human-centered four-physical-source-SKU, shoppable outfits. An LLM passing schema checks, an individual photo-critic approval, or 3 creative concept headings does **not** prove this. Do not pad a rejected third card or relax real source, budget, rights or photographic quality gates.

## Finding 1 — the initial artist could call a garment-category list a complete fashion artwork

The 120B Creative Director JSON schema required `artDirection`, but strict JSON validated the mere presence of a string and `normalizePlan` accepted `artDirection:''` as a complete creative plan. The source-grounded SKU artist could then receive three distinct plan titles/roles/palettes **with no specific garment-to-garment artistic intent**. The previous `creativePlanStatus:planned` log never proved meaningful design descriptions.

`3c8f325` fixes the actual **artistic output contract**, not a color/subtype/brand mini-rule. Any accepted plan must contain >=85 safe visible characters in `artDirection` *after* sanitizing links, email and markup. The existing NVIDIA prompt already asked for a **90–230 character** concrete explanation of proportions, construction, drape, textile and shoe relationship; the strict API schema remains backward compatible with provider JSON subset. Empty, generic 25-character placeholders and repeated URL-only text no longer count. Valid first concepts survive and the existing independent Creative Director can fill holes with genuine source-validated new art; no substitute catalog SKU or outfit invented. Staging trace now safely reports `creativePlanDetailedArtDirections` count (0–3), never the prose itself. All source admission, independent original-photo critic and 4-product contract remain unchanged.

**Tests**: 564/564 Stylist 6 Node tests + 42/42 gateway/portable build tests passed on the exact change. Regression tests exercise blank and shallow art plus redacted URLs, while preserving genuinely different material/drape directions with identical nominal garment role categories.

## REAL authenticated date-night original-photo result on exact LIVE `3c8f325`

Classic / New / Date night, real iOS 26.5 paid source-catalog generation, isolated Render 72s staging:

| Metric | Actual verified receipt |
| --- | --- |
| Initial Creative Director | `planned` |
| All three source-grounded concept headings | `creativePlanCount:3` |
| **Actual nonempty detailed artistic intentions** | `creativePlanDetailedArtDirections:3` |
| First real SKU compositions | 6 proposed, 5 admitted |
| **Independent original-photo approvals** | **1/5** |
| Customer-facing FOUR-real-product outfits | **1** |
| Server / wall clock | ~36.1s / ~36.7s |
| Photo rescue | `timeout` after ~4.9s, none accepted |
| Original-photo-first authoring | `not_run` (no external inference rights-verified cache rows) |

The human-inspected native result is `Textured tailoring`: a light-blue shirt, dark jacket, dark trousers, refined dark footwear. A plausible outfit, but not three distinct, impressive Date looks and no evidence of superior fashion judgement. This proves that **three full-length LLM artistic descriptions are necessary but NOT sufficient for real aesthetic excellence.** Model-driven original photograph criticism still rejects four of five.

## Finding 2 — no actual human fashion preference training signal

Read-only Supabase audit on 18 Nov: 8,554 active catalog items, 7,556 visual profiles, 7,525 image embeddings, and **ZERO rows in `public.stylist_look_feedback`**. There are 46 user-supplied manually captioned reference looks and separately 12 brand inspiration records / 50 associated products. Neither of those is an independently rated 30-scenario blind human QUALITY dataset or new legal external-photo inference authorization. The existing explicit feedback model (`feedback_taste.mjs`) can learn from likes/dislikes on complete outfits, but cannot learn from zero ratings. Do NOT invent positive votes or treat the NVIDIA original-photo critic as a human judge.

The 7 latest privately recovered actual source looks on 18 Nov were all **concept-id eligible** for feedback; this rules out the immediate theory that all feedback buttons are hidden due to invalid concept IDs. However the existing controls were below the four-item grid, pricing and tips (far below the first viewport), with no human ratings collected. That made meaningful preference accumulation unlikely even though the data path exists.

`451d007` moves the existing **explicit** thumbs-up/thumbs-down controls directly below the complete-look title/personally grounded evidence, above its 4-product grid. Adds discoverable `testID`s and selection state, keeps strict source provenance eligibility, a single pending write, real Supabase RLS authenticated owner row and no votes until the user expressly taps. No user attributes/photo contents/PII are logged or sent to analytics by this change. It does not label existing cards automatically; it unlocks a future user preference signal if people actually review.

**Client verification:** `__tests__/services/stylistFeedbackEligibility.test.ts` 9/9 passed; `npx tsc --noEmit` passed. **Native build/UI of this fresh screen is still unverified**; earlier iOS tests exercised the previous installed JS bundle. No TestFlight, App Store, production merge or new Supabase schema modification in this round.

## Binding NEXT big step — learning actual human visual preferences, not tiny rules

1. Obtain auditable source-specific permission to cache and externally infer on original garment JPEGs for at least 3 authentic live shoppable products in each of TOP/LAYER/BOTTOM/SHOES. The existing first-photo creative board code is ready with synthetic fixture tests, but the **live photo-license cache is zero**; there is no permission to invent.
2. Use the current feedback-on-outfit interface to accumulate genuine explicit positive/negative judgments about the *complete looks*, and independent 3-person blind aesthetic evaluation of >=30 held-out source/photo comparable scenarios. A score of >=70% stylist-6 over stylist-5 preference and three genuinely visually different four-SKU outfits are STILL REQUIRED before release. The screenshot-only internal fashion critic is not a substitute for these humans.
3. Once the original-photo-first artist is legally enabled and human observations are collected, calibrate complete-outfit and entire-portfolio choices to actual pairwise user preferences. Do not reintroduce failed split-inventory model races, arbitrary color forbiddance, makeup third garments or PHOTO threshold changes just to inflate counters.

**Release decision:** NO-GO for Stylist 6 production. Stylist 5 production remains intact.
