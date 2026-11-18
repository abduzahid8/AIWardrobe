# Stylist 6 — real four-SKU artistic intention survives into the PHOTO audition, still not 3 exceptional outfits

**Date:** 18 November 2026. **Checked staging code:** `43f41731b5d2319f9a73867d447206246e9ece40`, isolated Render service `srv-dbtd336i0phs73fd0sbg`, deploy `dep-dbupogek1f9s73b0ie50` **LIVE**. Branch `codex/stylist6-visual-first-20261110` / PR #15. **Owner acceptance NO-GO.** Production Stylist 5 untouched.

## Goal: autonomous human-caliber taste, NOT a technically valid set of three cards

The goal is **THREE visually outstanding and genuinely DIFFERENT personalized outfits**, each comprised of exactly FOUR original catalog/source-authenticated active purchasable SKUs and each independently PHOTO-approved. No palette/subtype bans, invented products, forced filler, altered original-image verdict floors or stolen hosting/inference permissions may turn a superficial card count into a false release claim.

## Substantive root cause and code repair

The 120B Creative Director planned three actual 85+ character editorial designs (material relationship, proportions, footwear, occasion). The first independent source-SKU creator then worked from these. But `sourceVerifyComposedOutfits` silently **threw away the original editorial briefs** and manufactured `ai_composed_N` with title `AI-styled complete look N` and *exactly the same roles and palette as the chosen items*. That makes its existing concept-fidelity quality signal tautological: a selected SKU combination is always identical to its own manufactured source brief, regardless of whether it realizes the creative artist's original point of view. It also destroys the per-plan ancestry shown to the photo-review audition, allowing a false claim of three independent artistic creations.

Commit `43f4173` preserves the **actual authenticated source-validated Creative Director brief** for each of the first three concept-commissioned source-SKU looks, including its ID (`ai_director_1/2/3`), real role vocabulary, palette, silhouette and safe model-written full-look artistic direction. The existing score's **small soft concept-fidelity adjustment** can now measure what the artist actually intended, instead of comparing clothes against a brief invented from those same clothes. The 4th–6th independent creative ideas and later photo-informed corrections remain *self-attributed* `ai_composed_N`, never falsely claimed as original director artwork. A plan missing a trusted version, exact original position or a meaningful source-validated art direction cannot acquire director provenance. Client/user messages cannot nominate their own fake creative plan, and the original source/budget/weather/retail and PHOTO jury gates remain unchanged.

Code tests replay six fully legitimate four-source-product compositions plus three source-grounded creative plans, verify all three real intent descriptions reach the scorer, verify the three independent alternatives are NOT mislabeled, enforce 24 nonrepeated source products and reject overbudget combinations. **565/565 S6 tests and 42/42 gateway/portable server tests passed on the exact change.** These are structural correctness tests, NOT aesthetic acceptance scores.

## Actual authenticated iOS visual results, native product photos inspected

| Scenario on SAME deployed revision | Initial proposed / source admitted | PHOTO approved / attempted | Actually displayed | Measured engine time | Editorial result |
| --- | ---: | ---: | ---: | ---: | --- |
| Classic / New / **Date night** | 6 / 5 | **3/5** | **3 complete 4-item outfits** | ~50s | Photo portfolio applied, 1 eligible pairwise-independent triple, original-photo visual jury applied, separate independent jury unavailable |
| Classic / New / **Work** | 4 / 4 | **2/4** | **1 complete 4-item outfit** | ~40s | Photo portfolio applied, **0 eligible independent triples and 0 pairs**, visual revision timed out; no unreviewed padding |

Both requests were genuine native iOS 26.5 simulator actions on the staging server, with `creativePlanCount:3` AND `creativePlanDetailedArtDirections:3`, source merchant offer admission, `catalogFourItemContract:true`, `catalogFourItemDropped:0`, and original-photo critic on exact proposed physical merchandise. Both user-facing photo groups were actually inspected on the simulator. **Date improvement from previous 1/5 (1 displayed) to 3/5 (3 displayed) is one LIVE canary, not a controlled A/B or proof of stable taste uplift.** Actual live source stock and stochastic creative inference can change between runs.

### Human visual audit of the three Date looks

1. `Smart knit layering`: dark high-neck/knit, grey tailored outerwear, dark trousers, dark boots. Plausible but conventional.
2. `Cardigan & Chinos`: another very dark knit + grey formalish outerwear, brown/earth chinos and brown boots. Too structurally similar to look 1: changed trousers and SKU identities do **not** equal a new artistic mood.
3. `Blazer & Chinos`: lighter top with dark navy blazer, navy/dark trousers and dark leather low footwear. Clearer contrast in color/style, but still conservative for a personalized evening collection.

Read-only database inspection of the resulting private completed deck **only projected non-identifying styling summaries**: two displayed Date looks share concept ID **`smart-knit-layering`** (`signature` + `refresh`), and the third is **`grounded-separates`** (`discovery`). Only **2 of 3** originally source-validated Creative Director plans reached the expensive original-photo judging and the displayed deck. The remaining third card used a different source-grounded editorial concept. Therefore **all-three-photo-approved != all-three-genuinely-independent personal fashion designs**. A photo jury with no independent cross-model result must NOT be claimed to have established human taste.

Work produced just `Refined Tailoring` after PHOTO critic approved two looks but the final editorial jury found they were not an independent pair. This confirms the source-SKU artist often fails to produce genuinely different whole-look solutions on the first pass even with valid original prose. Do not flatten this into a new arbitrary item/subtype combination ban.

## Binding next big problem to solve, not a small style rule

- **First artist must choose from legally accessible actual garment photos**, not only previously extracted numeric visual facets. There are currently **zero recorded source-hash/hosting/external-model-inference photo-license grants** in `catalog_photo_cache_v1`. The admin-only compliant ingest and four-role photo menu already exist, but a live original-photo-first claim would be false without vendor permission. Do NOT turn on the mode with fabricated grants or assume public photos are free to rehost/process. Minimum lawful pilot is 3 exact active SKUs per role; more visual material/fit breadth is advisable to produce three independent stories.
- **Three independent whole-outfit artistic commissions, each executed in real source SKUs**, must be attainable BEFORE expensive photo audition, rather than relying on 6 opaque token arrays to happen to cover the three editorial concepts. The observed 2/3 actual Creative Director coverage is a newly measurable failure in the live journey, not a justification for three static fashion pairing formulas or a softer critic.
- **Human taste calibration:** there are still no submitted legitimate per-look user ratings in the connected `stylist_look_feedback` table as last audited today. The mobile screen now brings EXPLICIT rating controls closer to the look, but the new native screen bundle was not yet built and no fake positive feedback may be seeded. For genuine fashion-world acceptance, require >=30 held-out S5/S6 same-source three-look pairs, >=3 independently blind human judges, >=70% verified S6 preference, zero retail/rights/privacy failures. No such human results exist.

**NO MAIN MERGE / PRODUCTION PROMOTION UNTIL the owner actually sees beautiful, distinct looks, not just successful technical code.**
