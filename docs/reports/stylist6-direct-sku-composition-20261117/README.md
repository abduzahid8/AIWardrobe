# Stylist 6 — autonomous real-product composition: 17 November 2026

Status: **REAL authenticated iOS staging verified; NOT quality accepted for production.**

## Root problem and actual correction

On the same Classic / Work / New Clothing iOS request, choosing among ranker-preassembled outfits returned ONE NVIDIA-approved final look on the c4a4104 iteration (7 of 8 lacked acceptance). Endless garment-title vetoes and high test counts do not solve bad outfit assembly.

New path:
1. Active source-authenticated Supabase catalog with original merchant links, photos, item prices and user-exclusion checks.
2. AI Creative Director designs the requested style. The NEW AI SKU Composer (backend/stylist6/sku_composer_ai.mjs) gets up to 12 true source-item choices per top, outerwear, bottom and shoe slot. Four per slot may be independent pre-admitted catalog alternatives NOT previously combined by the ranker.
3. NVIDIA Nemotron 30B **actually picks three outfits via exact product tokens**, with unique role-column choices. An unsupported JSON-schema uniqueItems key was reverted after an HTTP provider error; the backend performs its own mandatory token uniqueness check.
4. The backend rechecks the NEW combinations against actual SKU provenance, item identities, stock claims, brand/colour exclusions, pinned anchors, weather, structural feasibility, and strict verified FX budget before using them.
5. The independent NVIDIA 120B visual model reviews the actual product photos within each COMPLETE outfit, and rejects weak ensembles.
6. The final AI Fashion Editor compares independently photo-approved combinations as whole decks of 1–3 and picks only verified, merchandise-unique, compatible sets. It is permitted to leave out the third.
7. Existing verified source-first selector remains fallback if AI composer/provider fails. Original production Stylist 5 is untouched.

## Actual iOS + Render staging receipts (not mocked)

| Scenario / exact staging SHA | Approved photo verdicts | Displayed | AI direct SKU composer | Final AI portfolio editor | Server time |
| --- | --- | --- | --- | --- | --- |
| Classic / Work, c4a4104 (before SKU composition) | 1 of 8 | 1 | no | not run | 59.9s |
| Classic / Work, d2f8054 | 5 of 8 | 3 | YES | chose 3 | 56.2s |
| Classic / Work, a5e5228 (exploration draft) | 3 of 8 | 2 | no: model rejected composition | chose 2 | 47.4s |
| Classic / Work, 2c966e3 (unsupported NIM schema) | 2 of 8 | 2 | no: provider unavailable | not run | 48.9s |
| **Classic / Work, 33fb1b1** | **4 approved, 7 valid of 8; one unrelated image unavailable** | **3** | **YES: 3 proposed and source-admitted** | **YES: 3 selected** | **39.9s** |
| **Classic / Formal, 33fb1b1** | **4 of 8** | **2** | **YES: 3 proposed and source-admitted** | **YES: intentionally chose 2, though 2 feasible triples were offered** | **58.7s** |

Staging runs used the authenticated iOS simulator AIWardrobe-S6-QA-20261115 and LIVE Render staging srv-dbtd336i0phs73fd0sbg at 33fb1b16e74e8bbfab383014e8180c718812f278. Matching Render structured logs are tagged stylist6_safe_visual_quality. Local proprietary merchant screenshot evidence was inspected on the owner's Mac; it is not committed.

Classic / Work final three: (1) white Oxford, navy formal trousers, dark leather loafers; (2) light-blue Oxford, navy pinstripe blazer, cream pleated trousers, blue suede shoes; (3) pale-blue shirt, taupe blazer, brown trousers, blue suede loafers. All are genuine sourced items. Looks (2) and (3) still share similar tailoring architecture, so 3/3 DOES NOT prove excellent fashion.

Backend: 504 targeted suite tests passed, TypeScript type checking, ESLint and Node compute bundle successful. Client curated-two-look copy later fixed with 35/35 frontend service tests. No protected source images, private SKU identities, secrets or authentication tokens were committed.

## Explicit blockers — DO NOT CLAIM WORLD CLASS YET

- Independent human blind quality: no new three-rater 30-scenario comparisons; required >=70% Stylist 6 wins vs Stylist 5 not proven.
- Visual fashion-profile gaps: 671 of 1,649 active shoes, 4,094 of 5,267 upper-body items, 1,259 of 1,638 lower-body items currently profiled.
- Merchant size-specific inventory, checkout, Uzbekistan delivery, complete same-currency landed cost and external photo usage rights not independently verified. UI must show Check at store rather than pretend in-stock.
- Cloud latency 40–59 seconds; must measure p95 and real production unit cost.
- Supabase staging function proxy pin was NOT promoted by this isolated Render staging work; do not equate latest GitHub SHA with production or Vercel preview.

Future agents: keep authentic model SKU composition -> immutable hard commerce gates -> independent full-photo critique -> model-level whole-deck edit. Do not return to endless product-title regex patches or present three displayed cards as proof of elite styling.
