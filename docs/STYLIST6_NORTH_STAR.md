# AIWardrobe — STYLIST NORTH STAR (binding project specification)
Version: 2026-11-17 · status: ACTIVE · owner priority: excellent scalable personal AI stylist.
This is the single source of truth for Stylist 6 decisions. Read with root `AGENTS.md` before editing the stylist.

## Non-negotiable product outcome
Given a person's style preferences, owned wardrobe, occasion, local weather, and budget, AIWardrobe selects up to THREE complete, visually coherent, diverse, individualized outfits with authentic catalog/owned items, accurate merchant links and honest offer availability. If the catalog cannot support three excellent looks, return fewer with an explicit reason; never fabricate or pad mediocre looks. The output is assessed on complete real product photographs, not HTTP success, test counts, or the presence of an NVIDIA badge.

The target is not 100% subjective approval (cannot be guaranteed); the measurable acceptance target is >=70% independent human blind preference for Stylist 6 over Stylist 5 on >=30 comparable held-out situations, with 3 independent evaluators, alongside ZERO hard-rule violations and a documented diverse-look rate. The 2026-11-13 blind evaluation showed 5 S6 wins / 5 S5 wins — parity, not superiority.

## System responsibilities and authority
1. **Personal Style DNA** — grounded ONLY in explicit preferences, opt-in profile, previous feedback, owned wardrobe, context. Track favorite silhouettes, formality range, fit, color/texture preferences, brand exclusions, user-owned anchors, and evidence provenance; never infer sensitive attributes or invent measurements.
2. **Fashion Intelligence** — understand every shortlisted actual garment photo + title as visual features: category/subtype, colors, pattern, fabric/texture (when visually observable), volume, construction, formality, weather suitability, provenance and image reliability. Unknown metadata stays UNKNOWN; never describe a title parser as visual analysis.
3. **Creative Director** — for the specific person and occasion, plan THREE genuinely different looks (signature, refresh, discovery) *before* item retrieval: silhouette, outfit architecture, palette, material relationships, footwear, formality. Use an approved style-reference library; do not hallucinate garment SKUs. Generate multiple candidate combinations grounded in catalog inventory.
4. **Deterministic feasibility & shopping truth** — hard gates for real SKU, ownership, stock claims, anchors, temperature/safety, weather, budget using verified FX, catalog media, shop URL. A model cannot override hard gates. `live_in_stock` requires real size-specific seller evidence; otherwise label `Check at store`. Merchant public image URLs do not establish commercial or inference rights.
5. **Independent fashion judge** — inspect exact real candidate photos as a COMPLETE OUTFIT, using bounded verified garment titles only as context. Grade occasion, silhouette/proportions, color harmony, material compatibility, footwear, personal match, and deck variety. Veto visual mismatches. Model judgment is evidence, not proof. An outfit failing the minimum quality threshold is not displayed to fill a quota.
6. **Learning** — user thumbs/explicit feedback should improve subsequent decks. Measure preference shift on fixed scenarios; user-chosen pinned pieces take precedence over generic taste.
7. **Presentation/scale** — retain exact server-approved outfit IDs and per-look visual evidence, show honest availability, explain the style decisions, support bilingual/localization, caching, concurrency, latency and cost budgets.

## Rules for ALL agents and contributors
- Do NOT spend the main development budget adding one-off regex/category vetoes for every bad outfit. A narrow safety/occasion guard is allowed only with real image failure evidence and a generalizable mechanism. Root-cause improvements to creative planning, item understanding, and evaluation take priority.
- Never equate 3/3, HTTP 200, `NVIDIA_APPLIED`, a CI pass, or ranking metrics with excellent fashion.
- Never report a visual improvement without an exact before/after set of source item IDs + actual photos, fixed scenario and independent evaluation. Keep a private contact sheet and evaluation record; never commit downloaded merchant imagery or secrets.
- Never claim Stylist 6 has defeated Stylist 5 without blind human held-out evaluation (>=30 with 3 reviewers; >=70% preference). AI-only judge is a proxy and must be labeled as such.
- Do not create Stylist 7 or rewrite the app while Stylist 6 can be improved via compatible seams. Keep `generate-outfits` production Stylist 5 until release gates + explicit owner rollout decision.
- Favor measurable root cause resolution over commit volume. Preserve existing authentication, subscription, RLS, one-debit semantics, immutable price provenance, consent and rollback.
- Avoid recommending catalog SKUs with stale/offline imagery or unverified store pages. Never misrepresent stock or shipping.
- Never silently change the user's owned garment requirements, budget, or occasion for diversity.
- Every substantive change must reference a failing real scenario (or a cross-scenario root cause), show before/after, verify native iOS where possible, and log deviations to this plan.

## Workstream priority and release gates
| Stage | Deliverable | Evidence to exit |
| --- | --- | --- |
| A — Style QA baseline | 30+ diverse held-out scenarios + complete product-photo contact sheets + S5 baseline | reproducible immutable dataset; manual rubric covering 3 looks |
| B — Catalog intelligence | high-salience shoes/tops/layers analyzed from actual photos; trustworthy SKUs | coverage by slot and subtype, spot checks, stale-evidence rate |
| C — Creative Director | context-aware planned direction blueprints + inventory grounded composition | meaningful architecture diversity, no poor quality fill, request-personalization ablations |
| D — Personalized taste | opt-in profile and signed feedback model | same occasion, different profiles produce appropriate different outfits |
| E — Independent evaluation | model/provider diversity and independent human blind assessments | >=70% S6 preference on >=30 scenarios with >=3 reviewers |
| F — Scalable staging | authenticated 3-look iOS acceptance, latency/p95, quota/cost, stock/rights correctness | deterministic release-gate report and rollback plan |

## Current known blockers (2026-11-16)
- Static hard-coded fashion concepts and seed ranking dominate the composition process; hosted vision has primarily been judging preassembled looks.
- Personal taste / feedback support is bounded and useful but is not a robust style identity or personal creative planning.
- Cloud staging lacks authorized reference-photo media and uses honest text blueprints instead.
- As last checked, only 114 of 1,649 active shoes had stored visual fashion profiles, with the remaining new footwear not auto-enqueued.
- Seller size-specific stock, Uzbekistan delivery/landed cost, merchant image rights and real checkout are unverified.
- Archived independent NVIDIA-family blind S6 vs S5 result 5W/5L with n=10; zero independent human panel wins evidenced.
- 2026-11-16 staging v55 successfully made 3 authorized looks and checked UZS budget using CBU rates, but later fashion/code changes are NOT automatically in staging. Verify exact deployed SHA before attributing results.

## 2026-11-17 — mandatory REAL-SKU AI decision architecture

- Root cause: choosing preassembled score-ranked looks failed real Classic / Work photographic quality (1 final look from 8 candidates). The solution is not more product-title regexes.
- The new fashion AI must compose THREE distinct whole ensembles by choosing ACTUAL source-verified SKU tokens for top/layer/bottom/shoes, not merely rank previously assembled outfits. Existing source-provenance, owner exclusions, budget/FX, physical/weather and locked-anchor checks are mandatory before a new composition is eligible for independent full-photo judgement.
- An independent NVIDIA whole-outfit photograph judge approves/rejects each composition. A SECOND AI fashion editor selects only from entire already-verified feasible portfolio choices (1, 2 or 3). It may deliberately choose TWO even when individually approved triples exist; the UI must explain this honestly.
- Authenticated native iOS Classic / Work on deployed staging 33fb1b1: 3 model-composed source-valid new outfits, 4 independent photo approvals out of 8 requested, final 3 selected, ~39.9 seconds. Classic / Formal: 3 new source-valid compositions, 4 photo approvals, final AI chose 2. This is engineering progress, NOT 70% independent human blind S6 superiority.
- Active shoe visual profiles rose to 671/1,649 (the historic 114/1,649 in 2026-11-16 blockers is not current). Merchant stock, rights, checkout and Uzbekistan delivered prices are unverified.
- Complete grounded evidence and failures: docs/reports/stylist6-direct-sku-composition-20261117/README.md.

## Each work session's completion report must include
GitHub SHA; staging Supabase function version + pinned READY compute SHA; code/quality changes; real scenario IDs & images; what objectively improved; what deteriorated; how many looks were provider-judged vs fallback; product stock/rights truth; QA gate counts; production status; exact NEXT highest-leverage root cause.
