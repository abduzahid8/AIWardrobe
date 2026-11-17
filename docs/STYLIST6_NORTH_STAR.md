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

## 2026-11-17 — LIVE gateway correction and photographic rejection evidence

- Root-cause environment finding: the normal iOS QA gateway was still bound to a 50s Vercel runtime; the real AI SKU composer and final portfolio editor in `cloud_handler.mjs` are deliberately enabled only for server budgets >=70s. Code existing in GitHub was therefore NOT enough to make the default QA endpoint use it.
- Correction: commit `4a85034` changes only the `aiw-stylist6-node-staging` source entrypoint to import the existing 85s Render proxy. Supabase function **v61 ACTIVE**, `verify_jwt=true`; Render staging compute **33fb1b1 LIVE**, 72s internal budget. Preserved JWT, HMAC, merchant/budget/source checks and no-retry quota semantics. Production `generate-outfits` (Stylist 5) untouched.
- Critical distinction: v61 deployment and source inspection are proven; an authenticated **post-v61** iOS generation and end-to-end quality assessment have NOT yet been confirmed.
- Fresh Render structured traces from before the gateway switch on the same compute code: at 10:03 UTC **3 SKU-composed and admitted; 3/8 photographic approvals; 2 displayed; editor applied** (47.1s). At 10:07 UTC **3 SKU-composed and admitted; 1/8 photographic approvals; 1 displayed; editor not run** (59.0s). Multiple rejects were `style_mismatch` or `below_quality_floor`.
- These are **counterexamples** to treating a successful model composition, a green deploy, or previous 3-card runs as proof of great taste. Most recent real failures originate in aesthetic **proposal quality**, not missing valid SKUs; the full-photo critic is correctly stopping unconvincing results.
- NEXT highest-leverage improvement: inspect exact photo contact sheets and rejected whole-look relationships on the fixed failed scenarios; make the model propose better *complete* silhouette/material/footwear combinations before admission or use photo-critic feedback for a bounded second composition. Prove improvement on the SAME items and preferences, not via new regex bans or changing the acceptance threshold. Then rerun an authenticated v61 request and independent >=30-scenario human blind quality gate. Keep offer/rights/size claims honest.

## 2026-11-17 — targeted footwear visual evidence backlog

- Live Supabase: 8,554 active catalog SKUs; 6,024 active v1 visual fashion profiles. Tops: 4,094/5,267, bottoms: 1,259/1,638, shoes: 671/1,649. Of 978 shoes without profiles, 977 had never entered catalog_visual_jobs.
- The configured visual worker was enabled, but no catalog visual pg_cron job was found. Calling its ordinary `seed` action would enqueue the entire catalog; that is NOT required to repair the shoe bottleneck.
- Inserted **64 targeted source-hash-valid high-priority pending jobs** with original HTTPS photos, merchant links and nonzero prices: 20 loafers, 16 derbies, 16 Oxfords, 12 boots. Rechecked Supabase counts: 64 distinct active shoes in `pending`, zero stale hashes/invalid merchant links. These are QUEUED, not analyzed, and no worker processing was claimed.
- Existing source-aware, exact-hash `catalog_visual_batch` contract remains unchanged. Do not claim shoe coverage is improved until completed visual rows exist. Avoid automatically seeding thousands of other garments without a cost and throughput plan.
- Regression test `__tests__/features/stylist6CloudPublication.test.js` now protects the default QA route against reverting to a 50-second compute path where the full composer/editor is silently disabled. New code test has not been locally executed here; CI result must be checked.
- Next: run the existing authenticated visual worker for the bounded pending jobs when its operator access is available; verify how many finished with image-observed attributes and are used by actual outfit candidate discovery. Audit exact rejected outfit photos before changing editorial thresholds.

## 2026-11-17 — owner correction: exactly FOUR real shoppable items per look

The owner explicitly identified a regression: the native iPhone Stylist6
generated three-garment catalog looks even though each shoppable recommendation
must contain FOUR actual products. Root cause: cloud input always set
`optionalLayer:true`; `resolveOutfitComposition` allowed an omitted
outerwear slot and the AI SKU composer legitimately emitted `outerwear:'none'`.
An attractive photo verdict on the three remaining items cannot silently
fulfill the owner's four-product contract.

- On the isolated >=70s Stylist6 Render catalog staging path, every full look
  must contain **one real top + one real appropriate outer layer + one real
  bottom + one real pair of shoes**. `optionalLayer:false` is sent to the
  original outfit designer; no invented synthetic fourth product.
- The cloud response **enforces all four SKU-backed roles again** before
  mapping to iOS, rejecting missing/repeated/unsourced fourth items.
  The fallback may honestly return one/two complete looks, or a non-success
  when none passes. Never pad a rejected/missing fourth item, suppress
  NVIDIA objections or claim a three-item look fulfills four-item styling.
- Wardrobe and mixed user-owned sourcing, and short-budget Vercel/S5
  deployment paths are not changed by this staging contract. Production
  Stylist5 remains untouched. Dedicated acceptance tests include 72s
  authenticated cloud with four real merchant SKUs, partial invalid deck
  filtering and independent whole-look photo evaluation.
- **Known product-design constraint:** physically sensible dressing at 30°C
  or an explicit request to avoid a jacket may call for an accessory instead
  of a layer. This existing four-slot catalog schema has NO independently
  sourced accessory role. We must not silently turn a 30°C request into a
  heavy jacket; an accessory-capable four-product outfit needs real merchant
  accessory inventory, schema and independent full-photo judgment. Until
  those exist, strict four-item staging may have fewer/no suitable looks
  rather than a fabricated perfect result. This is OPEN product work.

## 2026-11-17 — six-look autonomous AI fashion audition and real personalized taste

- Original 72s Render feedback loop from c459bef was **LIVE**, not merely in a branch. Three logged Real-photo runs after it: 10:27 UTC -> 6/8 photo approved, 3 displayed, final editor applied; 10:31 UTC -> 5/8 photo approved, 3 displayed, editor applied; 10:36 UTC -> 4/8 photo approved, 2 displayed, editor applied. This is mixed evidence, NOT a blinded human quality win. All three showed 3 independently AI-composed and source-admitted candidates; the proposed second photo revision was not run in these traces.
- Strategic improvement: instead of generating THREE AI ensembles plus FIVE look-builder reassemblies, the autonomous SKU compositor can now propose SIX **different full outfits** in ONE bounded NVIDIA call. This uses the same real source-bound 48-product menu and existing eight-candidate independent photo review, exact budget/merchant/source/weather gate, hard no-product-reuse, security and final editor. Invalid or duplicate AI proposal is discarded individually; at least three distinct source-valid proposals required to accept the six-look audition result. Fewer than six usable per-slot products fall back to the original three-look call.
- A fashion AI must actually learn the wearer's own preferences. The compositor now receives bounded, anonymous **explicit user thumbs up/down outfit semantics** (maximum two liked plus two disliked holistic style signatures) and exact-token prior user-rated merchandise affinity. The model sees only generic garment attributes and a liked/disliked indication; never user IDs, raw past item IDs, timestamps, feedback rows or image URLs. Feedback shapes creative selection but never bypasses hard gates or the independent photo critic.
- GitHub staging branch changes span `sku_composer_ai.mjs`, `fashion_deck.mjs`, `cloud_handler.mjs`, and the SKU/composition test suites. These change the deployed Stylist 6 canary architecture, NOT Stylist 5 production.
- **Quality is NOT accepted yet**. For the new six-look+explicit-feedback build, require an actual Render LIVE pinned SHA, post-deploy authenticated iOS trials on the same held-out scenarios, photo rejection breakdown and real images; compare rates, full-outfit style, distinct silhouettes, and elapsed time against the earlier three-AI-look baseline above. Keep the final >=70% independent blinded human panel threshold. Do not claim an upgraded stylist based only on the new AI candidate count.

## Each work session's completion report must include
GitHub SHA; staging Supabase function version + pinned READY compute SHA; code/quality changes; real scenario IDs & images; what objectively improved; what deteriorated; how many looks were provider-judged vs fallback; product stock/rights truth; QA gate counts; production status; exact NEXT highest-leverage root cause.
