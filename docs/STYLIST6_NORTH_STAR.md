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

## 2026-11-17 — licensed, original-source product photo cache foundation

New server-only `catalog_photo_cache_v1` live Supabase table is RLS-protected;
anon/authenticated have no SELECT/INSERT, service_role only. **Zero READY
rows exist** until real SKU-specific hosting and NVIDIA inference rights
are confirmed and a JPEG is actually inspected, uploaded and hashed. The
admin-only offline CLI `scripts/stylist6/ingest_licensed_product_photos.mjs`
defaults to DRY RUN, requires explicit rights evidence and verifies the
source row both before and after download/upload. Every cached image is
bound to exact current SKU plus original image URL, rights scope and raw
JPEG SHA256, with a 30-day verification limit. Cloud handler has a
server-only first-party photo resolver; corruption/revocation/deadline
falls back to original instead of fabricating photo verification.

The 72s staging creative AI may invoke VISUAL-FIRST only if both existing
server flags are set AND at least 3 legitimately stored, current photos
per TOP/LAYER/BOTTOM/SHOES are in the discovered catalog. It never runs
a merchant CDN fan-out as part of visual composition. The unchanged
independent whole-outfit PHOTO critic and final visual collection editor
continue to own quality vetoes. **Neither licensing nor aesthetic release
has been proven**. See `docs/STYLIST6_LICENSED_IMAGE_CACHE.md`
and GitHub issue #17 for acceptance.

## 2026-11-17 — correct photo-first availability and licensed image-cache gap

Native iOS tests of primary product-photo visual composition (`dbefcdd`,
`6ae1c2c`) failed: both recorded photo-board `deadline`, text AI
fallback, and only **1 independently photo-approved look out of six**.
Even the six-photo-per-role asynchronous quorum was too slow for
unreliable source CDNs. The next staging attempt uses a photo board of
**at least three independently authorized and decoded real product images
per category**, composing three real four-product looks from pictured
SKU tokens; full budget/merchant/photo/hard gates are unchanged. It is
a bounded, testable availability experiment, NOT an aesthetic guarantee.

Supabase has 3,853 existing public `shop-catalog` objects but only
1,534/8,554 ACTIVE item image_url values point to first-party assets,
ALL shoes. No top/bottom first-party product URLs exist in the current
shop_catalog. A source-hash-verified, licensed local photo-cache ingest
is needed for cheap reliable worldwide merchant scale. Never silently
substitute cached photos for an updated SKU or duplicate third-party
merchant photos without confirmed rights and provenance.

## 2026-11-17 — primary creative decision must see REAL source PRODUCT PHOTOS

Real iPhone QA on b9b7863 proved the AI-first catalog composer and
independent full-photo critic work, and the collection-level visual editor
finally applied on actual merchant garments (6 composed/5 admitted,
3/7 photo-approved, two compatible displayed looks). However the
compositor had still seen TEXT labels only and picked very similar
merino+cardigan architectures.

Staging-only candidate solution: supply the creative SKU model one JPEG
contact board of the actual authorized original product photographs
before picking any complete outfits (up to 48 anonymous P-tokens,
four roles). If source media/VLM unavailable or fails validation, fall
back to prior source-valid TEXT composer. Once VLM proposes 4-product
looks, the existing immutable budget/merchant/physical/source gates
and independent NVIDIA PHOTO judge must accept each whole ensemble;
final visual editor chooses only preverified complete collections.
No fictitious garment, no invented on-body fit or purchase offer,
no relaxed quality thresholds. Synthetic NVIDIA visual composer
integration succeeded with six source-valid looks in 2.5s. This is
**NOT** proof of improved real-client fashion quality until live
`skuCompositionPhotoFirst:applied` + actual native screenshots are
collected and the blind human acceptance gate passes.

## 2026-11-17 — visual photo jury must be CURRENT and actually image-capable

A real NVIDIA API probe found the old final visual jury endpoint
`nvidia/nemotron-nano-12b-v2-vl` returns **410 Gone** despite old docs
advertising it. Replaced with the currently hosted
`nvidia/nemotron-3.5-super-vl-120b-a12b` in a separate final
fashion-portfolio editing call. It may select only source/physical/photo
verified 1–3-look portfolios and retains text fallback. The image-board +
live model worked on synthetic imagery (HTTP 200, one valid F-token),
but REAL iOS 3+ source-photo approvals and visual-jury
`applied/ready` are still required for acceptance.

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

## 2026-11-18 — fix actual source-photo montage on Render before claiming two juries

First pinned \`aadab9a\` iPhone Date night generated three verified
source-SKU four-item outfits and five individual photo approvals,
BUT the final photographic contact sheet missed a 3.8s deadline:
\`photoEditorialVisualMedia=deadline\`,
\`photoEditorialIndependentJury=not_run\`; no 2nd model ran.
A local optimization now resizes original-photo bytes into raw
RGB and uses ONE public SVG token-label layer (instead of 4+N
full-canvas SVG layers and per-cell JPEG re-encodes).
All original media rights, missing-photo, no-unreviewed-filler
and total-time bounds remain unchanged. A real iPhone request
must still prove the new original-photo jury is running.
Evidence:
\`docs/reports/stylist6-user-reference-text-fashion-memory-20261118.md\`.

## 2026-11-18 — independent Kimi vision studio jury over ACTUAL photographs

Real staged iPhone Classic/Date night \`a1dfc12\` delivered three
original-photo approved 4-SKU complete looks (7/8 decisions and
~39.6s), yet all THREE resembled knit cardigan + chinos.
The SAME Nemotron family had produced, photo-judged and curated
its own fashion. The creative goal of three distinct beautiful
ensembles is still NOT proven.

Isolated S6 staging now makes one concurrent PHOTO comparison with
\`moonshotai/kimi-k3.1-pro\` (actually probed against the connected
NVIDIA image API, unlike alternatives that returned 410/timeouts).
Each lab independently sees the SAME authorized real source-product
JPG contact sheet and an anonymized menu of existing verified complete
outfit portfolios. The independent model can veto a bloated set ONLY
by selecting a strictly contained, already server-verified F-token
portfolio. It cannot add, replace, invent or approve a missing SKU.
Absence, slow response, model disagreement or invalid selection never
weakens the original NVIDIA independent-photo quality gate. The new
staging-safe enum \`photoEditorialIndependentJury\` measures whether
the new editorial opinion actually differs from the Nemotron editor.
This is not a forced diversity ban or proof of better taste.
See \`docs/reports/stylist6-user-reference-text-fashion-memory-20261118.md\`.

## 2026-11-18 — real whole-look artistic intent, not merely category lists

Pinned \`080e0fc\` authenticated iOS Classic/Work: FIVE source-feasible
four-item AI outfits, ZERO independent PHOTO approvals; rejected for
\`below_quality_floor\` and \`model_veto\`. The original 30B director
emitted only a name, a 3-value silhouette enum, palette and category
subtypes. Worse, two artistically different designs sharing category
roles were discarded BEFORE SKU composition.

In isolated 72s S6 staging only, the concept-director now chooses a
stronger 120B model and writes an actual professional \`artDirection\`
for each of its THREE independent outfit ideas: proportions, garment
construction, drape, material relationship, color and footwear.
Those bounded textual intentions reach the original source-bound
120B merchandise composer for its own creative SKU decisions. Same
item categories with DIFFERENT designed materials/silhouettes may
survive; exact duplicates are still deduplicated. Real merchant
source/owner/budget/weather/photo rights and the independent whole
look PHOTO critic and final editor are unchanged. Normal 50s
preview remains with its 30B planning model. This has NOT yet
passed real staging or human quality acceptance. Full receipt:
\`docs/reports/stylist6-user-reference-text-fashion-memory-20261118.md\`.

## 2026-11-18 — genuine TEXT fashion inspiration, not product-title rules or unlicensed pictures

Real iOS Classic/Date night PHOTO judged 3/5 initial source looks, and
rejected all 3 alternative ensembles even when the second AI saw
unseen merchant inventory; the two displayed choices were still
cardigan/chino variants. More SKU choice alone is insufficient.

The project already holds 46 user-uploaded MANUAL visual-outfit
captions (not a certified human taste corpus; photo-media rights
are explicitly UNVERIFIED). We now retrieve up to three distinctive
**text-only outfit architecture references** for the user's actual
style+occasion before the 30B Fashion Creative Director and 120B
exact-source-SKU composer make their own design decisions. This
helps the AI see varied silhouette, layer, shoes and material
relationships without inventing a SKU, requiring a photographed
item's appearance without permission, repeating a tiny banned-title
rule, or overriding user exclusions. Full source, four-product,
budget/merchant/weather, physical and independent visual PHOTO
approval remains mandatory. We NEVER send the original reference
JPEGs or their filenames to provider APIs. Tests and full provenance:
`docs/reports/stylist6-user-reference-text-fashion-memory-20261118.md`.
Real iOS and 70% human blind taste validation remain OPEN.

## 2026-11-18 — AI creative repair explores NEW merchant items, not the same 48 rejected tokens

Real 120B Classic/Work finally ran the genuine PHOTO portfolio VLM ->
second AI source-SKU composer -> independent NEW PHOTO review -> final
portfolio choice; one alternative was approved, but both displayed
looks still used tailored jackets. Real Classic/Date night had three
initial PHOTO-approved outfits but final visual editor selected two
very similar cardigan/chino looks; new source re-composition from the
SAME 48 source items sent three proposals to the critic and earned
ZERO approvals. Core issue is narrow creative candidate search.

On isolated staging, the second AI now gets a **new** exact-SKU menu
from previously UNSEEN authenticated catalog garments across four
roles, still sourced from the existing 400-row snapshot, honoring
owner-excluded IDs, visual evidence and all original physical,
photo/license, budget, merchant and final-replay gates. Only if this
menu cannot cover every garment role does it fall back to the original
source candidate pool. Diagnostic:
`photoEditorialCreativeInventory=unseen_source_skus`.
See `docs/reports/stylist6-native-two-photo-approval-editorial-20261118.md`.
This expands real AI fashion decisions; it does NOT guarantee a
great third look or satisfy the human blind acceptance gate.

## 2026-11-18 — two PHOTO-approved native looks MUST receive a real editorial AI opportunity

The first LIVE native response after stable completed-deck recovery
confirmed 2 source-shoppable iPhone cards, 9 NVIDIA whole-outfit photo
judgments / only 2 approvals and an unused ~40-second server creative
budget. The final PHOTO portfolio editor was silently not run due to an
artificial >=3-approved precondition. On isolated 72s staging, now
invoke that genuine independent visual fashion editor for 2 approved
looks as well: it can critique the missing THIRD direction and request
a new AI source-SKU design, then its photo-verified second pass. Never
invent an unphotographed third look. Exact QA and source-backed
regression: `docs/reports/stylist6-native-two-photo-approval-editorial-20261118.md`.
User-visible two looks is a transport PASS, NOT aesthetic acceptance.

## 2026-11-17 — photo portfolio fashion editor now can CREATE corrective new styling directions

Previously the final independent actual-photo fashion editor could only
discard near-clone card 3, never request a new genuinely different
fashion direction. On isolated 72s staging only, a photo editor may now
return a machine-validated `missingFashionDirection` in addition to
its ORIGINAL exact verified F-token choice. When there is >=22.5s
server time and unused independent-photo quota, a bounded second
AI Fashion Director chooses unused merchant SOURCE SKUs informed by
the photographic complaint. Every new 4-item outfit is re-source-,
price-, physical- and merchant-verified and independently PHOTO
critic-reviewed. A second real photo portfolio choice and original
NVIDIA replay are required before any changed look is shown. If
anything fails, keep the original approved partial deck. No weak
padding, no model-skipping, no S5 production change. See
`docs/reports/stylist6-photo-editorial-creative-revision-20261117/README.md`.
This is not an aesthetic win until real iOS and human blind evaluation.

## 2026-11-17 — 120B fashion designer staging audition (NOT acceptance)

Correct native iPhone scenario NEW CLOTHING / CLASSIC / WORK on the
published `7fb8df2` 30B direct SKU fashion model returned ZERO
independently PHOTO-approved looks, with explicit
`stylist6_visual_quality_unmet`. Previous 1/9 approval was a
**different** Classic/ANY scenario: QA accidentally toggled off the
already-selected Work occasion, verified from its pre-generation screen.
Do not treat that first result as Work.

Actual paid NVIDIA source-menu provider probe, matched synthetic
catalog/Work brief, independently called the 30B text composer and
the 120B VL model through the same original strict JSON-schema
source-token validator: BOTH generated six complete source-unique
four-garment proposals (HTTP 200); 120B took ~1.98s vs 30B ~2.61s.
This is latency/format evidence ONLY, not a human aesthetic score.
On isolated >=70s Render STAGING ONLY, audition the more capable
`nvidia/nemotron-3.5-super-vl-120b-a12b` for initial creative SKU
selection, keeping the independent original-photo critic, verified
SKU/offer, price, fixed 4-article and weather hard gates unchanged.
No model selector or CI test can replace the required REAL native
and blind human 70% fashion-preference evaluation.


## 2026-11-18 — source-photo quality and footwear-scale checkpoint (verified)
**This section is evidence, not aesthetic acceptance.** PR #15 remains isolated. Production Stylist 5 is unchanged.

- Verified Render staging LIVE on exact source `9c8cdf0` after both dedicated Stylist 6 GitHub checks passed. The source-bound four-garment catalog contract is now enforced BEFORE expensive photo evaluation on the isolated AI-first path, not solely at publication. The photo-critic-informed second AI pass has a four-source-garment regression test. This is NOT a real native fashion approval.
- Supabase verified 1,649 active shoe SKUs and 735 actual source/image-hash-current visual shoe profiles before the next work cohort. All 735 source hashes and source image URLs matched their current job and catalog rows. A previously pending 64-shoe cohort completed: total succeeded visual jobs 6,020 -> 6,084, no pending work at that snapshot.
- Another 908 eligible currently active shoe SKUs with real HTTPS product/shop URLs and verified shoe slot were queued in two source-hash-bound, insert-only cohorts (512 + 396). GitHub Actions `stylist6-footwear-visual-backfill.yml` runs FOUR workers, max 128 images/worker and max 2,400 seconds, and is explicitly guarded with `STYLIST_VISUAL_SKIP_SEED=1`. First workflow run `44235881082` was in progress at checkpoint. **Queued is NOT processed**; independently query job success and current profile identity before reporting coverage gains. The new 396 must be handled in a subsequent bounded run; check its actual workflow status, never assume completion.
- The source-hash-verified **licensed** first-party photo cache `catalog_photo_cache_v1` had **zero rows**. Therefore `photoFirstCatalogReady=false` and the AI SKU designer still works from names and authenticated derived visual attributes, not its own direct photo board. The independent whole-look critic can see real source photos, but that happens AFTER composition. Download/inference host allowlists are technical controls, **not merchant image rights**. Do NOT bypass or fabricate media licensing to enable photo-first.
- Last documented real iOS Classic/Date night three-look output used near-identical cardigan/chino silhouettes despite 7/8 independent photo approvals. Later real Kimi secondary visual jury hit a source-photo montage deadline; the optimized raw-image/one-SVG contact sheet and Kimi provider sampler repair subsequently reached Render staging, but real native confirmation of the latest two-model PHOTO editor and genuine silhouette improvement was **not yet captured** in this checkpoint.

**Next decision sequence — do NOT drift:**
1. Verify current queued shoe profiles succeed with exact source/image-hash, and inspect relevant style-specific availability (not just broad catalog size).
2. Run paid, authenticated **native** photo-backed QA on EXACT `9c8cdf0` or later deployed SHA for Classic/Work, Classic/Date, Old Money/Casual, Winter/Work. Capture only authorized original-source product-photo contact sheets privately. Log three true four-SKU shoppable looks, or honest fewer, plus Kimi/Nemotron photo editor statuses and architecture diversity. No claim of success from CI or 3/3 cards alone.
3. Make the **composer see actual SKU photographs BEFORE selecting** only after documented external-AI and image-hosting rights for at least 3 exact current source SKUs in each of four roles, then >=12 per role; use the bounded already-implemented `catalog_photo_cache_v1`/SKU board, no live 48-CDN fanout.
4. Where native photos still show cloned silhouettes, improve model-led whole-look design and creative revision from original PHOTO evidence; do not append product-name regex bans or relax the photo rejection floor. Include a same-inventory before/after visual comparison.
5. Before scale or S5 replacement, achieve >=70% Stylist 6 wins in >=30 held-out blinded S5 pairwise scenarios with three human reviewers, zero source/price/auth/privacy violations, verified native delivery and rollback.

## 2026-11-18 — independent creative model + photo-portfolio outage repair (current server checkpoint)

**Only verified source/runtime evidence, NOT a successful fashion release.**

- The 908 source-hash-queued active shoe images were processed by the bounded
  visual worker: Supabase now reports **6,992** succeeded catalog visual jobs,
  zero pending/leased. Exact current-source catalog joins confirm **1,643 of
  1,649** active shoes have current `catalog_visual_fashion` profiles. Six still
  lack current visual profiles. Earlier `unknown` worker jobs remain historical;
  never count them as successful. The authorized product photo-first cache
  `catalog_photo_cache_v1` still has **zero rows**, so the initial creative
  design cannot truthfully claim it viewed a licensed image-board before SKU
  composition. Host allowlists are NOT third-party media licenses.
- `42584b0` authenticated Classic/Any iOS generated TWO honest four-source-SKU
  results after 9 photo judgments; both were nearly identical knit/cardigan
  silhouettes. This was Any because the native QA flow accidentally toggled
  the selected Work occasion OFF. Fixed subsequent QA flows to capture the
  selected occasion BEFORE submitting the paid request.
- On source-deployed `649598d`, correct native Classic/Work showed THREE
  full four-source-SKU looks. Diagnostics: six composed and six admitted, six
  independent photo judgments, four photo approvals, three displayed.
  Screenshots showed genuinely different leather-jacket, trench-coat and
  blazer-based outfit constructions; the trench-coat merchant JPEG loaded
  with a delayed native image render. This is evidence of variation for ONE
  request, not consistent beauty.
- Correct native Classic/Date night on that revision gave 11 PHOTO judgments,
  seven individual approvals, but the final Kimi photographic portfolio vote
  narrowed the final deck to TWO repetitive knit/cardigan-led looks. This is
  a genuine failure of collection-level aesthetic diversity despite passing
  per-outfit tests and strong catalog coverage.
- `8e09ff5` adds real **independent Kimi K3.1 Pro CREATIVE rescue** after
  photographic rejection, instead of the Nemotron composer repeatedly
  rewriting its own stale concepts. Only source-menu-valid tokens are
  admitted. On unavailability/malformed output, fall back once to Nemotron.
  An actual Kimi hosted API probe returned valid three-four-item JSON in
  ~2.36s; 224/224 relevant local tests and both integrated GitHub CI jobs
  passed. Authenticated native Date night returned THREE photo-verified
  cards (5 initial proposed source-feasible, 4 PHOTO approved) but its final
  120B PHOTO portfolio VLM timed out, so Kimi CREATIVE rescue was **not**
  exercised and the rendered 3 cards were NOT a confirmed visual-fashion win.
- `0bb855a` repairs that concrete photo-portfolio failure: when the PRIMARY
  visual 120B editorial call times out/returns invalid selection, the already
  parallel, source-verified, photo-informed independent Kimi editor may supply
  a strictly validated original F-token from the exact previously approved
  outfits instead of discarding its completed real-photo judgment for a
  text-only editor. The safe diagnostic enum is `standalone`; when Nemotron
  succeeds, Kimi retains its previous restrictive subset-veto role. Kimi
  cannot add SKUs, invent photos or bypass whole-outfit critic/replay.
  **226/226** focused local tests and **2/2** dedicated GitHub CI jobs passed;
  Render staging deployed LIVE on exact `0bb855a4`.
- The first paid native attempt against LIVE `0bb855a4` was **INCONCLUSIVE**:
  a concurrent Metro restart caused Maestro to report the app stopped before
  result assertion; a second setup attempt was interrupted by Expo Go's
  developer menu. This is an iOS QA environment collision, not evidence that
  the backend model failed or that its new photo fallback succeeded.
  A different development agent is also updating native JPEG negotiation
  on the same branch; do not mix that client's code SHA with the Render
  backend SHA or run two Maestro drivers against the same simulator.

**Next high-leverage validation:** hold Metro + authenticated simulator
stable for the entire generated request, capture the four real item photos
for each of three proposed results, plus the `photoEditorialVisualJury`,
`photoEditorialIndependentJury`, `photoEditorialCreativeArtist`, candidate
counts, verified source/price facts, and real occasion. Repeat correct
Classic/Work, Classic/Date, Old Money/Casual and Winter/Work. The `standalone`
receipt should appear only when a real PHOTO Kimi judgment substitutes for
failed primary VLM; no fabricated quality approval. If repeated architecture
clones remain, use additional genuinely independent outfit-conception and
human visual QA, not a lowered reject floor. Before any production switch,
prove >=70% wins across >=30 held-out blind paired human reviews with three
judges. Stylist 5 production remains unchanged.

## 2026-11-18 — source-authentic TWO-STUDIO real fashion acceptance checkpoint

**Verified result: TWO beautiful-enough photo-approved looks; NO acceptable
three-look collection. PRODUCTION S5 NOT CHANGED.** CI/pass status is not a
substitute for real independent visual and human acceptance.

- Staging on exact source SHA dd02fd52700f738aa7ef9e0f9d076df77d4131e0
  ran a new two-studio creative audition BEFORE independent photo criticism.
  Nemotron initially builds six exact-source-SKU proposals. Kimi K3.1 Pro sees
  unused, admitted real merchandise and anonymous prior constructions, then
  creates three independently conceived alternatives from disjoint SKU tokens.
  If any Kimi proposals invent tokens or its API is unavailable, retain the
  initial Nemotron source-verified slate. Both studios' actual proposals must
  still pass identical four-real-item, price, merchant, context, SKU integrity
  and original whole-outfit PHOTO gates. Their existence never guarantees
  that the resulting outfits are tasteful or commercially useful.
- Both GitHub CI checks passed on dd02fd5; 229/229 focused local tests passed.
  The Render isolated staging service was verified LIVE on the exact SHA.
- Real authenticated iOS 26.5 simulator test on dd02fd5 used NEW CLOTHING /
  CLASSIC / DATE NIGHT, zero free-form owner brief and no locked SKU.
  The blue sweatshirt and QA title shown above the setup screen came from
  a developer-only qaJpegSourceProbe native JPEG widget, NOT the baseItem
  anchor. This distinction was confirmed from the Maestro accessibility
  hierarchy. Do not infer user outfit anchor from that diagnostic banner.
- LIVE Render safe QA logs at 2026-11-18T05:44:55Z confirmed:
  elapsedMs=58046; skuCompositionStudio=nemotron_plus_kimi_initial;
  skuCompositionProposed=6; skuCompositionAdmitted=5;
  catalogFourItemContract=true; catalogFourItemDropped=0;
  attempted=10 original-photo whole-look judgments; approved=4;
  displayed=2; photoRevisionArtist=cross_lab_kimi;
  photoRevisionAttempted=2; photoRevisionApproved=1;
  photoEditorialVisualJury=applied;
  photoEditorialIndependentJury=agreed;
  photoEditorialVisualMedia=ready;
  photoEditorialEligibleTriples=0; photoEditorialEligiblePairs=2;
  photoEditorialCreativeRevision=applied;
  photoEditorialCreativeInventory=unseen_source_skus;
  photoEditorialCreativeRevisionAttempted=3;
  photoEditorialCreativeRevisionApproved=1;
  photoEditorialCreativeArtist=nemotron_fallback.
- Key finding: the actual initial independent SECOND artist DID RUN; the
  second original-photo judge also DID RUN; the two independent portfolio
  editors agreed TWO were all that deserved publication. A third visually
  acceptable, source-unique, aesthetically differentiated collection option
  did not exist. Never quietly pad with an unreviewed third look.
- The current visual-profiling backfill covers 1643/1649 active shoe SKUs.
  However the owner-LICENSED first-party initial photo-board cache,
  catalog_photo_cache_v1, still contains ZERO source-verified entries. The
  first AI artist sees derived visual intelligence, not actual authorized
  photographs. Later original-photo judges can evaluate, but cannot
  retroactively give the initial designer true photographic eyesight.

**NEXT EXECUTION — do not get stuck on small keyword rules:**

1. Inspect the ACTUAL source-admitted garment construction breadth available
   to the initial two studios. Full current catalog contains distinct jackets,
   tailoring, coats, bombers and overshirts; investigate why first auditions
   repeatedly yield similar knit/cardigan architectures and improve the
   model-led source menu / creative choice, not a product-name ban.
2. Commission fresh source-verified creative directions early enough that the
   original whole-look photo critic can assess three compelling DISTINCT
   fully shoppable four-SKU compositions within the server deadline. Validate
   on repeated real native screenshots and independent editor receipts.
3. Obtain genuine merchant/owner media hosting and external AI inference
   rights, then populate source-hash-bound licensed first-party photos and
   enable true PHOTO-FIRST composition. Never bypass the rights gate with
   public hotlinked merchant images or assert that a technical host allowlist
   equals legal inference permission.
4. Production promotion remains forbidden until repeated cross-occasion
   native styling quality, zero merchant/privacy/source errors and a held-out
   three-reviewer human blind comparison in at least 30 paired cases shows
   Stylist 6 >=70% wins over Stylist 5.

## 2026-11-18 — real-source inspiration diversity and iOS reality check (bd65cd8)

- Live staging SHA `bd65cd8094f6eece571ef03d1c7be5291254550c`; integrated and visual-first CI both successful; 251/251 focused local tests passed.
- Root-cause change: the 46 human-captioned owner fashion references were previously sampled as near-identical high-neck knit directions for Classic Work and Date. Retrieval now exposes genuinely different human-written top **and** layer concepts; independent Kimi also receives bounded human-captioned inspiration. There is **NO relaxed quality floor, forced 3-card fill, extra invented SKU or new photo rehosting**.
- Authenticated real iOS Classic / Date night: 8 AI proposals, 9 actual whole-look photo decisions, 4 approvals, **3 genuine four-source-product displayed looks** in 46.9s. Third photo editor arbiter chose the photo-feasible triple.
- Authenticated real iOS Classic / Work: 8 proposals, 11 real-photo decisions, only 2 approvals, **2 genuine four-product displayed looks** in 37.1s; second corrective Kimi wave 0/3 photo approvals. This **fails** the universal three-great-looks requirement.
- Manual original-SKU image inspection of Date night also found too-office tailoring and two similar blazer/trouser directions. A three-card result is NOT a great-stylist approval.
- Zero records in the licensed original-product photo cache; raw-photo creative composition remains OFF. Existing v1 photo profiles are useful but frequently lack confident fabric/color observations. No stock/delivery/size claims.
- **DO NOT promote S6 to production**: 30+ held-out 3-human-reviewer blind test requiring >=70% S6 preference has not been conducted; the fashion quality bar is demonstrably unmet.
- Concrete receipts, limitations, repro details and next creative-work focus: `docs/reports/stylist6-real-iphone-owner-reference-20261118.md`.

## 2026-11-18 — visual intelligence source transport recovery pilot (b954367)

- The 1,520 unprofiled active top/layer/bottom products were already `unknown`
  after THREE failed downloads, mostly on particular brand CDNs; blind
  worker reruns without a source-transport diagnosis would waste time.
- Neutral original JPEG GET recovered 5/6 sampled brand hosts when using
  an **honestly identified AIWardrobe client** instead of the blocked
  generic `python-requests` UA; Lacoste remained 403, no circumvention.
- Fix `b954367`: source-only honest UA + Accept header, no added
  credentials/cookies/redirect trust/host permissions. Two source
  transport tests pass, existing visual quality and price/auth gates untouched.
- EXACTLY 48 requeued current-hash garments across five previously failed
  hosts; four bounded GitHub workers in Actions run 44254765390. **No
  successful enrichment yet claimed** — check Supabase source-current
  profile/queue and completed GitHub jobs before considering further batches.
- Continue pursuing **beautiful, truly different outfits** on real iOS
  with genuine photo approvals and human blind review. Raising technical
  profile coverage alone does not close fashion taste or image-rights gates.

## 2026-11-18 — original photo-worker root cause PROVEN in bounded pilot

- **48/48 source-original product photo enrichments succeeded** after
  the honest AIWardrobe client-identification fix; each current
  `catalog_visual_fashion` URL/source hash verified. This is a real,
  measured rise of completed jobs from 6,992 to 7,040 (GitHub Actions
  run 44254765390 four out of four jobs SUCCESS).
- A **second 512-image exact-source bounded queue** (Armani 256,
  Loro Piana 104, Fred Perry 60, Zara 60, Massimo Dutti 32) is staged
  to improve verified source-photo attribute coverage. **Do not call it
  completed** until worker/DB receipts prove it.
- Lacoste HTTP 403 remains blocked. Merchant image hosting, third-party
  model usage rights, licensed photo-first cache, available sizes and
  delivery cannot be inferred from successful offline feature extraction.

## 2026-11-18 noon+ — photo-verified source evidence and honest world-class fashion gate

- Source visual enrichment confirmed 512/512 exact-current photo records:
  510 worker-success and two honest `no_confident_facets` cases, four
  GitHub workers SUCCESS. Active visual coverage: top 2,638/3,082,
  outerwear 1,838/2,117, bottoms 1,435/1,672, shoes 1,643/1,649.
- Repeating Work with the previous unchanged deployed creative algorithm
  **after** image enrichment still gave **2/11** photo approvals and 2
  native four-SKU cards. Source data recovery is necessary but insufficient.
- Staging new creative commit `adde1c8`: both independent designers see
  provenance-separated source-text hints and real confident photo features
  and are asked for complete-look scene/texture/proportion judgment;
  **252/252 focused tests, both CI SUCCESS; Render LIVE**.
- Authenticated native Work `adde1c8`: 3/7 photo-approved, one feasible
  triple, 2 shown. Authenticated Date night: 7/10 photo-approved, FOUR
  feasible triples, but visual jury chose **2**. Actual photo cards were
  two variants of blazer/trousers, NOT sufficiently differentiated.
- This reveals the enduring creative ROOT CAUSE: selecting many attractive
  **individual** four-piece candidates does not yield three genuinely
  different, occasion-right *whole-portfolio* designs. The independent
  photo editor's two-card veto is working as intended; never change
  its threshold or add an unapproved third. The next change must
  improve the original source-bound design DECISION across 3 artistic
  directions and prove it via same-source photo QA, not extra garment
  regex lists or fake images.
- Complete evidence and QA distinctions:
  `docs/reports/stylist6-real-iphone-owner-reference-20261118.md`.
- **NO production rollout** and no claim of an excellent personal
  stylist until authenticated repeatable 3×4, human-blind 30-case
  >=70% S6 preference, and retailer rights/stock/size gates.

## 2026-11-18 — independent designers must explore different REAL garment architectures

- The live Date-night source/photo editor rejected an eligible third look
  because the two displayed source-authentic looks both repeated
  **blazer + trousers**, despite different top and footwear merchandise.
- The second cross-lab AI stylist is now presented with a **dynamic,
  source-authentic complementary product menu** when the first studio
  repeatedly selected the same construction. The menu only withholds
  that over-represented subtype from the second studio when >=4 genuine
  alternatives from >=2 other known constructions exist for that role.
  There are no item-name regex bans, no invented garments and no changes
  to the first studio's choices or full-photo/editorial acceptance.
- Sparse source categories are unchanged; the alternate studio can still
  design looks within actual available inventory. The first model's
  approved blazers remain eligible for the final collection.
- Source menu behavior, abstention and identity preservation are directly
  regression-tested; **real native iOS aesthetic effect is not yet proven**.

## 2026-11-18 — dynamic exclusion of repeated source subtypes REVERTED

- Staging native Date-night canary of `6d6539b`: 8 genuine
  Nemotron+Kimi proposals, 7 independent original-photo evaluations,
  **0 photo approvals, 0 displayed outfits**. The second-studio
  dynamic construction-exclusion experiment therefore demonstrably
  failed its first live quality gate, even though 253/253 mocked
  local tests passed. Do not conflate source-item validity or CI with
  actual fashionable combination quality.
- The experiment restricting the alternate studio's product menu has
  been **reverted immediately** to the previously deployed, photo-tested
  `adde1c8` creative logic. Future source exploration must not
  suppress the actual possibilities of a creative artist merely to
  satisfy a subtype-diversity heuristic. Human reviewers must assess
  three excellent looks as a collection.
- Keep the confirmed 512 photo-profile recoveries and provenance-
  separated source text/image facts. Do NOT promote to production.

## 2026-11-18 — independent PHOTO jury for two looks: actual iOS QA, not aesthetic acceptance

- Exact documented evidence: `docs/reports/stylist6-live-visual-qa-20261118-followup.md`.
- Live staging `2d44a18` Date-night and Work both returned **2 four-item looks**. Work used independent Nemotron+Kimi source authors and original-photo jury; **3/11** photo approvals, 2 displayed, and still repetitive two-blazer construction. Date approved **2/3**, text-editor fallback when primary photo VLM timed out with only two approved looks and the second image-capable judge was skipped. No causal fashion win is proven across randomly changing source cohorts.
- Code `92c3a37` removed the unjustified **>=3** final-portfolio prerequisite for the *second independent original-photo judge*; all editor choices stay within already source-/budget-/original-photo-admitted F-token portfolios. Targeted tests and full **542/542** Stylist6 and **42/42** backend tests pass, unchanged S5 auth/quota hash.
- New LIVE Render staging `dep-dbuldc2jnfac73djv2dg` on SHA `92c3a37`: authenticated real iOS Classic / New / Date night, **8 AI proposals, 2/9 original-photo approvals, final 2**, 55,019ms. Original JPEG editorial board `ready`; primary visual jury **applied**, independent different-lab true-photo jury **agreed** (2-look size) despite **zero** feasible triples. This repairs the missing independent PHOTO-curation behavior but not the weak initial fashion composition.
- Production Stylist 5 remains unchanged, **NO-GO**. Merchant original-photo hosting/inference licenses remain unverified (zero READY rights-cleared cached original photos); visual-first SKU creation not live. Do not treat image-feature coverage, synthetic tests or AI editor agreement as a demonstrated excellent personal fashion stylist.
- Prioritize rights-cleared source-photo creative decisions and **same fixed inventory** comparative human taste evaluations; do not revive the failed hard subtype exclusion or lower photo-quality standards. The owner requires beautiful, genuinely different 4-SKU looks, not 3 output cards.

## 2026-11-18 — actual source-visual evidence enters INITIAL art direction (not regex fashion rules)

- Initial high-level AI Creative Director previously saw type counts and text moodboards only, while source-current visual item evidence and the user's explicit look feedback arrived after the three initial art directions. Root-cause change `963333af` supplies a bounded anonymized **verified garment visual atlas** and sanitized explicit whole-outfit likes/dislikes to the FIRST AI creative decision. No fabricated item/color, source leak or per-brand rules. Original-photo licensing status unchanged.
- Dedicated tests 544/544 and portable Node tests 42/42 PASS. Render staging deploy `dep-dbull2lckfvc73ciu56g` LIVE. Actual authenticated iOS Classic/Work: Creative Director consumed **23** current source photo-feature examples, proposed 6 source-admitted four-item looks; 4/11 independent full-photo approvals, 2 displayed, ~52.8s. Classic/Date night: consumed **29** verified examples; 8 proposals, 4/11 original-photo approvals, 2 displayed, ~59.8s. Both real photo editors ran and four-real-product contract preserved.
- Fashion goal still **NOT achieved**. The outfits remain too similar/office-influenced for three outstanding independent directions. New requests vary the catalog discovery cohort, therefore no causal aesthetic win can be claimed from comparing live counts. No production merge/promotion, no changing critic thresholds or implementing forced variety/type bans.
- Evidence/next controlled creative quality experiment: `docs/reports/stylist6-visual-atlas-creative-20261118/README.md` (immutable source-SKU A/B plus actual rights-cleared product photos and three-reviewer blind human fashion gate).

## 2026-11-18 — same-source real fashion A/B, not another garment ban

- Architectural improvement `0157f5e`, restored in `4b2442e8`: the first AI plans four visually related actual garments including a **meaningful** layer, then a second SKU artist receives three actual catalog-source-verified opaque P-token OPTION menus bound to those initial fashion stories. No fashion subtype/style bans, forced palette or softened original-photo quality floor. The second independently trained creative model can propose unseen real merchandise alternatives. Production Stylist 5 unchanged.
- Same-source QA enabled only for authenticated 72s Render staging via `159a322` and signed route `5d26fbd`, `STYLIST6_STABLE_COHORT_ENABLED=true`; keeps same user/style/occasion/weather/budget source **sampling seed** across new paid generation request IDs. Default behavior, per-user private scope, quota, source, budget, rights, stock and photo critic continue to apply. **Not** a physically immutable database/photo snapshot; merchant data can still change.
- Native iOS Classic/New/Date-night A/B with `stableSourceCohort:true`: A1 on exploratory source-grounded director **3 displayed / 4 of 7 photo-approved**, ~52s; B previous independent director (temporarily reverted creative feature) **2 displayed / 5 of 11 approved**, ~62s. Both proper four real goods and original-source photographs. A2 on restored source-grounded director, LIVE staging `4b2442e8`, **3 displayed / 5 of 7 photo-approved**, ~49s; independent photo jury unavailable, original photo jury applied. Exact data: `docs/reports/stylist6-source-grounded-creative-ab-20261118.md`. Full current tree **546/546 S6** and **42/42 Node** tests PASS.
- **AESTHETIC QUALITY STILL NOT PASSED:** the three actual native A2 looks visibly overuse dark knit / dark layers / black-navy trousers and loafers. An acceptable third photo-edited card is not equal to a genuinely different beautiful look. One-shot A/B and A2 count repeat are not a 30+ held-out blinded human taste benchmark. Do not declare owner acceptance, merge/promote S6, or invent visual licenses.
- **Next big milestone:** rights-cleared **genuine original-product-photo-first creative decisions**, not more hand-authored garment pair bans. Move beyond merely identifying verified photo features of individual garments: make autonomous fashion creator evaluate the **actual four photographed source SKUs as an integrated complete look** BEFORE deciding and then personally calibrate human aesthetic preference on fixed SKU/photo contexts. Source license/rights-evidenced photo cache currently has no ready entries; secure that source before external artist photo transmission. Independent human comparison of S6 vs S5 >=30 held-out contexts, 3 blind reviewers and >=70% S6 wins remains hard release gate.

## 2026-11-18 — source-current visual SKU options + real failure of serial initial creator

- `7e0d0c1`: first AI's three art-direction source menus now include a **second truly source-current photo-different actual item per garment role when such inventory exists**, bounded 12 source products per role; no hard outfit color/subtype rules. Date-night REAL iOS **3 displayed, 4/5 photo-approved, 46s**. Work REAL iOS **2 displayed, 4/11 approved, 53s**. Both four genuine products per card; Date still retains a generic nearly black third look. A three-card success alone is NOT fashion acceptance.
- `34a8574`: isolated 72s photo editorial late missing-third correction can spend the previously stranded ~17.5–22.5s, and gains an optional independently photo-vetted extra three candidate slots. No skipping original source/hard/photo checks and no forced third. Full current tree **548/548 S6** and **42/42 Node** PASS, LIVE Render staging `dep-dbums9m7bikc73ae7fjg`.
- Negative actual iOS Classic/Work results on the latest staging: **0/3** original-photo approvals and zero displayed, then repeat **1/3** approved and only one displayed; in both cases Nemotron alone supplied three initial real 4-SKU looks and the photo jury legitimately rejected the rest. These attempts did NOT reach the new late third-idea editor, because no two excellent outfits existed. Never claim it is proven helpful by those runs.
- Critical next *architectural* intervention, not one more mini-rule: initial two different-lab AI fashion creators must run **concurrently**, preserving an early independent creative source even when first designer runs late and produces only three acceptable-format candidates. Merge only real SKU-disjoint complete outfits, reapply source/price/availability/whole-photo gate and evaluate capsule style from true photos. Only then test on frozen context, evaluate visually as a human, and run independent blind taste preference gate. Original photo-cache commercial rights still unverified; do not claim original-JPEG first-pass AI.
- Exact evidence and owner NO-GO: `docs/reports/stylist6-real-visual-choice-and-creative-budget-20261118.md`; production Stylist 5 unchanged.

### 2026-11-18 experimental photo-budget rollback (do not count as styling success)

The 72s staging late editorial creative-budget change `34a8574` was unit-tested but **NOT triggered** by two real Work 0/1-photo-approval canaries. Because it adds optional model/photo evaluation cost without evidence of a net personal-fashion quality win, it was reverted as code `533b3826`; retained `backend/stylist6` and `backend/stylist-node` source equals the previously real-iOS-tested `7e0d0c1` source tree. The photo-approved 3-outfit Date-night candidate from `7e0d0c1` remains the current positive staging reference, but it still has a generic third dark look. **Next actual fashion solution is independent parallel first-pass artists, not a fourth model veto, post hoc extra spending, or forced third card.**

## 2026-11-18 — first-pass portfolio taste STILL the blocker, verified full-photo veto recovery

- Real 72s staging `7e0d0c1` expanded the FIRST creative composer options to up to two genuine **visually different photo-observed** source SKUs per garment role and commission, not new hard fashion rules. No photo-veto weakening, inventing a fourth SKU, or enlarging the per-slot token cap.
- The split **CONCURRENT** 120B + different-lab Kimi fashion author experiment (`dccf0be`) increased genuine source designs to 8, but actual iOS Classic/Work got **0/8** approved first run, **1** displayed next; Classic/Date night **2** displayed. This is not a better human stylist. It was REVERTED in `57e926f6` while retaining the real structural correction `bb475e3`: after **zero** original-photo approvals, a new creator can now use previously UNSEEN prepared authentic SKUs and original-photo reject observations, with all independent photo/source/budget gates intact. This new recovery is proven by code integration tests; **no live zero-rescue success claim yet**. The earlier late-budget creative experiment (`34a8574`) likewise remains reverted.
- Final EXACT LIVE Render staging `dep-dbun865g1s2s73cchpu0` SHA `57e926f6`: real authenticated iOS Classic / New / Work **2/5 photo approvals, 2 shown, 51.3s**; Date night **2/5 approvals, 2 shown, 41.5s**, all four REAL source products, independent JPEG portfolio visual editor `applied`, no fake third. Full code **548/548 S6** and **42/42 Node** PASS. S5 production unchanged. Full evidence: `docs/reports/stylist6-creative-fashion-recovery-20261118.md`.
- **AESTHETIC QUALITY NO-GO:** date cards STILL reuse light blue shirts + conventional blazer silhouettes; one was captioned `Precise office tailoring` in DATE night. AI visual agreement does NOT substitute for actual personal fashion taste. User needs THREE creative exceptional varied shoppable outfits, not three output tabs. Focus on licensed original-product-photo-first Creative Director and human blind preference learning, not new regex subtype bans or arbitrary per-axis thresholds.

## 2026-11-18 — licensed original-photo FIRST eligibility made executable; art-director invalid-plan recovery

- Source checks on linked Supabase: `shop_catalog` **8,554 active real offers** but **0 original photo cache records** and **0 independently verified original-product-photo rights grants** in `catalog_photo_cache_v1`. This does not establish that no contracts exist elsewhere. Do not enable photo-first live inference, fake licenses or infer rehosting permission from public URLs. The owner goal is still 3 exceptional truly different shoppable **4-real-product** looks. Production Stylist 5 untouched.
- `7bf4e2a`: when lawful cache records appear, the staging creative composer now allocates **at least 3 exact current source-hash verified and rights-authorized photograph SKU choices per garment role** within the SAME 12-per-role / 48-token menu, even if those SKUs fall outside the generic 650-item source shortlist. Source/budget/current merchant identity gates unchanged. Generated synthetic-fixture original-photo board checks all four columns with 12 true authorized mock photographic positions, and a missing-license board abstains. With 0 real grant rows, the live original-product photo-first route stays **NOT_RUN**. No new color or subtype bans.
- Kimi-FIRST independent first-author canary `0565271` returned no valid Kimi primary creative result; Nemotron recovered and delivered 2 Date outfits/2 of 5 photographic approvals. **Reverted** in `adc1e823` rather than adding provider time or claiming success.
- A real `7bf4e2a` Date night iOS run exposed `creativePlanStatus:invalid_plan` / `creativePlanCount:0` and only **1/5** photo approval. `468c7f2` adds one bounded independent 30B creative-first rescue only when the 120B art plan is entirely invalid, using the SAME real source subtype and authenticated photo feature evidence, never fabricated catalogue items; full integration tests prove source-valid fallback but subsequent live 120B planning succeeded without needing backup. Final `468c7f2` LIVE staging authenticated Date **2 shown / 3 of 7 photo-approved / ~59s**, Work **2 shown / 2 of 4 approved / ~52s**, 4 real SKUs each, `creativePlanCount:3`, first-photo mode NOT_RUN (no rights), photos checked by later independent editorial. 557/557 Stylist 6 and 42/42 server tests PASS.
- **AESTHETIC ACCEPTANCE: NO-GO.** The two Date photographed looks still share conservative light-top/brown-tailoring construction. No fabricated third, no photo-threshold relaxation, no unlicensed first-photo images. Exact evidence: `docs/reports/stylist6-photo-first-licensed-art-20261118.md`. Next task is **lawfully licensed actual original garment photos BEFORE the first creative SKU decision**, then blinded human comparative outfit taste evaluation on held-out same-source contexts. Earlier conjectural parallel-model suggestions in the historical log are superseded by real canary negative results and their reverts.

## 2026-11-18 afternoon — real photo-correction canary did NOT meet human fashion target

- Live Supabase: **8,554 active offers, 7,556 source-current visual profiles, 7,525 768d visual embeddings, 0 rights-cleared cache JPEGs, and 0 persisted human look-feedback rows**. Stored visual observations do not confer photo inference rights and do not replace human taste labels. Previous `9ca61c0` evidence-first 48-product menu passed code tests but real Classic/Date gave **1/4** photo-approvals, only **1** shown; reverted in `ae518aab`.
- An autonomous 30B **PHOTO-CRITIC-INFORMED SECOND designer** (`604436e`) can now transform the independent JPEG critic's actual negative outfit relationships into 3 REAL alternative four-item compositions within ~1.3s; all 3 then go through real original-photo approval. Live Classic/Work **3/8 PHOTO approved, 2 shown**, of which **2 corrective outfits** were photo-approved; **0 independent three-look portfolios** were feasible. Classic/Date **1/8 approved, 1 shown**, 0/3 corrected outfits approved. This is working PHOTO-EDITORIAL engineering, NOT owner acceptance. Fast model is now kept **OFF** by staging env flag and code remains testable.
- Enlarging second AI's choice set to wholly unseen real merchandise even after 1+ initial approvals (`b7da322`) passed source-true integration tests, but real Work again **2 shown** after 3/6 photo-approved, and Date only **1 shown** after 1/5 photo-approved plus revision timeout. No three excellent unique looks. Reverted in `0a22bc58`, not promoted. Exact report `docs/reports/stylist6-20261118-original-photo-feedback-live-qa.md`.
- Current safe **LIVE isolated staging code** deploy `dep-dbuorisipgvc739j0ae0`, SHA `0a22bc58`, both negative canary flags `false`; last full code **559/559 S6** and **42/42 Node** tests PASS. Production S5 unchanged, PR #15 unmerged, **Stylist 6 fashion quality NO-GO**. Next actual goal is first licensed JPEG creative decision plus independently human-calibrated whole-portfolio taste — not arbitrary product rules or padding.

- Final afternoon full QA: added independent **timeout** fallback for first 120B art director in `2dfc627d` (previously only `invalid_plan`), reproduced exact local timeout and global-abort safety in 2 new tests; total **561/561 S6 + 42/42 Node** PASS. `81f3cf0` LIVE Render `dep-dbuothjm8hqs73b0r4h0`, both failed experimental flags false; real iOS Work on this code yielded only **1/5 original-photo-approved and 1 displayed**, 6 true source candidates, 3 original creative directions, art backup not used. **NO-GO** still. This is an architectural reliability improvement, not an established artistic victory.

## 2026-11-18 — partial fashion art repaired, still no three excellent native outfits

- First-art `2dfc627`: one independent source-grounded 30B model can now rescue a locally TIMED-OUT primary 120B Creative Director, but never after the 72s request itself has ended. Tests preserve actual source subtype and photo quality gate, not fake model output.
- Creative art completeness `58a6aa19`: previously a REAL Date night left `creativePlanStatus:planned_partial` with **only 2/3 directions**. Instead of a small fashion-rule workaround, an independent 30B agent is now allowed to see those two anonymized, source-validated artistic directions and propose a missing third. Both first decisions are preserved; missing, duplicated or unavailable backup never manufactures the third. Full tests **563/563 Stylist 6 and 42/42 Node** PASS.
- REAL final iPhone `58a6aa19` Date night **1 displayed / 1 of 3 original-photo approvals, 3 initial creative concepts, backup `not_needed`, ~36s**. Prior `81f3cf0c` Work **1 displayed / 1 of 6** and Date night **1 displayed / 1 of 5, only two original art concepts**. New partial-plan recovery is proven by integration tests but DID NOT FIRE in final real canary; it is NOT an aesthetic success claim. Full comparison + exact sha: `docs/reports/stylist6-partial-art-completion-native-check-20261118.md`.
- 30B post-photo revision can produce real new garments, but genuine photo evaluation is still rejecting most combinations. 3 cards are NOT an acceptance goal by themselves; no proof of three truly beautiful distinct outfits. 0 current first-photo externally licensed SKU cache rows; first visual designer still sees source-current image FEATURES, not licensed original SKU JPEGs. Source/legal rights and real human style benchmark are the decisive blockers. **STYLIST 6 PRODUCTION NO-GO; S5 untouched**.

## 2026-11-18 — real first-artist artistic content and explicit human taste, both independently measured

- **`3c8f325`** fixes a deeper creative contract error: previous `artDirection:''` in the required JSON schema was accepted as one of three fashionable AI designs. Runtime now requires >=85 safe descriptive characters of actual garment-to-garment artistic intention (within the prompt's existing 90–230) before a direction is accepted, without new color/type restrictions; known safe genuine concepts remain and independent AI fallback can fill missing direction. 564/564 Stylist 6 and 42/42 gateway tests pass.
- Real authentic iOS Classic/New/Date night on LIVE `3c8f325`, original source catalog/photo critic: first art `planned`, **3 of 3 substantive artistic descriptions**, 6 composed/5 source admitted, **1 of 5 photographic approvals, only ONE shown**, ~36s. Native `Textured tailoring` light-blue shirt/dark tailored jacket still a generic single concept, not a stylist-quality win. Model PHOTO revision timed out. Thus *better plan wording alone is not enough*.
- Read-only Supabase audit: **ZERO** real user `stylist_look_feedback` rows, despite thousands of product visual profiles and 46 manually captioned inspirational flat lays. These are NOT blind human preference scores. **`451d007`** exposes the existing RLS-protected explicit thumbs controls near each result's title rather than hidden after the entire four-item grid and tips. Nothing auto-rates; 9/9 feedback unit tests and TypeScript compile pass. NEW NATIVE SCREEN build unverified; do not claim deployed or live user feedback.
- Independent original merchant photo first-pass rights-verification records remain ZERO, so first artist still sees source-current extracted features and inspiration text, not source-original image JPEG. Next actual goal is authorized original-photo-first generation + real human preference learning across complete four-item looks and entire 3-look collection. Neither three accepted prompt descriptions nor the existing independent photo reviewer qualifies for the owner's fashion excellence acceptance. **S6 PRODUCTION NO-GO; S5 unchanged.** Exact evidence in `docs/reports/stylist6-creative-art-and-real-taste-signal-20261118.md`.

## 2026-11-18 — creative concept continuity through REAL SKU design; first native three-card photo success does not prove taste

- `43f4173` repairs a substantive original FIRST-artist defect: after AI art-directing source-SKU outfits, `sourceVerifyComposedOutfits` replaced every authentic creative concept with a self-fulfilling generic brief. Original AI-led concepts 1–3 now retain their trusted, substantive artistic intentions and canonical `ai_director_1/2/3` identities through source admission, soft concept-fidelity and the paid PHOTO audition. Independently invented looks 4–6 and post-photo rescues are NEVER falsely attributed to the original director. Physical four-item source, exact availability/budget, original-image rights and visual quality vetoes unchanged. 565/565 S6 + 42/42 backend tests pass.
- **Authentic native Date night on LIVE `43f4173`:** six initial source looks, five admitted, **3/5 independently PHOTO-approved, THREE displayed**, ~50s. I visually inspected ALL THREE: two near-duplicate black knit/grey layer concepts; a third navy blazer/trouser look remains conventional. Read-only completed-deck diagnostic: **two displayed outfits are `smart-knit-layering` (`signature`/`refresh`), only one `grounded-separates` (`discovery`)**; only 2 of 3 real Creative Director concepts were shown to the PHOTO judge. **DO NOT CLAIM three genuinely different great fashion looks.** Independent cross-model photo-editor vote was unavailable.
- **Authentic native Work on SAME revision:** four real looks, PHOTO approval **2/4** but only **ONE displayed**, photo portfolio editor 0 eligible independent pairs/triples, secondary artist timed out, ~40s. So the Date one-off isn't proof of consistently exceptional or reliably three-look artistry. Goal remains full whole-outfit aesthetic diversity and human-calibrated taste, not 3 syntactically valid cards. Pilot original-photo rights cache still empty; no manufactured grants or fake customer ratings. Exact evidence `docs/reports/stylist6-creative-intent-source-photo-portfolio-20261118.md`. **S6 PRODUCTION NO-GO, S5 untouched.**
