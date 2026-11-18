# Stylist 6 — native iOS original-photo fashion acceptance QA · 2026-11-18

**Verdict: real staging engineering flow works, fashion goal NOT MET, production Stylist 5 NO-GO.** Do not confuse independent MODEL photo approvals with human blind aesthetic acceptance. Goal remains up to THREE exceptional, distinct personalized looks with FOUR sourced merchant items each and honest stock/rights.

## Environment and source pin

- Actual authenticated **iOS 26.5 simulator**, `com.aiwardrobe`, isolated signed-in QA identity and one Generate action per request, Maestro completed exit 0 on BOTH sessions, genuine 72s server-owned Render compute and Supabase JWT/HMAC staging gateway.
- Compute pinned EXACT `a99c268e711c7df98bd18dbc2bb36b99e56338ba`, deployment `dep-dbut6888v5cc73a3pekg` **LIVE**. QA native client JS was loaded by localhost Metro from a detached, clean worktree of exactly that commit; pre-existing primary developer worktree and production S5 untouched.
- `stableSourceCohort=true` in both server receipts. Their requests differ by occasion, so this is NOT a controlled before/after same-SKU fashion benchmark. Private screenshots and product identity retained on the Mac; NO merchant JPEGs or private IDs committed.
- The licensed original-photo *FIRST-PASS* composer was correctly disabled: `skuCompositionPhotoFirst=not_run`; public merchant image rights to first-party host and transmit to generative provider have NOT been proven, cache has 0 ready grants. Independent AFTER-composition permitted original-product-photo critic runs normally.

## Real native Classic / Work — original source photos

Render `stylist6_safe_visual_quality` **2026-11-18 16:41:47.886 UTC**:

| Field | Verified result |
| --- | --- |
| First source AI producer | Nemotron, parallel AI-FIRST; 6 proposed / 6 source-admitted |
| Independent whole-look original-photo review | 6 attempted; 3 individually approved; 3 rejected (2 model_veto; 1 style_mismatch) |
| Final original-photo portfolio board / Nemotron visual jury | ready / applied; feasible photo triple existed; human/AI curator chose TWO |
| Independent different-lab Kimi photo jury | unavailable (not a vote) |
| Original source-grounded art plans | only TWO; independent 30B fallback timed out |
| Shown to actual iOS | TWO complete **4-real-item** cards; fourth missing/repeated/synthetic: 0 |
| Render elapsed | 52,231ms |

Actual iOS visual audit: (1) pale blue shirt, brown tailored jacket, grey trousers, brown loafers, labeled `Clean separates`; (2) light top, navy double-breasted outer layer, dark trousers, black Chelsea boots, labeled `Coat & Trousers`. Professionally plausible but only TWO directions, neither establishes clear fashion superiority or three outstanding distinct looks.

## Real native Classic / Date night — independent whole outfit photo failures

Render `stylist6_safe_visual_quality` **2026-11-18 16:51:23.522 UTC**:

| Field | Verified result |
| --- | --- |
| First creative planner | THREE original detailed art directions; independent fallback not needed |
| First source AI producer | Nemotron, 6 proposed / 5 source-admitted |
| Whole-look source-photo review across first and photo-revision waves | 8 attempted, only **1 approved in reported merged review** |
| Photo rejection reasons | below_quality_floor, style_mismatch, occasion_mismatch among valid verdicts |
| Bounded independent photo-informed SKU correction | **ran** using previously unseen source-verified merchandise; three re-reviewed, one separately approved per revision diagnostic |
| Result to actual iOS | ONE complete **4-real-item** look; never invented a weak second or third |
| Final photo portfolio editor | not_run because no valid 2-look portfolio |
| Render elapsed | 45,428ms |

Important nuance: the source-photo revision `photoRevisionStatus=composed`, `photoRevisionAttempted=3`, `photoRevisionApproved=1` is proven in the FULL structured log, despite broad first-page queries appearing to show `not_run`. The final count remains one. Thus the model made new source-authentic merchandise proposals but did not achieve a publishable varied collection. This is a FIRST COMPOSITION/WHOLE-LOOK coherence failure, not an iOS card-layout bug.

## Bounded root-cause code response (after real native tests)

The source-menu fact diversity changes `2b26cfaf` and `749c98d9` plus tests `a99c268e` expanded anonymous image-observed garment construction attributes without banning categories or lowering quality. A fixed read-only 60-item-per-role live Supabase snapshot yielded more photographic facets in the SAME 48-product P-token menu, but these live iOS outcomes prove fact exposure by itself did NOT solve taste.

The Work receipt exposed a further exact-index bug: when a 120B Creative Director's three-slot plan is partial (e.g., original plan IDs **1 and 3** survive), the source SKU composer formerly rebuilt their look numbers from compacted array indices, silently relabeling #3 as #2; the final photo editor also required plan ID to equal its array position. Original artwork could disappear or be attributed to the wrong actual outfit. This is general model-output integrity, not a garment ban.

Corrections in follow-up staging PR:
- `a5dacd4e`: use source-valid original `ai-director-[1..3]` IDs as artistic commission identity through grounded exact-SKU P-token planning; legacy synthetic no-ID directions remain backward compatible.
- `c8faf2d8`: original final PHOTO portfolio jury also maps art to ORIGINAL plan IDs even when source array skips an index; duplicate/spoofed identities do not overwrite original art.
- `bc408b23`: when creative plan #2 is missing, the original 120B **SKU creator itself** receives a bounded, explicitly numbered new artistic commission from existing P-token source menu, without inventing a product or demanding three approved looks.
- `1aebccc4`, `88c0b44e`, `2d2bbd50`: isolated >=72s staging concurrently permits 30B missing-art backup 8.5 seconds, clamped to <=9s; previous cap was 6.5/7s. The 72s global request deadline, source admission, human consent, photo originality, data privacy and production S5 stay unchanged.
- Source identity, anonymization, bounded fallback and real photo portfolio tests in `1812a302`, `c0b863c6`, `28744f24`.

A separate provider probe on a **SYNTHETIC** source-only garment catalog returned 3 valid 30B art directions in ~4.3s at an explicit 9s deadline. This shows the same model can do the task in a controlled probe; it does NOT prove the longer fallback has cured authenticated Work.

## Remaining release gates

1. Pin the exact successful CI code SHA to LIVE Render; refresh authenticated native Work and Date night at the same allowed SKU opportunities where possible. Monitor `creativePlanCount`, successful index preservation, exact source AI decisions and original PHOTO approval/veto, 4-SKU completeness, final distinct real looks, latency and all abstentions. Do not claim a causal aesthetic gain without identical frozen SKUs and photo side-by-side evidence.
2. Investigate why independent Kimi K3.1 photo editor remains unavailable: a real provider synthetic JPEG probe on 2026-11-18 timed out after 12.5s; /v1/models advertised Kimi but availability is not success. Gemma 3 4B/12B synthetic JPEG endpoints returned 404, Gemma 4 31B timed out; do not invent a second working vision jury.
3. Actually improve FIRST-PASS complete-look aesthetic composition. Longer model deadlines, more SKU tokens and extra photo judgments are not taste acceptance.
4. Validate blind human S6-over-S5 >=70% in >=30 matched held-out scenarios with >=3 reviewers, with original four-item photos, zero source/budget/weather/anchor/privacy errors, legitimate merchant/rights/size/delivery claims and safe staging rollback before any S5 production promotion.

**No subjective fashion improvement, human preference score, uploaded copyrighted product JPEG or rights authorization is asserted by this report.**
