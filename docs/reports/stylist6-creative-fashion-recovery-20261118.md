# Stylist 6 — 18 November 2026, independent AI studios, full-photo veto recovery and native fashion QA

**Decision: production NO-GO. Owner fashion target NOT achieved.** Requirements remain: an autonomous personal stylist with taste, three genuinely different beautiful shoppable looks, four REAL source-valid goods per look, personalized to human intent and scalable. No invented fourth goods, hard photo veto relaxation, or generic outfit-category prohibition.

## Root cause and structural repairs

- The new original-image FEATURE-guided Creative Director (`963333a`) plans actual verified garment colors/material/silhouette evidence, but the SOURCE item composer often produced generic blazer/trouser combinations. `0157f5e` joined three artistic concepts with their real P-token source SKU options; the first artist was also made responsible for a coherent four-item outfit (including an intentional layer, never a last-minute coat filler).
- `7e0d0c1` introduced up to two different source-current image-observed real product options for the same artistic concept/garment role, where authentic verified photos support the difference. Source menu max remains 12 per role; user preferences, exact source admission, budget, original-photo critic and cross-outfit photo editor unchanged.
- Actual Work PHOTO review sometimes rejected **all original AI compositions** (0/3 or 0/8). The early photo-informed creative correction had an accidental `visualApproved >= 1` gate: when *all* photos failed, it never attempted a new design, even with sufficient request time. `bb475e3` removes this broken prerequisite **only on 72s staging** and offers a second AI a prepared pool of genuine previously UNSEEN source SKUs plus anonymized actual PHOTO rejection observations. All new combinations must be independently reviewed on real original source product images again. If they fail, zero card is correct. A deterministic full integration test reproduces eight accepted-by-source but rejected-by-photo items followed by a fresh reviewed and source-verified second wave. **Live proof of this zero-veto rescue firing is still pending** because the final restored real runs had 2 photo approvals.

## Experiment rejected: two concurrent models with split inventories

`dccf0be` tested concurrent Nemotron + independent Kimi fashion creators, assigning each genuinely disjoint physical source merchandise. Both models were requested concurrently, rather than depending on remaining time after the first response. 549/549 S6 tests and 42/42 server tests passed, and actual source candidate count increased to eight. **But actual aesthetic quality got worse**:

| Exact live staging condition | Work | Date night |
| --- | --- | --- |
| Split parallel studios, `dccf0be`, with zero-photo rescue code `bb475e3` | 0/8 original-photo approvals, 0 shown in first Work canary; in another Work generation 2/3 approvals, **1 shown** | 3/4 photo approvals, **2 shown** |

The hosted receipt recorded first-pass studio `nemotron_parallel` rather than a valid Kimi combination in the second Work and Date canaries, revealing that parallel launch alone is **not** reliable independent fashion authoring. The same 72s budget and data-dependent SKU cohorts were used. The new shared source pool was partitioned, reducing garment options per AI designer, and the increase in candidate count did not improve photo-verified fashion taste. More models and more candidates are NOT evidence of style quality.

**Action: revert `dccf0be` in `57e926f6`.** Preserve `bb475e3` independently: zero-photo-veto creative recovery does NOT depend on simultaneous twin studios. This follows owner intent not to accumulate small rules and not to ship worse art just because tests are green. The large early photo-edit time-budget experiment `34a8574` was also previously reverted (`533b3826`) after adverse native Work QA; it is NOT active in final staging.

## Exact final LIVE staging and native iOS results

**Deployed `57e926f6c4db7d925bd3efaa5b26a8683bfdadab`** on isolated Render service `srv-dbtd336i0phs73fd0sbg`, deploy **`dep-dbun865g1s2s73cchpu0` LIVE**. Supabase JWT/HMAC signed gateway, source catalogue, iOS 26.5 booted native `com.aiwardrobe` and existing production Stylist 5 unchanged. Server-owned `stableSourceCohort:true` source sampling is active for this one tester/style/occasion/budget; same discovery *seed*, not an immutable merchant inventory/photo snapshot. The exact final code passes **548/548 Stylist 6** node tests and **42/42** gateway/server tests, portable bundle ready.

| Native one-tap context | First genuine AI proposals | Original-photo critic approved / attempts | Displayed actual 4-SKU looks | Engine / delivery | Original-photo final jury |
| --- | ---: | ---: | ---: | ---: | --- |
| Classic / New / Work | 6 source-proposed, 5 admitted | **2/5** | **2** | 51.3s / 51.9s | primary visual editor `applied`, JPEG board `ready`, other lab `unavailable` |
| Classic / New / Date night | 6 source-proposed, 5 admitted | **2/5** | **2** | 41.5s / 42.1s | primary visual editor `applied`, JPEG board `ready`, other lab `unavailable` |

Both completed as paid authenticated native decks, `catalogFourItemContract:true`, `catalogFourItemDropped:0`, `skuCompositionStudio:nemotron`; each browser UI had real original merchant garment cards and external shop links. After initial photo verdict, `photoRevisionStatus:timeout` on both. Photo-informed creative revision remains unreliable in its ~4.9s timeout. Note: *two* individually approved original photos do not automatically imply three source-disjoint complete ensembles. Do not pad or lower quality thresholds.

### Actual inspected owner-facing fashion quality (not AI-agreement)

- **Work:** first `Refined Tailoring` pairs a pale upper garment with a warm brown tailored jacket, dark trousers and refined shoes; second `Coat & Tailored Trousers` uses a pale knit with a dark substantial outer layer and trousers. Both are plausible, but still conventional and not a 3-look creative capsule.
- **Date night:** first `Precise office tailoring` shows a pale-blue shirt with an ivory/light jacket and dark trousers. Second `Blazer & Chinos` shows a light blue shirt and dark blue blazer. They are not dramatically different designs and the first explicitly reads OFFICE despite a DATE setting. This is **not** excellence; model-led photo checks alone do not recognize a genuinely romantic/social collection. Merchant images inspected privately via native simulator; no copyright-uncleared photos committed.
- A suspected women's shoe in the earlier black Date card was investigated against the real Supabase product source; `Weston Loafers - Suede - Men's — Black` by Beckett Simonon is source-labelled as men's loafers. **No unsupported hard shoe exclusion** was made from a thumbnail impression.

## Next major work, no new micro rules

1. The source-first Creative Director still does **not** see full original source SKU pictures before choosing the outfit: `skuCompositionPhotoFirst:not_run`, because the independent *rights-verified original-photo cache* has zero READY licensed rows. Individual source-current photographic features are useful, but the AI does not really see garment-to-garment texture/silhouette relationships until **AFTER** assembling the candidate. Secure lawful original-image external inference rights for appropriate catalog products and hydrate a compact source-verified 4-role image board **before the first decisive AI artist**, not only for the veto critic. Do not assume authorized product display means permission for rehosting or external AI training/inference.
2. The creative system must learn full-look and **portfolio taste**, not enforce random hard subtype/palette bans. Use humans to label coherent vs office-uniform vs weak date ensembles on identical fixed verified SKU/photo cohorts. Calibrate the artist and second photo director with blind human relative preferences, not only an approval-count objective.
3. Keep the no-corner-cutting release gate: three excellent, truly different four-SKU garments when available, >=30 held-out fixed-context S5 vs S6 evaluations by >=3 independent blind humans with >=70% preference for S6, exact budget/source/merchant/identity compliance, no missing merchandise or photo-inference rights violations. **Production NO-GO** until achieved. Do not restore split AI studios or the failed late-deadline threshold tweaks merely for more cards.
