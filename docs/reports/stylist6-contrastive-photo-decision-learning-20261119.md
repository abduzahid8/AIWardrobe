# Stylist 6 — contrastive photographed outfit learning, actual native QA (19 Nov 2026)

**Goal unchanged:** a genuinely autonomous personalized fashion expert creates **three beautiful, mutually distinct, coherent original looks, each exactly four real available SKUs**. Release status: **NOT MET / PRODUCTION NO-GO**. A green test, more AI calls or a forced three-card output is not a fashion-quality win.

## Implemented: decision learning from comparative, genuinely photographed full outfits

Exact server code **`1339be8453beb6ad0b01dc33260ccaccb23f578e`**:
- `c7f509bf`: the **FIRST**, realistically exercisable photo-informed source-SKU correction receives both the genuine independent photographic **approved whole-outfit evidence** and **rejected whole-outfit evidence** (anonymized four garment role characteristics, colors, observed relationships and bounded independent quality axes). Previously it received failure descriptions and accepted garment subtype names, but not what in the actual photographed successful outfit worked.
- `e023f9eb`: remove specific blazer/cardigan avoidance and avoid-accepted-architecture instructions that acted like arbitrary aesthetic rules. Instead the composer is instructed to learn **complete-outfit** relationships from contrasting positive/negative actual-photo observations, determine its own independently creative source SKU choices and respect explicit user taste.
- `ff9ebf13`, `1339be84`: regression test checks actual first photo-revision receives **one independent original-photo positive** and the real negatives, including whole-item relationship text, and verifies the autonomous model sees sanitized positive examples without source/user identifiers, URLs or fabricated photos; the prior prompt-specific assertion is made behavioral instead of matching an exact catchphrase.
- Source/auth/weather/budget/merchant/4-SKU/original-photo-jury/owner's dislikes remain hard truth, not stylistic bans. The new first-revision evidence uses **no additional provider call** and no extension of the total request deadline.
- Dedicated [GitHub Actions 44462337650](https://github.com/abduzahid8/AIWardrobe/actions/runs/44462337650) completed SUCCESS on exactly `1339be84`.
- Render isolated staging **`dep-dbv86rmk1f9s73bp3fb0` LIVE**, exact `1339be84`; production S5 remains untouched.

## REAL iOS 26.5 authenticated staging evidence

The iOS QA simulator and Mac Metro performed authenticated **one Generate request per occasion** with explicit source "new clothing", cloud S6, real original-photo quality on, through Supabase JWT/HMAC to current Render. Both Maestro flows finished exit 0. The root Render receipts, not offline simulations, are the denominator for independent photo approvals.

| Intent | Source AI | Proposed | Actual photo attempts | Photo approved | Displayed | Compute |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **Classic / Date night** | Nemotron + Kimi initial | 8 | 7 | **1** | **1** | **42,838 ms** |
| **Classic / Work** | Nemotron + Kimi initial | 8 | 8 | **3** | **2** | **50,296 ms** |

Both `catalogFourItemContract=true`, `catalogFourItemDropped=0`, `skuCompositionPhotoFirst=not_run`, `creativePlanCount=3` and `stableSourceCohort=true`. These are two **different occasions with live source inventory**, not a matched frozen before/after or a human blind fashion evaluation.

- **Date night:** after six originally admitted source looks, the early independent original-photo critic rejected five original combinations; bounded contrastive correction **did run** and was rephotographed for **one** additional item combination, with **0** extra approvals. Merged critique 1/7, no eligible final photo portfolio. Real rejection reasons: 3 `below_quality_floor`, 3 `style_mismatch`, 1 `approved`. Artistic backup supplied missing plans, but no source/photo-approved third.
- **Work:** 3/8 independent photo approvals; a feasible three-look portfolio existed but the actual photo collection editor chose **two**. First early retry did not run in this context; the later editor requested three additional authentic source-SKU concepts from unseen inventory but **0/3** earned new photo approval. Different-lab independent photo jury `unavailable` (do not claim agreement). Native UI revealed two still-conventional tailoring looks, including a light top/brown tailored jacket and another light top/navy tailored layer, with limited visual variety.

**Factual learning:** the contrastive transfer **does** execute in the real Date flow, but this run demonstrates **no aesthetic quality win**. A still text-only original SKU composer receives sparse SigLIP-derived garment facts, not a model view of actual original whole garment photographs before pairing. Thousands of catalog SKU profiles are sparse on crucial construction, drape and length traits; a better-written text prompt cannot infer unseen fabric/fit. The hosted photo critic sees the original four-product board **AFTER** composition. A first-pass original-photo board cannot be activated by inventing licensing: `catalog_photo_cache_v1` had 0 granted rights-bearing READY rows in the last verified snapshot.

## Next decision-learning architecture — no new garment rules

1. **Visual-before-choice teacher**: establish explicit rights to cache/process the original merchant photographs for a bounded real SKU cohort, OR improve approved local/on-device multimodal image understanding without unauthorized external photo transmission. Only claim image-first if original pixels were actually seen by the creative model. Persist provenance and confidence, leave unseen details UNKNOWN. Don't treat prompt hints as photographs.
2. **Train on real preferences, not generic heuristics**: collect privacy-safe full-four-SKU photo critic *comparison* outcomes with complete source-revision provenance, plus separately marked blind human owner/reviewer preferences. Train and calibrate a learned visual outfit-reward/preference model against held-out outfits; do not hard-code disliked colors, cardigans or shoe types from a few failed examples. Existing personal explicit dislikes remain user constraints, not learned universal bans.
3. **First-pass ranking/assembly**: use that trained image-aware preference signal when choosing real SKU combinations **before** running expensive final original-photo verification; then let the independent photo jury remain the final veto and let the AI compare capsule-level three-look diversity.
4. **Evidence threshold:** frozen same-SKU same-context S6 vs S5 side-by-side original four-product boards, >=30 held-out situations, >=3 independent blinded humans, >=70% S6 wins, zero source/privacy/budget/stock/4-item violations. Neither of today's results meets this standard.

**No product photo bytes, user secrets, source item identifiers, external image hosting rights or human preference labels are fabricated or committed. This report records actual failures rather than replacing them with more rules.**
