# Stylist 6 decision gate — prove fashion quality BEFORE building the new engine

**Status (2026-11-10): STOP FULL REBUILD. Research/feasibility only until a winner is demonstrated.**
**Override:** This decision supersedes the instruction in `STYLIST6_AGENT_MANDATE.md` to immediately implement the full end-to-end Stylist 6 pipeline. Do NOT merge, deploy new routes, incur paid inference, alter Supabase production/staging or train a GPU model without later authorization.

## Why we are stopping

The V5 architecture primarily combines pre-set outfit formulas + item retrieval + deterministic quality gates; the NVIDIA ranker can only reorder a short list of complete looks, with tiny utility weight. Screenshots prove some generated looks render but do not prove human fashion satisfaction.

The isolated `backend/stylist6/prototype.mjs` is **not yet a fashion solution**: its hosted visual critic and creative director are just injected interfaces; passing 8 mocked tests shows contract behavior, NOT quality. Implementing it fully without human A/B testing would recreate the same risk under a new design.

Research directions to test (not assume to work):
- ICCV 2025 Text2Outfit: joint text-driven outfit composition and retrieval using multimodal representations. Research code/training approach is **not** drop-in deployment for our catalog/no-local-GPU constraint. https://openaccess.thecvf.com/content/ICCV2025/html/Zhai_Text2Outfit_Controllable_Outfit_Generation_with_Multimodal_Language_Models_ICCV_2025_paper.html
- WACV 2024 Lost Your Style?: semantic levels of item, style and whole outfit for retrieval. https://openaccess.thecvf.com/content/WACV2024/html/Jang_Lost_Your_Style_Navigating_With_Semantic-Level_Approach_for_Text-To-Outfit_Retrieval_WACV_2024_paper.html
- 2026 Loom preprint: hybrid FashionCLIP retrieval + explicit material/occasion compatibility; reported 9.3% hard violation rate in its evaluation, so it is not production-proof by itself. https://arxiv.org/abs/2605.09830
- OutfitTransformer WACV 2023: retrieval-oriented composition representations, a candidate technique rather than a verified ready-made integrated service. https://openaccess.thecvf.com/content/WACV2023/html/Sarkar_OutfitTransformer_Learning_Outfit_Representations_for_Fashion_Recommendation_WACV_2023_paper.html

## Architectural alternatives to compare in the pilot

**A. Reference-first / lookbook-to-shop (preferred HYPOTHESIS, not final decision)**:
Curate/licence a small, high-quality benchmark set of REAL full looks from stylists and lookbooks in style + occasion groups. From a desired look, extract *outfit-level* layout, formality, silhouette, palette and complementary slot roles. Find real catalog items for each role using verified metadata and existing image embeddings **within same product role** only. Assemble candidates minimizing distance to reference while preserving hard fashion fit, budget, identity, source and season. Evaluate side-by-side against actual editorial reference and each other with a hosted multimodal critic + blind humans. References are not the items to sell; shopping output must be current catalog SKUs. Note IP/licensing before embedding/redistributing examples. Image-to-image nearest neighbor alone does **NOT** establish item-to-item compatibility.

**B. Creative-director-first / direct Stylist 6**:
Test the existing `backend/stylist6/prototype.mjs` with REAL hosted creative prompts + hosted image critic, using identical catalog snapshot and strict gates. This is a pilot only, not native endpoint/deployment.

**Control C. Current Stylist 5**:
Capture actual V5 outputs for the same inputs, not canned mock fixtures. Pin engine SHA, catalog snapshot, FX/time/locale and user preferences; do not take older iOS screenshot as an authenticated version proof.

## Pilot — a gate, not another rewrite

1. Freeze 12 representative inputs out of `benchmarks/stylist6/scenarios.json`, covering Old Money/Classic/Semi-Classic, Work and Date night, Casual, low-budget UZS and strict price constraints, wardrobe anchor and SKU source integrity. Use same full catalog snapshot and source hashes for A/B/C. Pilot results MUST be actual generated catalog SKUs and photos — no handcrafted answers falsely labelled AI.
2. Obtain a *small* reference quality set: at least 2-3 human-curated real looks per scenario (approximately 30-36 total); judge whether suitable comparable items exist in the catalog BEFORE blaming composition. Reference artwork must have a legitimate evaluation/reuse basis. Save neutral brief descriptors and licenses; do not upload third-party source images to public app.
3. Implement two **disposable offline evaluation adapters** for A/B behind a single deterministic replay contract. Do NOT wire into production, replace V5, or build iOS UI first. Validate any external hosted model's image-input capability with official API docs and a single authorized dry-run before bulk calls. No local GPU or retraining.
4. For each input, create 3 outfits per approach, compile identical unlabeled side-by-side image grids. Blind human panel (preferably 3 reviewers with differing tastes) judges: (a) style/occasion fit, (b) actual visual compatibility, (c) attractiveness, (d) individuality across three looks, (e) would the user wear/buy it. Track unanimous/majority and disagreements; VLM critic score is supplementary only, NEVER the sole judge.
5. **Hard contract:** any invented SKU, identity mismatch, missing verified product image/link, bypassed locked anchor, invalid FX or breach of a real cap is blocker, not trade-off. Record retrieval recall: are there acceptable item alternatives in each slot? If not, the bottleneck may be catalog depth/metadata and more AI cannot solve it.
6. Pilot winner requirement (pre-declared *heuristic*, not statistical proof): at least 8 of 12 majority votes prefer the new approach against V5, zero critical fashion mismatches, 100% SKU/source/budget/identity compliance, and 3 visibly distinct outfits where feasible. Also measure approximate inference cost and latency; do not call it deployment-ready even if passed.
7. If A or B wins, implement **only the winner** as a coherent end-to-end system and expand to a 30+ input blind held-out A/B, target >=70% preference. If neither wins, STOP full rebuild: report whether problem is poor catalog coverage, visual representation, concept planning, inaccurate complementarity, or uncalibrated judge. Iterate on the measurable cause, not on one-off garment text bans.
8. Run live paid test calls and any quota-debit acceptance only with a dedicated authorized paid user, explicit owner acknowledgement, secure credentials and bounded total spend. Never run in CI or assume prior tests authorized future spend.

## Metrics the experiment MUST print

Per run: SHA; catalog snapshot/source hash; model IDs; requested style/occasion/budget; shortlist recall by slot; hard rejection reasons; visual critic version; actual images for human review; final look identities and silhouette/footwear families; quote currencies; latency stage p50/p95 where meaningful; cost estimate with known provider rates; provider failures and abstentions. Redact private account and tokens.

## Final decision format

| Candidate | Real SKU hard compliance | 12-scenario blind wins | Critical fashion issues | Looks distinct? | Estimated per-request cost | Median/p95 latency | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V5 | NOT MEASURED | baseline | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | control |
| Reference-first | NOT IMPLEMENTED | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | unproven hypothesis |
| Creative-first | MOCK CONTRACT ONLY | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | unproven hypothesis |

**No evidence => no architecture approval.** This specifically prevents the previous failure mode: months of green unit tests without beautiful, distinguishable outfits for real users.
