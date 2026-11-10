# CODEX MANDATE — BUILD AND PROVE STYLIST 6, ONE ARCHITECTURAL LEAP

You are principal fashion-recommendation ML engineer + senior TypeScript/React Native + secure backend engineer.

**Repository:** https://github.com/abduzahid8/AIWardrobe
**Working branch:** codex/stylist6-visual-first-20261110
**Baseline:** branch codex/stylist-node-compute-20261109 at addd3ed00859b96e44a9c068f28268f030b12e46 (PR #14, transport only)
**New design:** docs/architecture/STYLIST6_VISUAL_FIRST_ADR.md
**Candidate prototype:** backend/stylist6/prototype.mjs (not yet used by app)
**Contract tests:** backend/stylist6/prototype.test.mjs
**A/B manifest:** benchmarks/stylist6/scenarios.json (35 scenarios)

GOAL: replace the limited template-first V5 method with a **visual-first, concept-generating, SKU-grounded, whole-look assessed stylist**. Build the entire candidate in an isolated end-to-end staging path and compare to V5 using real outputs. Do not do incremental word-ban rules or merely improve descriptions. The prototype's mocked visualCritic is NOT evidence of quality.

## Phase A — verify the system before coding
1. Fast-forward this branch; inspect actual Git commit, PR #14 status, Supabase function versions and Vercel deployment. Do not reset local unrelated changes.
2. Audit backend/supabase/functions/generate-outfits/index.ts, src/domain/outfitConcepts.ts, src/domain/validatedQuartets.ts, src/domain/outfitDeckSelection.ts, and backend/supabase/functions/_shared/nvidiaLookRanking.ts.
3. Verify live Supabase shop_catalog (7,912 at last check), catalog_garment_intelligence (7,020), catalog_visual_fashion (5,467), catalog_image_embeddings (5,437 x 768 floats), catalog_stylist_capsule, and image/source hash provenance; report exact live counts and stale/misclassified coverage before assuming they are unchanged.
4. Measure a real V5 baseline on the 35 cases, using a dedicated permitted test user and pinned catalog/FX snapshots; collect returned metadata, latency, visual samples, server logs, no secrets. Do not run paid tests without explicit approved test conditions.

## Phase B — implement Stylist 6 as a complete, separate engine
1. Use hosted model APIs; do NOT require a local GPU. Test which NVIDIA or other existing licensed provider endpoints actually support the needed text creative-director and image understanding requests. Explicitly validate model access and bounded cost/latency.
2. Creative director generates 6–8 genuinely different style briefs by occasion and user preference: silhouette, garments, color hierarchy, layer role, materials, formality, footwear family. No SKU IDs in brief outputs. Validate schema and novelty (do not accept six renamed copies).
3. Retrieve real shop/closet candidate SKUs per brief using visual fashion profiles plus source taxonomy; do not bottleneck on a single universal 24-per-slot shortlist. Reuse cached image features only with correct hashes. Do NOT pretend image embeddings are text embeddings or complementary-item scores; only use similarity when its space and recall are verified. An extension/migration is optional and needs review.
4. Generate many plausible real-SKU full-look candidates from those briefs, then call an actual hosted multimodal visual critic on securely composed product contact sheets and outfit descriptions. Judge the whole set: style, compatibility, proportion, silhouette, texture, color, dress code, occasion, originality, quality. Keep IDs grounded to real catalog items. The current backend only gives the model attributes and lets it rank 12 prebuilt looks; that must change.
5. Reuse the existing trusted hard constraints (JWT, paid quota, RLS policy, source, canonical identity, locked garments, ownership, rotation, product links, FX budget). Extract a safe admission/validation boundary as needed; never charge twice or silently replay after errors.
6. Select 3 truly distinct looks jointly (different silhouettes/construction, layering, footwear family where inventory permits) with measurable novelty. If feasible inventory is insufficient, return a truthful partial result rather than three cosmetic variants.
7. Derive concise, evidence-backed outfit descriptions; localize en/ru/uz. Do not hallucinate visual facts, material, price or availability.
8. Provide strict response schema, provenance, provider/model, request trace ID, per-stage timing, real server CPU/wall and safe cost accounting. No secrets/user data in logs.

## Phase C — staging + native device
1. Keep Supabase production generate-outfits v207 and existing aiw-stylist-node-staging v4 unchanged. Create a SEPARATE protected Node preview and staging Edge route for Stylist 6, with proven HMAC user-JWT auth. No new public unsigned paid endpoint.
2. Create a development-only mobile endpoint selector. The release build must always use production; unit tests enforce this. Stamp actual client SHA and backend generationVersion in dev diagnostics.
3. Build and install the Mac iOS simulator binary with latest source. Avoid stale Metro and OTA bundles. Exercise the 35 benchmark cases via same-input V5/S6 A/B and capture screenshots of representative results.

## Phase D — prove business-quality results, not just code
1. Run unit/integration/contract tests, security checks, Deno/Node/TypeScript, and simulator verification. CI mocks must NEVER make paid API calls.
2. Run approved real authenticated smoke: at least 5 sequential S6 requests and corresponding V5 controls; confirm exactly one quota debit per admitted request and no change for denied requests. Verify gallery IDs/images/source URLs/price/FX and strict budget. No retries after transport failure that might double-charge.
3. Blind human fashion A/B on at least 30 scenario outputs. Goal: >=70% prefer S6 to V5, no critical mismatch, 100% hard constraint compliance; **these are targets until measured**, not success claims. Report confidence, distribution, failures, and original screenshots.
4. Measure cold/warm p50/p95 latency, Node CPU and wall time, model inference cost per accepted outfit and 5xx rate. If quality improvement is achieved only by prohibitive cost/latency, STOP and redesign.
5. Produce full before/after report using exact Git SHA, source snapshot, model config, code links, accepted/rejected candidates, category diversity and provenance. Explicit GO/NO-GO.

## SAFETY
- DO NOT merge PR #14, modify Supabase production v207, replace staging v4, apply migrations, expose service-role/NVIDIA/HMAC tokens, lower RLS/auth or remove budget checks.
- DO NOT invent completed fashion experiments, production behavior, runtime measurements or provider capabilities.
- DO NOT add dozens of one-off hard-coded bans as the architecture. Concentrate effort on creative planning, broad grounded retrieval, visual outfit evaluation, global diversity and measured user preference.
- You may implement, locally test and push to this isolated branch. For paid production cutover, irreversible migration or release permissions, STOP and request explicit owner authorization.

## DELIVERABLE
Push a coherent feature-quality implementation to codex/stylist6-visual-first-20261110. Return final SHA, architecture diagram, exact commands/results, real staging logs (redacted), authorized quota deltas, p50/p95 CPU/wall, >=30 blind A/B evaluation outcomes with screenshots, and explicit release status. Do not claim the system is finished before these checks succeed.

**Begin with end-to-end architecture integration, not another single garment-rule exception.**
