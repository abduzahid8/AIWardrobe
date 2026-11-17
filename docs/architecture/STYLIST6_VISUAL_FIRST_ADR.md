# ADR-006 — Stylist 6 Visual-First Outfit Generation (PROPOSED, NOT IMPLEMENTED)

**Date:** 2026-11-10
**Decision:** Replace template-first outfit generation and near-inert provider reranking in an **isolated Stylist 6 engine**, keeping the current Stylist 5 route untouched until genuine A/B evidence proves improvement. This is a system-level replacement; **do not add another sequence of garment-name veto rules** as the primary solution.
**Starting SHA:** `addd3ed00859b96e44a9c068f28268f030b12e46`
**Working branch:** `codex/stylist6-visual-first-20261110`
**Existing Draft PR:** #14 (protected Node transport for the old engine, not a quality approval).
**Production guard:** Supabase `generate-outfits` v207 MUST NOT be modified or redirected.

## 0. Verified cause, not a hunch

- `src/domain/outfitConcepts.ts` maps Old Money and Classic to only three predeclared concept families each. The plan **starts from formulas**, then builds quartet permutations from a capped per-slot pool (`validatedQuartets.ts`).
- The ordinary request evaluates up to 256 full combinations (`backend/supabase/functions/generate-outfits/index.ts`), then NVIDIA **ranks up to 12 pre-made looks**. The model explicitly cannot propose/change products (`_shared/nvidiaLookRanking.ts`).
- The NVIDIA contribution to final editorial utility is only `0.012 * rank`; the model receives item text + image-derived attributes, **not the garment photos**. AI is not acting as a stylistic author.
- At examination: `shop_catalog=7912` priced/photo/source rows; `catalog_garment_intelligence=7020`; `catalog_visual_fashion=5467`; `catalog_image_embeddings=5437` with **768-dimensional arrays**. There is no installed `pgvector` extension. Do not pretend those image-only vectors are text embeddings or suitable for complementary-item scoring without a calibrated method.
- User-provided real iOS screenshots (08:16–08:18 local, 2026-11-10) show more varied names and some 3-look diversification after earlier work, but still many formal blazer+derby results for different briefs, plus questionable all-over checks under blazers and weak visual/formality hierarchy. Screenshots do **not** establish which backend route or generation SHA ran; collect provenance.

## 1. New pipeline: concept-first, visually judged, catalog-grounded

```
Authenticated user request (style, occasion, owned/new, anchor, budget, weather, feedback)
    |
    v
[A] Validated request + EXISTING JWT, quota, budget FX snapshot and hard constraints
    |
    v
[B] Creative director LLM creates 6-8 DISTINCT briefs (NO product IDs):
    theme, aesthetic, occasion register, silhouette, fabric, palette,
    layer composition, shoe family, avoid/repetition intent.
    |
    v
[C] Grounded catalog retrieval per brief:
    hybrid source-taxonomy / visual metadata / title / brand / price / text,
    cached high-confidence image profiles and compatible visual embeddings,
    broad recall (multiple garment families, no 24-item global bottleneck).
    Optional precomputed approximate nearest-neighbor retrieval only after
    matching embedding-space proof and measured improvement.
    |
    v
[D] Outfit constructor: assemble many complete SKU-only outfit candidates
    seeded by briefs + complementary roles. Require real source URLs, image
    URLs, canonical product identities and exact price facts.
    |
    v
[E] Whole-outfit VISUAL critic, hosted (no local GPU):
    evaluate contact sheet of real 3-4 product photos + metadata;
    scores calibrated for style, cohesion, formality, silhouettes,
    contrast, materials, individuality and occasion. The critic may
    explain/reject, but never invent or silently replace product IDs.
    Provider image support, quota and per-request cost must be verified.
    |
    v
[F] Existing hard gates RE-VALIDATE:
    identity, source, ownership, locked anchor, currency FX budget,
    physical completeness, size/season if known, accessible product URL,
    non-contradiction. On failure, repair by choosing another REAL candidate
    with bounded attempts, not inventing a product or relaxing a safety gate.
    |
    v
[G] Global 3-look editor: jointly choose genuinely different LOOKS
    (distinct upper construction, outer/layer family, silhouette, footwear
    family when inventory permits; avoid repeated SKU and near-clone looks).
    May return fewer with an honest inventory-limited reason.
    |
    v
[H] Grounded explanation using chosen SKUs and observed visual evidence;
    no fabricated details. Stable en/ru/uz display via existing client.
    |
    v
[Output + safe telemetry + optional trace for offline blind evaluation]
```

**Key architectural inversion:** generation starts from *different outfit stories* and relevant visual examples; only then does retrieval and composition happen. No hardcoded three concepts are allowed to be the sole generator. The provider must materially influence which ensembles are proposed/selected, not just rearrange 12 predetermined outputs.

## 2. Interface boundaries

- Implement independent `backend/stylist6/` code: `briefs`, `retrieval`, `composer`, `visualCritic`, `selection`, `qualityReport`, `handler` and provider adapters. No duplicate auth/billing implementation: reuse the hardened original admission boundary through a carefully designed adapter. Do not double-charge.
- Explicit versioned contracts:
  `Stylist6Request = normalized request + secure source snapshot`;
  `StyleBrief = {id,concept,register,silhouette,palette,texture,slotIntent,layerPolicy,footwearFamily,noveltyAxes}`;
  `Candidate = {catalogIds, conceptId, productSnapshot, provenance}`;
  `VisualCritique = {candidateId,styleFit,visualCohesion,occasionFit,formalityConsistency,novelty,confidence,evidence,flags}`;
  `Stylist6Result = existing client response contract + generationVersion + engineIdentity + traceId`.
- JSON schema validated at every AI boundary. User text and catalog names are untrusted data; prompt injection cannot become instructions. No user token, secret, prices without verified quote, or private wardrobe photo metadata sent beyond necessary provider scope.
- Treat availability/missing image/price as **unknown**, not fabricated. Preserve one pinned currency conversion snapshot. Product photo hashes + facet source hashes must match; stale profiles are ignored.
- Build creative and visual critics as **replaceable hosted adapters**. NVIDIA may be reused but confirm model access, image support, multi-image constraints and cost before enabling. If vision is unsupported, report degraded mode; don't label text-only rules as visual judgement.
- Existing `catalog_image_embeddings` (768 floats, image SHA keyed) are useful only if matched to actual catalog images. Cosine similarity alone measures visual resemblance, NOT outfit complementarity; never use it as a standalone fashion-quality score. No local GPU or model training required for MVP.
- Bound and expose each stage's cost, time and candidate counts; exact CPU/wall timing end-to-end on Node. Explicit no-automatic-retry policy for paid admission.

## 3. Isolation and rollout (do NOT touch existing release)

- Create a protected **separate preview Node route** for Stylist 6, plus `aiw-stylist6-staging` via a secure Supabase proxy. Do NOT repurpose existing `aiw-stylist-node-staging` or `generate-outfits`. Use HMAC and user JWT, server-side credentials only, POST only, no redirect, no unsigned diagnostic generator.
- Mobile only opts into this route in dev/test builds behind an explicit runtime selector and logs the **actual** backend generationVersion; ordinary mobile and production remain unchanged.
- Preserve PR #14 and Stylist 5 production v207 as A/B baseline; capture a verifiable rollback bundle before any future release.

## 4. Evaluation is part of the architecture (not optional)

Create replayable **same-input, same-catalog-snapshot** comparisons with Stylist 5, using `benchmarks/stylist6/scenarios.json`, locked UTC/currency/weather inputs and deterministic request IDs. The selected screenshot scenarios are mandatory (Old Money+Work, Classic+Work, Semi-Classic+Date night).

Compare real generated outputs rather than test doubles:
1. **Hard contract pass rate:** 100% no invented SKUs, anchors, protected auth, quota exact +1, hard budget obeyed, source URLs & actual images. Any failure = NO-GO.
2. **Fashion quality:** blind human A/B panels evaluate coherence, occasion appropriateness, visual taste and distinctness. Required goal: clear majority prefer S6, **target >=70% head-to-head preferences and no repeated critical outfit defect**, measured on at least 30 varied scenarios. This is a *target*, not a claimed result.
3. **Deck diversity:** 3 distinctive compositions when feasible; measure real silhouette, upper family, shoe family, palette and product overlap, not just different concept labels. Record inventory feasibility.
4. **Practicality:** request success, cold/warm p50/p95 wall and Node CPU, provider tokens/cost, bad-link rate, low-budget feasibility. Compare with V5 under identical limits, avoid worsening user-visible latency beyond an agreed bound.
5. **Robustness:** exercise 30 benchmark scenarios, photo-missing, unknown image, locked anchor, repeated generations, mixed currency, wardrobe-only and catalog-only; no silent downgrade to old engine.
6. **Actual iOS:** test one target build and verify returned client/server SHA, source, quota, actual buy links, image quality and localized explanation; capture screenshots.

Do not award a release GO based solely on green CI, a single pretty screenshot, static fixtures, or unauthorized JWT-negative transport tests.

## 5. Engineering deliverables — one coherent architecture PR

- New engine code and adapters; request/response schema, visual critic contact-sheet composition with image SSRF safeguards.
- Catalog-side retrieval strategy and evidence joins (no DB mutation until migrations reviewed).
- Offline deterministic test harness with mocked providers **and** separate user-authorized real metered staging tests; benchmark reports with real outputs.
- Robust A/B switch for dev only and proof from simulator.
- Migration/rollback plan, performance and cost table, security review, baseline vs S6 comparative QA.
- Commit complete cohesive PR against this branch and mark NO-GO until positive authenticated evidence. Avoid 'style check' micro-patches to V5 as the primary plan.

## 6. Suggested internal sequence (one release-sized change)

1. Capture baseline V5 accepted look outputs/metrics and confirm staged auth/quotas and test account.
2. Build S6 concept-first orchestrator plus real-catalog retrieval and hosted visual evaluator; add hard-gate adapter.
3. Complete validation harness and test 30 scenarios from fixed catalog snapshot, improve architectural bottlenecks only.
4. Run bounded live staging and blind human assessment, install instrumented iOS preview, compare.
5. Produce final evidence and GO/NO-GO. Production cutover is separately authorized.

**We are redesigning the engine, not promising instant quality without experimental evidence.**
