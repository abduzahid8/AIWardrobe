# Stylist 6 — World-Class Personal AI Stylist Upgrade: QA & Release Readiness Report

- **Date**: 2026-11-19
- **Branch**: `ai_fashion_stylist`
- **Baseline SHA**: `a99c268e` (`origin/codex/stylist6-visual-first-20261110` tip before upgrade)
- **Upgrade Milestone SHAs**:
  - `4554aad0` — Pillar 1: Personal Style DNA, CIELAB color theory, 4-item outfit grammar, opt-in appearance analyzer, DNA-powered personal taste
  - `5d6c68bc` — Pillar 2 & 3: Structural catalog hygiene, per-direction retrieval, orthogonal Creative Director, expanded SKU composer, editorial Stylist's Notes
  - `0ae0b3a0` — Pillar 4: Multi-provider AI router, proportion-aware `1024×1024` / `1800×1000` editorial boards, surgical 1-slot refiner, 6-axis visual critic & cross-lab jury
  - `93e0cce1` — Pillar 5 & 6: Unified taste learning, catalog quality flywheel, multi-tier cache + request coalescing, pgvector/taste SQL migrations, 1M-MAU scale benchmark
  - `9dac9e01` — Pillar 7: Client/UX integration (`AIOutfitmaker.tsx`, `stylist6Dev.ts`, Personal Style DNA, 1-tap slot swap, Stylist's Notes, structured feedback, try-on handoff)
  - `6a899a27` → `52923c77` — Full 5-pillar integration into `cloud_handler.mjs` & `fashion_deck.mjs`
  - `f06d3339` — Localization (`en`/`ru`/`uz`) for new `outfitMaker` keys & fine-knit merino top slot parity
  - `404ec768` — **5 verified live-eval root-cause fixes + 7 regression tests** (code SHA under post-commit live verification below)
- **Production Stylist 5 (`backend/supabase/functions/generate-outfits/index.ts`)**: **UNTOUCHED**
  - SHA-256: `f90ced33ca3b0034c61cd2dfa289b189728dc75398cd72329f0b7c1854efc5e1` (`git diff a99c268e..404ec768 -- backend/supabase/functions/generate-outfits/index.ts` = 0 lines)
- **Catalog Evaluation Snapshot**: Frozen live `shop_catalog` snapshot (`8,554` active rows; `8,329` adapted across `3,768` tops, `1,323` outerwear, `1,591` bottoms, `1,647` shoes).
- **Release Gate Status**: **ENGINEERING VERIFIED ON LOCAL + LIVE VLM EVALUATION · NOT YET CLEARED FOR PRODUCTION PROMOTION** (see §6 for mandatory remaining owner release gates).

---

## 1. Cross-Scenario Failure & Quality Gaps Addressed

Before this upgrade (`a99c268e`), a 7-pillar Fashion Director & Architecture audit identified the following systemic gaps:

1. **Personalization & Appearance Blindness**: No opt-in appearance/undertone/contrast analysis, no body-proportion directives, and no owned-wardrobe DNA synthesis feeding creative planning or scoring.
2. **Algorithmic / Generic Styling**: Creative Director lacked explicit body/color/texture orthogonality, thermal suitability scoring (`31°C` summer vs `5°C` winter), and CIELAB perceptual color harmony.
3. **Catalog & Slot Hygiene Leaks**: Missing structural PDP/media validation, swimwear/eveningwear leaks into daytime work/casual, and `2,473` catalog rows with `color: "unknown"` in merchant titles despite verified `visual_fashion.attributes.color` metadata.
4. **Visual Evaluation & Refinement Gaps**: Hardcoded `inFlight >= 2` concurrency bottleneck, no proportion-scaled studio flat-lay boards (`1024×1024` / `1800×1000`), no 1-slot surgical refiner (`refineLookSurgically` / `swapSingleSlot`), and multimodal portfolio narrowing dropping 3-look verified portfolios down to 2 looks even when verified triples existed.
5. **Closed-Loop Learning & Scale**: No unified behavioral taste profile, no per-SKU visual quality flywheel quarantine, no L1/L2 coalesced caching, and no 1M-MAU throughput/cost model.
6. **Live-Eval Root-Cause Fixes Landed in `404ec768`**:
   - **Fix 1 (P0 HTTP 500 Crash)**: `surgical_refiner.mjs` passed 1 arg to `priceToBudgetCurrency(item)` instead of 2 (`priceToBudgetCurrency(item, currency)`), crashing cross-currency budgeted `refineLookSurgically` & `swap_slot` in `budgetFx.ts:referenceAmount`; also defaulted `currency = budget?.currency` in `cloud_handler.mjs`.
   - **Fix 2 (Photographic Color & Subtype Propagation)**: `catalog_adapter.mjs` now populates `color` from source-hash-verified `visual_fashion.attributes.color` (`confidence >= 0.8 && margin >= 0.1`) when title color is `'unknown'`; `mapDeckItem` preserves `subtype` and resolved visual color.
   - **Fix 3 (Verified 3-Look Portfolio Restoration)**: `fashion_deck.mjs` restores the highest-scoring server- and photo-verified 3-look triple from `verifiedPortfolios` after multimodal photo jury narrowing when `eligibleTriples > 0` and revision cannot improve, keeping `unreviewedFillAdded === 0`.
   - **Fix 4 (Warm-Weather Discovery & Composer Quorum)**: `cloud_handler.mjs` adds warm-weather (`temp >= 24°C`) discovery subtypes and `fashion_deck.mjs` checks composer quorum against `resolvedPerDirectionPools.unifiedPool` (`minSlotCount = 3`).
   - **Fix 5 (Visual & Taste Polish)**: Vertical header stacking on `1024×1024` look boards (`editorial_board_renderer.mjs`), hard exclusion of `swapped_from_sku` & explicit slot-targeted negative-rated SKUs (`cloud_handler.mjs`), daytime exclusion of evening/tuxedo/dinner stud shirts (`catalog_hygiene.mjs`), and auto-derived `'brown'` avoidance for opt-in Cool Winter profiles unless explicitly favored (`personalStyleDna.ts`, reinforced in `per_direction_retrieval.mjs` and `creative_director_ai.mjs`).

---

## 2. Live Real-Photo Before/After Evaluation (Same Catalog Snapshot & Same Personas)

### 2.1 Phase 4A Paired Before (`a99c268e`) vs After (`52923c77` + Fixes 1–4)

Evaluated on the **same frozen 8,554-SKU snapshot** with real merchant photographs and real hosted NVIDIA NIM + OpenRouter VLMs across 5 diverse personas (`S6-LIVE-01..05`) plus interactive swap/feedback (`S6-LIVE-06`).

> [!IMPORTANT]
> **AI Proxy Jury Disclaimer**: The blind A/B verdicts below come from two independent cross-lab Vision-Language Models (`google/gemini-2.5-flash` via OpenRouter and `nvidia/nemotron-3.5-super-vl-120b-a12b` via NVIDIA NIM) inspecting randomized `ALPHA` / `BETA` portfolio contact sheets. Per `docs/STYLIST6_NORTH_STAR.md`, **AI-rater votes are an engineering proxy ONLY and do NOT constitute independent human blind panel acceptance.**

| Scenario | Before (`a99c268e`) Looks / CT / Thermal | After (`52923c77` + Fixes 1–4) Looks / CT / Thermal | Blind AI Proxy: Gemini 2.5 Flash (After vs Before) | Blind AI Proxy: Nemotron 120B VL (After vs Before) |
| --- | --- | --- | --- | --- |
| `S6-LIVE-01` Old Money + Work (Warm Autumn, 13°C, $1,800) | 2 looks · CT `0.600` · TH `0.794` | 3 looks (12 SKUs) · CT `0.792` · TH `0.803` | **AFTER** (`7.50` vs `5.40`) | **AFTER** (`6.80` vs `4.10`) |
| `S6-LIVE-02` Classic + Work (Cool Winter, 16°C, $1,200) | 2 looks · CT `0.748` · TH `0.794` | 2 looks (8 SKUs) · CT `0.889` · TH `0.816` | **BEFORE** (`5.60` vs `5.90`) *(flagged warm brown)* | **AFTER** (`6.30` vs `4.60`) |
| `S6-LIVE-03` Semi-Classic + Date Night (Deep Autumn, 15°C, $1,100) | 2 looks · CT `0.600` · TH `0.765` | 2 looks (8 SKUs) · CT `0.904` · TH `0.880` | **AFTER** (`7.80` vs `4.90`) | **AFTER** (`7.30` vs `4.40`) |
| `S6-LIVE-04` Smart Casual + Hot Summer (Warm Spring, 31°C, $750) | 1 look · CT `0.922` · TH `0.722` | 1 look (4 SKUs) · CT `0.874` · TH `0.775` | **AFTER** (`6.63` vs `3.25`) | **AFTER** (`6.00` vs `2.80`) |
| `S6-LIVE-05` Accessible $350 Budget + Work (Soft Summer, 18°C) | 1 look · CT `0.600` · TH `0.703` | 1 look (4 SKUs) · CT `0.879` · TH `0.807` | **AFTER** (`6.60` vs `4.80`) | **AFTER** (`6.50` vs `4.50`) |
| **Aggregate (AI Proxy Only)** | Mean CT `0.694` · Mean TH `0.756` | Mean CT `0.868` (`+0.174`) · Mean TH `0.816` (`+0.060`) | **4/5 AFTER** (mean `6.83` vs `4.85`) | **5/5 AFTER** (mean `6.58` vs `4.08`) |

- **Combined Cross-Lab Blind AI Proxy Win Rate**: **9/10 (`90.0%`)** for AFTER; mean overall score **6.70 vs 4.46 (`+2.24`)**.
- Notice the single Gemini loss (`S6-LIVE-02`) directly caught two real taste flaws: Cool Winter receiving a warm brown blazer/trouser/shoe look and a black-stud M&S dinner shirt appearing in daytime work — both directly remedied in Fix 5 (`404ec768`) and re-verified below.

---

### 2.2 Post-Commit Live Verification Directly Against Committed `404ec768`

Executed directly against the committed `ai_fashion_stylist` working tree (`404ec768`, zero scratch overrides) with live NVIDIA NIM + OpenRouter calls and real merchant product photographs, and visually inspected on all 4 rendered contact sheets (stored privately in scratch outside git):

| Scenario / Check | Outcome on Committed `404ec768` | Measured Metrics & Visual Verification |
| --- | --- | --- |
| **`S6-LIVE-01` Old Money + Work** (13°C, $1,800, Warm Autumn) | **PASS** · HTTP `200` · `41.5s` · `3/3` photo-verified 4-item looks (12 unique SKUs) | `anyEveningTuxedoShirt: false` (M&S `ms-clp60542508-z0` dinner shirt cleanly excluded by URL/title hygiene; L01 uses plain white Supima cotton shirt `ms-clp60762923-z0`). 3 distinct outerwear + footwear pairings: navy peacoat + brown suede Chelsea boots (`$1,519.63`), camel belted trench + black leather oxfords (`$972.35`), taupe wool overcoat + brown suede loafers (`$883.54`). Mean CT **`0.895`** (`+0.103` over Phase 4A), Mean TH **`0.823`**. |
| **`S6-LIVE-02` Classic + Work** (16°C, $1,200, **Cool Winter High-Contrast**) | **PASS** · HTTP `200` · `59.0s` · `3/3` photo-verified 4-item looks (12 unique SKUs) | **`anyWarmBrownPiece: false`** and **`anyEveningTuxedoShirt: false`** across all 12 SKUs. Pure Cool Winter palette visually verified: L00 midnight navy peacoat + navy shirt/trousers + black oxfords (`$922.76`), L01 navy blazer + crisp white dress shirt + tailored trousers + navy suede loafers (`$565.17`), L02 charcoal wool blazer + slate-grey oxford + steel-blue trousers + navy calf derbies (`$628.91`). Mean CT **`0.813`**, Mean TH **`0.804`**. |
| **`S6-LIVE-06` Part A — 1-Tap Surgical Shoe Swap** (`action: 'swap_slot'`) | **PASS** · HTTP `200` · **`25ms`** · `status: 'swapped'` | Swapped `elegant-beckett-simonon-8185826377914-bordeaux` → `elegant-beckett-simonon-8224545767610-brown` (`$219`), `locked3Preserved: true` (GANT cream blazer, H&M white shirt, J.Crew black trousers locked). Visually confirmed `1024×1024` board stacks title (`y=42`) and subtitle (`y=66`) cleanly with **zero overlap**. |
| **`S6-LIVE-06` Part B — Closed-Loop Post-Feedback Shift** (`swapped_from_sku` + `wrong_shoes`) | **PASS** · HTTP `200` · `33.6s` (Creative Director cache hit in **`7ms`**) · `3/3` photo-verified 4-item looks (12 unique SKUs) | **`swappedSkuEliminated: true`** (`elegant-beckett-simonon-8185826377914-bordeaux` excluded from all 3 looks). 3 distinct evening date looks (navy peacoat + forest-green polo + cognac oxfords; warm taupe blazer + golden-brown merino polo + espresso loafers; navy blazer + cream merino knit + cognac horsebit loafers). Mean CT **`0.800`**, Mean TH **`0.842`**. |

Private contact sheets inspected (not committed to git per merchant-image rights policy):
- `S6-LIVE-01_postfix_portfolio.jpg`
- `S6-LIVE-02_postfix_portfolio.jpg`
- `S6-LIVE-06_postfix_swap_after.jpg`
- `S6-LIVE-06_postfix_feedback_portfolio.jpg`

---

## 3. Automated Test, Typecheck, Build & Security Verification (`404ec768`)

| Verification Suite | Command | Result |
| --- | --- | --- |
| Stylist 6 Backend Unit & Integration Tests | `node --import tsx --test backend/stylist6/*.test.mjs` | **675 / 675 PASS** (`0` fail) |
| Stylist 6 Scripts & Eval Harness Tests | `node --import tsx --test scripts/stylist6/*.test.mjs` | **70 / 70 PASS** (`0` fail) |
| Cloud Compute Node Typecheck & Tests | `npm --prefix backend/stylist-node run typecheck && npm --prefix backend/stylist-node test` | **42 / 42 PASS** (`0` fail) |
| Cloud Compute Production/Staging Bundles | `npm --prefix backend/stylist-node run build` | **PASS** (`stylist-bundle.mjs`, `stylist6-bundle.mjs`, `stylist6-http-route.mjs`; S5 `sourceSha256` unchanged) |
| Root TypeScript Typecheck | `npx tsc --noEmit` | **PASS** (`0` errors) |
| Mobile App & Domain Jest Suite | `npx jest --ci --silent` | **326 / 326 suites · 3,561 / 3,561 tests PASS** (`0` fail) |
| Repository Secret Scan | `node scripts/security/secret-scan.mjs` | **PASS** (`3,427` text files, `0` findings) |
| Production Stylist 5 Integrity | `shasum -a 256 backend/supabase/functions/generate-outfits/index.ts` | **UNTOUCHED** (`f90ced33ca3b0034c61cd2dfa289b189728dc75398cd72329f0b7c1854efc5e1`) |

---

## 4. Scalability & Cost Benchmark (`scripts/stylist6/scale_load_benchmark.mjs`)

Measured on `404ec768` over the full adapted 8,329-SKU catalog snapshot (`174.39ms` catalog adaptation):

| Concurrency | Elapsed (ms) | Throughput (req/s) | P50 (ms) | P95 (ms) | P99 (ms) | Cache Hit Rate | L1 / L2 / Coalesced / Miss | RSS Delta (MB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 10 | 51.38 | 194.6 | 45.55 | 51.18 | 51.18 | 43.3% | 0 / 0 / 13 / 17 | +4.55 |
| 50 | 45.77 | 1,092.3 | 40.93 | 43.86 | 44.92 | 68.0% | 78 / 0 / 24 / 48 | +1.38 |
| 100 | 55.52 | 1,801.1 | 40.88 | 52.22 | 52.37 | 74.7% | 174 / 24 / 26 / 76 | +1.84 |

- **1M-MAU Sizing Model** (`2.5M` generations/month; `0.96` avg RPS / `5.76` diurnal peak / `14.4` spike):
  - Compute: `6` steady-state / `22` peak-autoscale Node instances (`2 vCPU / 2 GB`, `24` concurrent in-flight/node via adaptive semaphore, `12×` density over legacy `inFlight >= 2` wall).
  - pgvector HNSW memory: `39 MB` RAM (`8,329` SKUs) → `468.8 MB` RAM at `100k` SKUs.
  - AI unit cost: `$19.50/1k` uncached → `$13.60/1k` cached (`-30.3%` reduction).
  - Modeled total monthly cost at 1M MAU: **`$35,561/mo`** (`$0.0356/MAU/mo` = `$34,000` AI + `$782` Compute + `$180` Redis + `$599` Postgres).

---

## 5. Follow-Up Polish & Residual Catalog Observations

Honest audit of the post-fix `404ec768` live outputs surfaced three follow-ups (items 1 and 2 are now resolved in the final polish commit; item 3 remains an operator catalog backfill task):

1. **Title Color Dictionary (`milk` → `cream`) — RESOLVED**: `canali-knt00252954` (`"Milk-colored extra-fine merino wool crew neck"`) previously mapped to `color: "unknown"` when lacking a `visual_fashion` row. Added `\bmilk(?:[- ]?colou?red)?\b` → `'cream'` in `backend/stylist6/catalog_adapter.mjs` with a unit regression test. Separate merchant row `aurelien-10228846690634` (`"Navy Pleated Metropolitan Trousers"`) still carries an upstream URL/visual color conflict (`tan` vs navy photograph) that warrants catalog QA audit.
2. **Independent Portfolio Jury Cross-Lab Failover — RESOLVED**: In the post-commit live runs, primary multimodal portfolio editing (`photoEditorialVisualJury: 'applied'`, `photoEditorialVisualMedia: 'ready'`) succeeded on 4/4 runs while the secondary cross-lab juror (`Kimi K3.1 Pro`) timed out within the 9.6s window. Wired `openrouterApiKey` and `enableCrossLabFailover: true` from `cloud_handler.mjs` into `selectPhotoEditorialDeck`, split per-provider timeout headroom cleanly inside `photo_editor_final.mjs`, and added a unit test proving automatic failover from a Kimi timeout to OpenRouter `google/gemini-2.5-flash`.
3. **Catalog Visual Profile & Embedding Backfill (Remaining)**: 977 active footwear SKUs still await `catalog_visual_jobs` completion, and pgvector embeddings (`backend/supabase/migrations/20261119090000_stylist6_pgvector_and_scale_indexes.sql`) must be populated once migrations are applied to staging.

---

## 6. Mandatory Owner Release Gates Still Open (Required Before Production Promotion)

Per `AGENTS.md` and `docs/STYLIST6_NORTH_STAR.md`, **none** of the following release gates are satisfied by unit tests, scale benchmarks, or AI proxy jury scores:

1. **Independent Human Blind Panel vs Stylist 5 (Primary Acceptance Gate)**:
   - Required: `>=70%` Stylist 6 preference over Stylist 5 across `>=30` diverse held-out scenarios evaluated blindly by `>=3` independent human reviewers, with `0` hard-rule violations.
   - Current status: **OPEN** (blind AI proxy jury achieved `9/10` vs baseline `a99c268e`, which is proxy evidence only; `scripts/stylist6/build_human_blind_package.mjs` is ready for operator execution).
2. **Merchant Image Rights, Live Size-Specific Stock & Landed Delivery Verification**:
   - Required: Confirmed commercial hosting/inference rights for `catalog_photo_cache_v1` (currently `0` READY licensed rows) and live merchant size/stock APIs before displaying `live_in_stock` (currently remains honestly labeled `Check at store`).
   - Current status: **OPEN** (operator/legal gate).
3. **Staged Cloud Deployment & Authenticated Native iOS Verification**:
   - Required: Deploy exact SHA (`404ec768` + final polish commit) to isolated Render Stylist 6 staging compute and Supabase edge proxy `aiw-stylist6-node-staging`, apply SQL migrations (`backend/supabase/migrations/20261119090000_stylist6_pgvector_and_scale_indexes.sql`, `backend/supabase/migrations/20261119091000_stylist6_personal_dna_and_flywheel.sql`) to staging Postgres, and verify authenticated 3-look generation + 1-tap slot swap + structured feedback end-to-end on native iOS without touching production Stylist 5 (`generate-outfits`).
   - Current status: **OPEN** (committing to GitHub does NOT automatically update pinned Supabase or Render staging).
