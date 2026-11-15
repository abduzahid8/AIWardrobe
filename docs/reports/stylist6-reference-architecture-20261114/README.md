# Stylist 6: editorial architecture, production-ready routing, and commercial truth

Scope: isolated PR #15 branch (`codex/stylist6-visual-first-20261110`), not live production traffic. Release verdict: **READY FOR STAGING VALIDATION ONLY** (production promotion **NO-GO** until remaining operational gates and owner sign-off complete).

Machine-readable deployment and audit snapshot: [`deployment-matrix-20261115.json`](./deployment-matrix-20261115.json).

## Engineering changes

- **Reference guidance**: 46 user-curated textual style blueprints (silhouette, palette, named garment roles) guide the visual critic when the exact reference photograph is not authorized or configured for inference. Verdicts never claim photographic reference fidelity without image evidence. Changes in blueprint invalidate cached verdicts; the mode is propagated per displayed look (`visualReferenceMode: 'blueprint' | 'image' | 'none'`).
- **Pre-visual editor & descriptor precision**:
  - Narrow cross-slot vetoes (`backend/stylist6/editorial_register.mjs`) stop elevated Classic/Old Money work outfits pairing casual T-shirts with dress footwear, flannel with formal office tailoring, or an athletic sneaker with a complete suit.
  - `src/domain/garmentDescriptor.ts` disambiguates `shacket`/`overshirt` before `shirt`, prevents `"denim shirt"` / `"chambray shirt"` from matching `jeans`, excludes `"shirt dress"` / `"tennis dress"` from `dress-shirt`, and keeps `evening` (`tuxedo|dinner jacket|dinner suit|opera pump|patent`) as a soft context score reduction rather than a hard exclusion so formal patent Oxford shoes remain eligible.
  - `src/domain/occasionRules.ts` enforces bidirectional style-occasion boundaries (`rejectForStyleOccasion`) and warm/mild-weather puffer/shearling/heavy-down climate rejections (`rejectForClimate`).
  - `src/domain/outfitCoherence.ts` bounds the warm/cool neutral temperature clash when grounded by a neutral anchor (`0.68`) vs ungrounded (`0.54`).
- **Visual deck guard & media decoding**:
  - A third displayed look must represent genuine non-shoe structural/material change or grounded broad palette contrast rather than a new SKU label (`backend/stylist6/visual_critic_hosted.mjs`).
  - `backend/stylist6/cloud_visual_media.mjs` transcodes WebP/AVIF/GIF product images to PNG (`sharp`) before sending inline `data:image/png;base64,...` payloads to NVIDIA NIM so WebP/AVIF CDNs never cause false abstentions.
- **Merchant & offer truth**:
  - Exact listing/SKU binding (including FR Group supplier-specific SKU), positive size quantity, confirmed Uzbekistan delivery, and a 30-minute freshness window are required for a `live_in_stock` label (`src/domain/offerTruth.ts`).
  - Mobile UI (`screens/AIOutfitmaker.tsx` and `locales/{en,ru,uz}/outfitMaker.json`) uses honest `Check at store` / `Проверить в магазине` / `Do'konda tekshirish` CTAs on Stylist 6 cards and blocks synthesized search fallback URLs (`offerVerification === 'search_fallback'`).
- **Production-capable server-side routing & secret rotation (Stage G)**:
  - `backend/stylist-node/transport/signature.mjs` supports zero-downtime HMAC secret rotation (`previousSecret` / `secrets`).
  - `backend/stylist-proxy/index.ts` supports server-controlled rollout (`STYLIST_ENGINE_ROLLOUT`: `'stylist5'` default, `'cohort'` via server-verified Supabase `/auth/v1/user` against `STYLIST6_ROLLOUT_USER_IDS` / `STYLIST6_ROLLOUT_PERCENT`, or `'stylist6'`), routing Stylist 6 requests to `STYLIST6_NODE_URL` (`/api/stylist6-generate-outfits`) without trusting caller flags.
  - `backend/stylist6/cloud_handler.mjs` accepts both `{ request: { ... } }` staging payloads and top-level `generate-outfits` mobile release payloads (`stylePreferences`, `prompt`, `anchorItemId`, `selectedItemIds`, `wardrobeOnly`, `catalogOnly`, `preferences.maxShopBudget`, `recentLookSignatures`, `language`), resolves `shop_` ID prefixes against catalog/wardrobe pools, ignores caller-supplied `wardrobeItems`/`catalogRows`, and records pre-debit vs post-debit crash-window status (`aborted_pre_debit` vs `terminal_recorded` / `in_progress_or_interrupted`).
- **Mobile entry point parity (`AIOutfitmaker.tsx` & `useDailyAIOutfit`)**:
  - `screens/AIOutfitmaker.tsx` detects Stylist 6 responses by `isStylist6CloudVersion(data?.generationVersion)` even when routed through `generate-outfits` in Release builds—preserving server-approved looks (`acceptStylist6CloudDeck`), reconciling CBU FX (`applyStylist6CloudBudgetFx`), and skipping client `consumeOrPaywall('aiOutfits')` double-charging.
  - `src/services/outfitGenerationService.ts` and `src/services/dailyOutfitCache.ts` recognize `/^stylist-[56]-/` as `serverPlan`, preserve Stylist 6 item/outfit metadata (`isWardrobe`, `priceAgeDays`, `priceFreshness`, `offerVerification`, `direction`, `fashionDiagnostics`, `compositionProtocol`), and allow 3-piece `optional-layer-v1` Stylist 6 decks in daily outfit cache validation while still enforcing live weather checks.

## Objective offline evidence

- **72 scenarios, frozen UZS catalog (`384` rows)**:
  - **Stylist 6 Fashion (`stylist6Fashion`)**: **161/161 (100%) coherent looks; 52 complete 3-look decks (55 scenarios with >=1 look); 0.8205 average coherence; 0.6594 average styling distance; 0 budget violations; 0 constraint violations**.
  - **Stylist 5 baseline (`stylist5`)**: 155/161 (96.3%) coherent looks; 52 complete decks; 0.7827 average coherence.
  - **Stylist 6 Previous (`stylist6Previous`)**: 113/120 (94.2%) coherent looks; 38 complete decks; 0.7061 average coherence.
- **72 scenarios, UZS catalog expanded with 153 verified UZ footwear items (`537` rows)**:
  - **Stylist 6 Fashion (`stylist6Fashion`)**: **167/167 (100%) coherent looks; 54 complete 3-look decks (57 scenarios with >=1 look); 0.8187 average coherence; 73 selections of new UZ footwear across 17 distinct new SKUs**.
- **Full committed catalog 4-tier audit (`40,989` rows across 180 files)**:
  - Tier 1 raw active: **38,658** (inactive: **2,331**).
  - Tier 2 admitted usable (`adaptCatalog`): **38,047** (`outerwear`: 6,149, `top`: 10,213, `bottom`: 6,654, `shoes`: 15,031; UZS admitted: 5,053 including 831 UZS shoes).
  - Excluded invalid/ineligible active rows: **611** (`non_outfit_category`: 270, `catalog_category_conflict`: 199 across 172 SKUs, `price_implausible`: 142 including GANT `x100` UZS inflation).

## Release blockers still open for public production cutover

1. **Reference-photo rights & bucket**: Specific reference-photo rights are not independently verified in the 46-photo index, and the staging compute Preview has no configured reference-photo bucket (`STYLIST6_REFERENCE_IMAGE_BASE_URL` unset). Blueprint mode is active and honest.
2. **Live merchant stock verification**: Static catalog JSON snapshots do not carry live per-size inventory timestamps within 30 minutes (`live_in_stock = 0`). Cards honestly display `Check at store`.
3. **Human blind preference panel**: Independent human majority blind win rate >=70% across >=30 scenarios is still pending (`independentHumanVotes: 0`).
4. **30-sample hosted visual latency**: Prior hosted visual smoke measurements (4 samples; p95 26,174ms) do not yet meet >=30 samples with p95 <=25,000ms. No new paid NVIDIA inference was run in this session.
5. **Owner approval**: Production `generate-outfits` (v209), database migrations, auth, RLS, and quotas remain untouched awaiting owner sign-off.

## Safe local checks

- `node --import tsx --test backend/stylist6/*.test.mjs scripts/stylist6/*.test.mjs`
- `npx tsc --noEmit`
- `PATH="$HOME/.deno/bin:$PATH" npm run typecheck:edge`
- `npm --prefix backend/stylist-node run build`
- `npm --prefix backend/stylist-node test`
- `npm run lint`
- `node scripts/launch/static-checks.mjs`
- `node scripts/security/secret-scan.mjs`
- `npx jest --runInBand`
- `AIWARDROBE_DETERMINISTIC_PLANNER=1 node --import tsx scripts/stylist6/fashion_replay.mjs .private/catalog-uz-20261113.json .private/fashion-replay-new-UNIQUE.json`
