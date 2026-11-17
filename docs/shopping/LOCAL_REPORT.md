# Local implementation report — 2026-11-02

**LOCAL IMPLEMENTATION COMPLETE**

**REMOTE ACTIVATION PENDING APPROVAL**

No hosted migration, deployment, sync, upload, GitHub push or PR was performed
by this task. Branch remains `stylist-four-slot-recovery`, based on `5784fc5`.
Risk: MEDIUM for client retrieval/presentation and the optional new read RPC.

## Implementation and requirement evidence

| Requirement | Implemented behavior | Verification |
|---|---|---|
| Shared identity | Explicit listing/physical/family/variant/visual identities; Unicode brand/name/color/base helpers | `catalogIdentity.test.ts` |
| Virtual families | Evidence-based grouping; low confidence stays separate; every row/price retained; selected IDs pinned | `catalogIdentity.test.ts`, `catalogFamilyView.ts` |
| Shopping presentation | Inspo and Home family representatives with accessible exact-row variant selection | `catalogVariantPicker.test.tsx`, `homeShoppingSelection.test.ts`, `shopCatalogSearch.test.ts` |
| Broad bounded discovery | Counted seeded circular windows today; server-ranked full eligible relation after RPC activation | `catalogDiscovery.test.ts`, local PostgreSQL assertions |
| Consumer integration | Home, Inspo, AIOutfitmaker, stylist evidence, generic search and completion use discovery | Screen/hook source, repository/pool/hook tests |
| Shopping diversity | Brand/model/role/color penalties within a quality band; anchors and native floors retained | `shoppingEvaluation.test.ts`, `shoppingIntelligence.test.ts`, `shoppingCatalogPool.test.ts` |
| Find Better | Overall/cheaper/closest-style/wardrobe/fit ranking and evidence through shopping service | `shoppingEvaluation.test.ts`, `shoppingAdvisor.test.ts` |
| Useful outfit gains | Bounded complete validated looks; equivalent owned substitutions and near-look deduplication | `shoppingEvaluation.test.ts`, `shoppingAdvisor.test.ts` |
| Wardrobe value / fit | Relative value with evidence/caps; existing measured fit engine; missing evidence stays unknown | Advisor tests and fit engine golden tests |
| Existing outfit rules | Physical identity, source isolation, rotation, anchors, style/activity/climate and budget remain authoritative | Existing diversity, variety, canonical-request, adversarial, rotation, slot and backend budget suites |
| Local server implementation | Additive invoker RPC, client compatibility, rollback and isolated real SQL tests | Migration, rollback and `catalog-discovery-local.mjs` |
| No remote mutation | No apply/deploy/push/sync commands; local SQL runner has no hosted connection | Tool execution record and runner's explicit local-only connection |

## Files in this shopping change set

- Identity/types: `src/types/shop.ts`, `src/utils/catalogIdentity.ts`.
- Family/presentation: `src/services/catalogFamilyView.ts`,
  `src/services/homeShoppingSelection.ts`, `components/CatalogVariantPicker.tsx`,
  `screens/HomeScreen.tsx`, `screens/InspoScreen.tsx`, and variant-number strings in
  `i18n/locales/{en,ru,uz}.json`.
- Retrieval/integration: `src/services/catalogDiscovery.ts`,
  `src/services/catalogRepository.ts`, `hooks/useShopCatalog.ts`,
  `src/services/stylistCatalogInventory.ts`, `src/services/shoppingService.ts`,
  `screens/AIOutfitmaker.tsx`.
- Intelligence: `src/services/shoppingIntelligence.ts`,
  `src/services/shoppingEvaluation.ts`, `src/services/shoppingAdvisor.ts`.
- Server: `backend/supabase/migrations/20261102163044_shop_catalog_discovery.sql`,
  `backend/supabase/rollback/shop_catalog_discovery.sql`.
- Verification: `scripts/validation/catalog-discovery-local.mjs`,
  `__tests__/fixtures/catalogDiscovery.sql`,
  `__tests__/fixtures/catalogDiscoveryAssertions.sql`,
  `__tests__/services/{catalogDiscovery,catalogRepository,homeShoppingSelection,
  shoppingAdvisor,shoppingCatalogPool,shoppingEvaluation,shoppingIntelligence,
  stylistCatalogInventory}.test.ts`, `__tests__/utils/catalogIdentity.test.ts`,
  `__tests__/hooks/shopCatalogSearch.test.ts`,
  `__tests__/components/catalogVariantPicker.test.tsx`.
- Documentation: this report, `FOUNDATION.md`, `SERVER_EVOLUTION.md`.

The obsolete ID/price endpoint-rotation helper from the first implementation was
removed. No app dependency or lockfile changes were needed. Supabase/Deno CLIs
were run from a temporary package cache; their generated CLI-version marker was
restored to its original content.

Catalog JSON, the international manifest and `scripts/expand-elegant-menswear.js`
changed independently during this task. Those edits were not overwritten. The
original Barbour deletion was likewise not restored by this task; that file later
changed independently. No claims about this task's diff include those changes.

## Behavior deliberately preserved

No edits to `generate-outfits`, shared physical identity/rotation/occasion/budget
rules, existing RPC definitions, applied migrations, RLS policies, auth,
subscriptions, rate limits, wardrobe/scan contracts, fit engine or offline sync.
OutfitShopSheet still represents exact selected products. No family IDs replace
existing catalog IDs or shop-prefixed generation IDs. Fallbacks, recent searches,
wishlist APIs, five-minute caching and reader cancellation remain supported.

## Performance and grouping findings

Browsing still loads at most 100 rows per page; stylist evidence at most four
100-row slots; completion at most 30 rows per slot. AIOutfitmaker requests four
30-row slices and retains an 80-row curated supplement only for refined requests.
No whole-catalog mobile payload, model-context expansion or image preloading.

Family construction is indexed with O(N log N) stable ordering. Shopping
selection is bounded to 100 outputs. Useful-outfit work is capped at 128 validator
calls by default per candidate; Find Better evaluates wardrobe gains for at most
24 retrieved candidates. Sampling limits and low confidence are returned.

Local fallback discovery adds a cached head-count request and at most two
bounded reads to wrap a window. This can add round trips relative to the old first
page, while retaining immediate bundled content. The activated RPC replaces
those with one bounded response and moves full-relation ranking to Postgres.
Local synthetic SQL execution measured approximately 114–132 ms for 100 rows over
7,800 fixture products. This is not a production/device latency claim.

The bundled catalog changed concurrently. One late local sample had 288 rows,
49 shoes and 288 conservative families, grouped in about 5.4 ms. Those are dated
local observations, never constants or production counts. Generic merchandising
phrases (including tailoring and non-iron descriptors) are not sufficient model
evidence. Production precision/recall is not established by synthetic tests.
A separately reviewed eligible Jack & Jones feed snapshot had 12 rows grouped
into 9 cards, with all 12 variants retained and a largest family of four. The
merchant adapter is scoped to official article numbers rather than title text.
Its reference extraction was checked against the [official product page](https://www.jackjones.com/en-gb/product/12306276_2048/regular-fit-chinos),
which labels article number 12306276. Unsupported merchants still require trusted
references or canonical product URL evidence.
Thin footwear inventory cannot be solved by fabricated variety.

## Validation

- Focused regression suite: **28 suites, 382 tests PASS** (30.689 seconds).
  Includes all new identity/family/discovery/UI/intelligence tests, the measured
  fit engine golden suite, and existing variety, shoe/physical rotation,
  canonical request, adversarial, slot, occasion and budget/FX suites.
- `npm run typecheck`: PASS.
- `npm run lint`: PASS, 0 errors; 1,363 repository warnings remain.
- All Edge Functions: PASS with the repository's frozen-lock check via temporary
  Deno 2.9.6 (`npx --yes deno check --no-config --node-modules-dir=none
  --lock=backend/supabase/functions/deno.lock --frozen
  backend/supabase/functions/*/index.ts`). No Edge source changed.
- Real local PostgreSQL integration and rollback: PASS; last measured execution
  125.821 ms over the synthetic fixture. Temporary clusters were stopped.
- `git diff --check`: PASS.
- Final merchant protocol guard: focused identity suite rerun; non-HTTP merchant
  URLs cannot supply trusted model evidence.

The entire repository-wide Jest suite and native visual QA were not run. The
reported coverage is the explicit shopping/outfit/fit regression scope above.
No hosted validation or ingestion scripts were run.

## Remote activation / rollback

Prepared remote object: only
`public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric)`
on Supabase project `fyqpifmrsftsfqibhwhy` (`wardrobe`). Exact source, grants,
benefits, risks, activation and tested rollback are in `SERVER_EVOLUTION.md`.
No table/index/RLS/Edge/secret/app_config changes or backfill are included.

The local implementation is usable without deployment: counted full-range windows
and offline fallback remain active when the RPC is missing. Server-balanced
brand/model/slot sampling requires separate approval to apply the migration.
The pre-activation global-window fallback can still reflect category/brand
imbalance; it improves full-range discovery without claiming the RPC's balancing
guarantees. Slot-specific stylist/completion retrieval remains balanced by slot.
No remote action is required to finish this local task, and none is authorized
implicitly by this report.

Before shipping the client, perform device/simulator visual QA and measure
hosted p95/query concurrency after any separately approved migration. Independent
Supabase advisor/security notices, including `shop_catalog_admin_write`, remain
a separate remediation project.
