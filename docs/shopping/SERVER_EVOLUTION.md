# Prepared discovery RPC and future schema evolution

The local RPC migration and client integration are implemented and tested.
Remote activation is pending separate approval. The client works without it
through counted bounded discovery and the existing offline fallback.

## Exact prepared change set

Migration: `backend/supabase/migrations/20261102163044_shop_catalog_discovery.sql`.
Creates only:

`public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric)`

Arguments: slot, source, literal search, seed, limit, offset, optional currency
and maximum native price. Returns original row fields as JSON, at most 100 rows.
It is STABLE, SECURITY INVOKER, with empty search_path and explicit EXECUTE grants
to anon/authenticated/service_role after revoking PUBLIC. Existing table SELECT
privileges and RLS remain authoritative. The function performs no writes.

No tables, rows, indexes, existing RPCs, RLS policies, Edge Functions, auth settings,
secrets, workflows or ingestion configuration are changed by this migration.
No backfill or dual-write is required. No generated family ID is persisted.

## Activation and rollback

Only after approval: apply this exact migration to hosted project
`fyqpifmrsftsfqibhwhy` (`wardrobe`), then verify RPC visibility, anon/authenticated
RLS behavior, page sizes and generation/budget regressions. The task has not run
those remote actions or opened a PR. Review the new SELECT function's role grants
with the deployment administrator before application.

Clients detect a missing RPC and retain the counted-window path. No remote
feature flag or app_config change is required. For rollback, apply only the
prepared `backend/supabase/rollback/shop_catalog_discovery.sql`; it drops the
new function. Clients fall back automatically. There is no data rollback.

Local verification command:

```
AIWARDROBE_PG_BIN=/path/to/postgres/bin node scripts/validation/catalog-discovery-local.mjs
```

The runner creates and destroys an isolated Unix-socket-only PostgreSQL cluster.
It never reads .env or a hosted database URL. Tests cover 7,800 synthetic active
rows plus negative fixtures, invoker RLS for anon/authenticated, explicit grants,
full-relation coverage, brands/shoe archetypes, page overlap, price floors, search,
currency/budget filters, invalid arguments and rollback. A measured 100-row RPC
query took approximately 114–132 ms on this development machine; hosted p95,
concurrency and query plans still need post-activation measurement. No speculative
indexes were added.

Risks: ranking CPU/sort work at larger catalog scale, changing inventory between
pages, reduced visible rows when provenance is incomplete, and a temporarily
stale PostgREST function schema cache. The RPC does not know user style/climate/
activity/anchors: callers and the existing Edge pipeline must retain their hard
gates. It is not an authoritative outfit-generation endpoint.

## Future model — design only

## Additive normalized model

- `shop_product_families`: brand, canonical model title, category/garment type,
  optional material/silhouette evidence, provenance, grouping confidence/version.
- `shop_variants`: family relation, color/size evidence, image and fit metadata.
- `shop_offers`: variant relation, merchant, original source row ID, external ID,
  URL, native price/currency, optional observed stock/shipping.
- `shop_price_history`: offer ID, observed native price/currency and timestamp.

First retain `shop_catalog` and every existing UUID as the public compatibility
listing ID. Add nullable relations or a compatibility view only after parity.
Never convert family IDs into existing generation request IDs. Preserve
shop-prefix conventions at client/Edge adapters.

## Backfill and write/read path

1. Preserve raw import rows before similarity decisions. Dry-run fixtures must
   show exactly which relation is inferred, confidence and evidence.
2. Backfill only trusted merchant model/group references or product URLs.
   Queue ambiguous titles for review; no title-based deletion or deactivation.
3. Keep unknown stock/shipping/history null. Do not infer a discount or seed
   historical observations from current price.
4. Add foreign keys and indexes on variant.family_id, offer.variant_id and
   `(source, external_id)`, and `(offer_id, observed_at desc)` for history.
5. Give public SELECT only to active catalog relations via invoker-safe reads;
   restrict writes to the existing admin/service authorization. Test anon,
   authenticated and admin cases locally. Do not broaden unrelated RLS.
6. Introduce idempotent transactional ingestion dual-writes only after offline
   parity tests; retain legacy reads and a feature flag during validation.
7. Switch discovery reads first, then Edge candidate metadata; keep exact IDs,
   all hard gates and candidate bounds. Shadow comparisons must be authorized
   before any remote deployment or write.

## Evidence required for cutover

Record row/variant conservation, confirmed/false family merges, price-floor
coverage, per-slot/brand/visual-family coverage, request bytes, server p50/p95,
mobile grouping time and memory, and validated deck/rotation/budget outcomes.
The reported ~115 footwear rows in the supplied audit are dated observations;
new measurements must identify their time and source. Ranking cannot manufacture
missing footwear brands or sizes.

## Rollback

For the future normalized model, revert consumers to flat `shop_catalog`.
Keep dual-written old IDs valid. Stop normalized ingestion before reverting its
code. Retain additive tables and observations until a separately authorized
cleanup; no destructive rollback is necessary. Do not remove old reads/writes
until measured parity and a completed rollout window.

All future SQL, deployment, backfill, sync, storage and GitHub actions require
the exact remote-action approval requested in section 1.3 of the supplied task.
Only the additive discovery RPC above is prepared for a separately approved application. Security advisor
notices (including `shop_catalog_admin_write`) remain a separate review.
