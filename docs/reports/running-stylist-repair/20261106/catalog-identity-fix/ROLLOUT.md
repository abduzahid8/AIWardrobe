# Catalog identity correction — prepared, not deployed

The failed v180 log at 2026-11-06 06:17 UTC records six fallback TOP pages and zero fresh eligible products. Offline replay against the saved 7,020-row catalog reproduces that failure with the actual displayed-product history.

Catalog-only eligibility received raw database IDs without `shop_` or `isShopItem:true`, so `garmentContextReason` returned `source_requires_catalog` for every row. The handler now passes the same shop identity used by returned products into eligibility. Source restrictions, authentication, quota enforcement, budget checks and page limits remain in place. Only the handler differs among the 37 verified source inputs.

All 38 focused retrieval tests pass. The corrected offline TOP replay finds 19 fresh eligible products in four pages and returns 12 candidates. These are retrieval results, not successful outfit generation. Bundled startup and missing-authentication checks passed offline with remote imports stubbed.

## Reviewable deployment artifact

Archive: `aiwardrobe-stylist-v2-catalog-identity-fix-edge-source.tar.gz`

SHA-256: `a5b0120e60a17500d6138c632774a0e7e6447553829d51ce3f5599dda65e0f08`

Generation marker remains `stylist-runtime-repair-v2`. Full input hashes, original archive checksum and bundler version are in `manifest.json`; the business-code change is in `corrected-source.patch`.

## Controlled rollout after approval of this corrected artifact

The previous approval named a different exact archive checksum. This corrected release has not been deployed.

1. Recheck the archive hash and the live function baseline: project `fyqpifmrsftsfqibhwhy`, function `generate-outfits`, ACTIVE v181, JWT verification disabled, bundle hash `ff4b5f02772d999c3ec57ccb973599ec7fbe75044a9e0f7e4c30ced3ba3570df`. Stop for any intervening deployment.
2. Re-read actual CLI help. Extract this archive into another isolated directory and use its `backend` working directory. Deploy only the explicit `generate-outfits` name with `--remote=fyqpifmrsftsfqibhwhy --use-api --no-verify-jwt`. Do not use prune or deploy any other resource.
3. Confirm ACTIVE numeric version and JWT configuration. Use the existing authorized simulator test account, after re-reading its legitimate entitlement. Run Classic Work, Classic Formal, then Casual and Travel only if those pass. Repeat one scenario with actual displayed history. Capture actual HTTP status, generation marker, counts, latency and product overlap; inspect four-slot identity, coherence, links and iOS descriptions. A local or mocked result is not hosted verification.
4. On critical failure, stop and redeploy only the verified v179 rollback archive with JWT verification disabled. Its SHA-256 remains `25e33ee8415ad048a4b2513fd723d5d1812891639793a462c86ec5353d81bf21`. Verify ACTIVE and source equivalence. Do not alter subscriptions, billing, secrets, RLS, database configuration or other production resources.
