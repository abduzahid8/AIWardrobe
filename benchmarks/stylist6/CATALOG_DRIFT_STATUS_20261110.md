# Catalog drift status — offline snapshot vs live production (2026-11-10)

## Offline snapshot characterization (measured, committed `data/`)

- Rows: 40,989. Active: 38,658. Shoppable (HTTPS media + price>0): 38,658.
- Currency mix of shoppable rows: EUR 15,408 / USD 9,593 / GBP 8,321 / UZS 5,336.
- **Distinct base products: 8,659; variant duplicates: 32,330 (79%).**
  The pilot's "38,388 admitted SKUs" shop roughly 8.6k real products ×
  per-shop variants. Any live comparison must use base-product IDs, not rows.
- Tool: `scripts/stylist6/catalog_drift.mjs` (`catalog-drift-v1`, 4 tests).
  Without `--live` it prints the characterization above and exits 0.

## Live side (NOT accessible from this environment)

- Audit-recorded live counts: `shop_catalog` 7,912 rows, 7,020 active —
  obtained by an independent authorized query, not re-verified here.
- No Supabase credentials exist in this environment (`SUPABASE_ACCESS_TOKEN`,
  `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` all empty), so no live export,
  no ID-level intersection, and no same-catalog pilot re-run were possible.
- Exact read-only export SQL is documented at the top of `catalog_drift.mjs`.
  Owner run: execute the COPY with server-side credentials to a private path,
  convert to JSON, then:
  `node scripts/stylist6/catalog_drift.mjs --live /private/live.json --report /private/drift.json`
- Only after the drift report exists may the 12-scenario pilot be re-run on
  the live universe (`pilot_12.mjs` accepts the same row shape). Do not claim
  the 99-look offline result transfers to live stock.

## Image availability under the authorized-access policy (measured)

`scripts/stylist6/image_access.mjs` (`image-access-v1`): allowlist-only fetch,
plain client headers, audit-logged, private cache. Against the 152 pilot
product URLs: **59 fetched + 9 cache-shared (variant-shared photos) = 68
accessible; 72 blocked (unapproved hosts, incl. known-403 brand CDNs);
12 failed on approved hosts (dead media — strongest staleness evidence).**
Approval of a host = an explicit allowlist edit. No bypass, no substitution.
