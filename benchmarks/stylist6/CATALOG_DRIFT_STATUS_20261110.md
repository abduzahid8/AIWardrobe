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

## Statistical intersection: snapshot UZS slice vs live UZS intelligence (2026-11-10)

ID-level intersection is BLOCKED (no live export in this environment).
The table below compares pilot-adapter taxonomy (snapshot) against the
audit's `catalog_garment_intelligence` taxonomy (live). Minima that coincide
to the unit prove shared anchor products; count gaps prove variant inflation
(snapshot) and taxonomy divergence (knit pullovers).

| Role/subtype | Live active SKUs (min UZS) | Snapshot UZS rows (min) | Reading |
|---|---|---|---|
| top / t-shirt | 317 (219,900) | 397 (219,900) | same anchor product; +80 variant rows |
| top / polo | 251 (238,141) | 675 (238,141) | same anchor; +424 variant rows |
| top / shirt | 364 (349,900) | 578 (349,900) | same anchor family |
| top / fine-knit | **8 (1,186,196)** | **418 (279,900)** | live keeps 268 knits in outerwear/pullover; adapter projects them to top |
| outerwear / pullover | 268 (396,902) | — (projected to top/fine-knit or jacket) | taxonomy divergence, not missing stock |
| outerwear / jacket | 232 (258,000) | 364 (258,000) | same anchor; +132 variant rows |
| outerwear / blazer | 119 (849,900) | 92 (849,900) | snapshot UNDER-counts live blazers |
| outerwear / cardigan | 66 (635,043) | 199 (635,043) | same anchor; +133 variant rows |
| outerwear / coat | — (not tabulated) | 42 (1,099,900) | snapshot-only coats need live check |
| bottom / trouser | 287 (396,902) | 613 (396,902) | same anchor; +326 variant rows |
| bottom / tailored-trouser | 113 (429,900) | 78 (649,900) | snapshot misses cheapest live tailored |
| shoes / loafer | **23 (1,299,900)** | **211 (649,900)** | snapshot loafer supply is 9× live; live cheapest is 2× snapshot cheapest |
| shoes / sneaker | 21 (549,900) | 150 (803,630) | snapshot misses cheapest live sneaker |
| shoes / minimal-sneaker | 11 (649,900) | 8 (1,829,000) | snapshot misses cheapest live minimal |
| shoes / boot | — (not tabulated) | 35 (1,199,900) | snapshot-only boots need live check |

Consequences, all verified from the two tables, no new data invented:

1. B01 (1.5M UZS, old_money/Work) under LIVE numbers: cheapest loafer alone
   (1,299,900) + cheapest t-shirt (219,900) = 1,519,800 > cap — any
   loafer look is infeasible; cheapest knit look (fine-knit 1,186,196 +
   loafer 1,299,900) is infeasible before trousers. Only a sneaker-based
   look (1,424,702 theoretical floor) could fit — incompatible with the
   reference-first knit briefs. Abstention stands; bottleneck = catalog.
2. Variant inflation confirmed both directions: snapshot UZS slice holds
   ~5,322 rows vs 3,006 live active UZS SKUs (~2,300 stale rows), while
   missing the cheapest live tailored-trousers/sneakers/minimal-sneakers.
3. `pilot_12.mjs` now accepts `--snapshot-dir` so the live export can be
   dropped in and all three arms re-run unchanged. Export shape mapping:
   live JSON rows use the same keys as `data/` rows plus optional
   `fashion_slot`/`fashion_subtype` (used for the pool slot when present,
   adapter projection otherwise). Live `outerwear/pullover` knit items now
   project to `top/fine-knit` with `live_intelligence_role_projection`
   provenance (tested); non-knit pullovers and cardigans stay outerwear.

## Footwear: the binding constraint (verified 2026-11-10)

Live UZS shoe supply per the audit: loafer 23 (min 1,299,900), sneaker 21
(min 549,900), minimal-sneaker 11 (min 649,900). Reference briefs demand
loafer in ~2/3 of layered looks plus derby/boot/chelsea — categories with no
tabulated live UZS supply at all. Snapshot UZS shows the same shape at lower
prices (loafer 211 from 649,900; derby 234 from 1,499,900; boot 35 from
1,199,900; chelsea 12 from 779,900): the 649,900–779,900 tier that makes any
sub-1.5M outfit conceivable exists ONLY as stale snapshot variants, not in
live stock. Any staging pilot must either raise the floor price point or
accept sneaker-only budget looks — an inventory fact, not a retrieval bug.

## Image availability under the authorized-access policy (measured)

`scripts/stylist6/image_access.mjs` (`image-access-v1`): allowlist-only fetch,
plain client headers, audit-logged, private cache. Against the 152 pilot
product URLs: **59 fetched + 9 cache-shared (variant-shared photos) = 68
accessible; 72 blocked (unapproved hosts, incl. known-403 brand CDNs);
12 failed on approved hosts (dead media — strongest staleness evidence).**
Approval of a host = an explicit allowlist edit. No bypass, no substitution.
