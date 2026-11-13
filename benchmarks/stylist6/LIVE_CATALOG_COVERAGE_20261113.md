# Stylist 6 — live retrieval coverage and release evidence (2026-11-13)

## Decision: fix catalog supply, not another style-rule loop

This is a live, **read-only Supabase audit** (project `fyqpifmrsftsfqibhwhy`) against the current catalog. It is **not** a claim that any listed item is in stock, deliverable to a specific country, correctly sized, or licensed for external visual inference.

- Active `shop_catalog` rows: **7,020**.
- Active rows whose authoritative `garment_type='shoes'`: **115**.
- Active enriched shoes passing a representative first-stage retrieval prefilter: **103** (full RPC also excludes certain model photos/unknown evidence).
- Current cloud discovery requests `p_limit=100` per slot; 100 is essentially the entire eligible footwear pool.
- Three different request-scoped seeds produce 100 rows per slot each, with **98–99 shoe SKUs shared** between any two windows. By contrast, top overlap is 32–39, bottom 31–36 and outerwear 38–43 (out of 100).
- No client-side diversity penalty can manufacture additional real, purchasable shoes. The pool must be expanded with verified merchant catalog data.

### Actual 100-item discovery-window overlap

| slot | seed A vs B | A vs C | B vs C |
| --- | ---: | ---: | ---: |
| top | 32 | 33 | 39 |
| bottom | 36 | 35 | 31 |
| outerwear | 43 | 39 | 38 |
| shoes | **99** | **98** | **98** |

Seeds used: `aaaaaaaaaaaaaaaa`, `bbbbbbbbbbbbbbbb`, `cccccccccccccccc`; each server-side seed is scoped as `s6cloud:<slot>:<seed>`. This measures catalog discovery coverage, **not** the diversity of final outfits or user visual preference.

### Most common active shoe sources

| source | active shoes |
| --- | ---: |
| Zara UZ white-bg | 39 |
| Geox official gallery | 38 |
| Thursday official | 12 |
| Geox UK new arrivals | 10 |
| ECCO UK product gallery | 10 |
| Mango GB product-only | 3 |
| Massimo Dutti UZ | 2 |
| Zara UZ new | 1 |

Archived inputs `data/shop_catalog_geox_shoes_20261030.json` and `data/shop_catalog_ecco_20261030.json` each contain 120 captured records; however only 48 Geox and 10 ECCO records from those named active sources are active live. **Do not blindly import or re-activate these old records**. Archived `is_active` is not fresh stock/delivery evidence, and a live HTTP 200 page is not proof the buyer can check out in the target country.

## Reproduce without mutation

```sql
-- 1. Availability of truly classified shoes
select count(*) as active_total,
       count(*) filter (where garment_type='shoes') as active_shoes
from public.shop_catalog where is_active;

-- 2. Distinct and shared IDs in three independent 100-item discovery windows
with seeds(s) as (values('aaaaaaaaaaaaaaaa'),('bbbbbbbbbbbbbbbb'),('cccccccccccccccc')),
slots(slot) as (values('top'),('outerwear'),('bottom'),('shoes')),
picks as (
  select s.s as seed, k.slot, c->>'id' as id
  from seeds s cross join slots k
  cross join lateral public.catalog_discovery_candidates(
    p_slot=>k.slot,
    p_seed=>('s6cloud:'||k.slot||':'||s.s),
    p_limit=>100
  ) c
)
select a.slot,a.seed as seed_a,b.seed as seed_b,count(distinct a.id) as shared
from picks a join picks b on a.slot=b.slot and a.id=b.id and a.seed<b.seed
group by 1,2,3 order by a.slot,a.seed,b.seed;
```

## Release status and explicit acceptance path

**Isolated staging:** Supabase `aiw-stylist6-node-staging` v20 ACTIVE, JWT verified, pinned to READY Vercel Preview of compute commit `ffd8ae631d3aec8fe5d03c80f4b269bfe213f0a9` (cross-SKU taste, full-deck visual review, request-rotated discovery). Source proxy updated on PR #15 at `db2416660a5df67922a3fa585a2eb84cef2ff0a5`. **Production: Stylist 5 v209, unchanged**. Current source-controlled Stylist 6 integration tests passed; general CI on latest commit may still be running.

Live request claims (read-only as of this audit): 21 terminal `200/complete` on 2026-11-12 and one `403/feature_limit_reached` on 2026-11-13. They **do not** prove that NVIDIA reviewed each displayed look, that the newest v20 was exercised with an authenticated entitled user, or that a retailer has exact stock.

**Next work is supply + genuinely independent quality measurement, not adding garment bans:**

1. Obtain a current authoritative footwear feed for one explicit destination market, with product/variant ID, size-specific stock, total landed price and checkout region. Verify display and third-party inference image rights separately. Prefer regionally purchasable sources.
2. Import only checked active variants; do not resurrect stale listings or equate a valid URL with purchase eligibility. Measure authentic distinct shoe pool growth and re-run the window overlap query. A useful **measured coverage goal**, not a styling rule, is ≥200 independently eligible shoes and ≤60/100 shared shoes across typical seed pairs, then tighten if real data supports it.
3. Re-run held-out 3-look evaluations with real eligible inventory and actual image-permission coverage. Distinct SKU identity is necessary but not sufficient for different silhouettes and footwear register. Require visual critic receipts for all three displayed cards or honest partial/abstention. Measure real wall time, token cost, catalog/stock evidence, and failure rates.
4. Do not treat the existing 5–5 blinded NVIDIA-family AI judgment (10 comparable cases; wide CI) as superiority. Use a larger independent blind comparison and/or human preference ratings with a predeclared success threshold, and retain safe rollback.
5. Promote only after an authenticated v20 smoke and quality gates are actually satisfied; no fake NVIDIA_APPLIED badge, no production cutover based on deterministic self-scores alone.

This plan breaks the rules-and-patches cycle by separating **data coverage**, **visual judgment evidence**, **independent taste success**, and **release/retail truth**. If a constraint fails, improve the underlying data or model evidence rather than inventing a more specific blacklist.
