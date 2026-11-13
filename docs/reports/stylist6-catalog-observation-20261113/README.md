# Stylist 6 — recover catalog observation timestamps in bounded cloud discovery

**Date:** 2026-11-13. **Branch:** `codex/stylist6-visual-first-20261110` / PR #15.
**Scope:** Isolated Stylist 6 Node backend only. No DB mutation, migration,
production switch, merchant API usage, or paid NVIDIA request.

## Reproduced bug

The live `shop_catalog` contains `last_seen_at`, but
`catalog_discovery_candidates` returns a fixed JSON projection without it
(`backend/supabase/migrations/20261107225500_catalog_footwear_source_guard.sql`).
Thus `adaptCatalog` always saw `last_seen_at=undefined` on real cloud
discovery and assigned `priceFreshness=unknown` to every item, regardless of
whether the source catalog import date was known.

Read-only SQL on 2026-11-13 confirmed **7,020 active listings**,
**2,554 with known** import observations, **4,466 unknown**. None of the active
listings was observed in the past 7 days. These are **feed import records,
not merchant-confirmed stock, checkout/size availability, refreshed prices or
shipping**.

Representative live discovery sample using the exact S6 per-slot limit=100
and request seed `aaaaaaaaaaaaaaaa`:

| Slot | Returned | Known import date | Unknown | Last seven days |
| --- | ---: | ---: | ---: | ---: |
| top | 100 | 46 | 54 | 0 |
| bottom | 100 | 25 | 75 | 0 |
| shoes | 100 | 38 | 62 | 0 |
| outerwear | 100 | 36 | 64 | 0 |
| TOTAL | 400 | 145 | 255 | 0 |

This confirms **material loss of existing provenance in the Node/RPC
boundary**, rather than missing AI ranking confidence.

## Implemented (compatible with existing live RPC)

New `backend/stylist6/catalog_observation.mjs` runs a **GET-only,
service-role-server-side** reread of only the previously selected catalog IDs
and the small set of existing canonical columns. Requests are bounded to 400
SKUs, at most 100 IDs per request and 5,400 encoded ID bytes per page. They
execute as a single bounded parallel batch with a 2.2-second per-call timeout.

- Reread observations attach only when current `shop_catalog` identity,
  active flag, source URL, image, category, garment type, currency and exact
  numeric price all match the discovery row. Mismatched, deactivated or
  deleted product snapshots are *excluded*, not silently repriced.
- Missing/invalid `last_seen_at` remains unknown; future timestamps are
  **never** declared fresh.
- Network failure or unexpected response is an all-or-nothing observation
  fallback to UNKNOWN recency; no partial "recent" provenance.
- If the RPC ever projects `last_seen_at` directly, the reread is skipped.
- Neither this timestamp nor an HTTP 200 means in-stock. Size, local delivery,
  checkout and image rights still require the independent merchant-offer
  evidence mechanism.

The source code change does **not** invent new merchant authorization
privileges; the existing cloud Node backend already uses a service key for
server-side RPC reads. No service key is sent to the user or the client.

## Local verification

```bash
node --test backend/stylist6/catalog_observation.test.mjs
node --import=tsx --test backend/stylist6/cloud_handler.test.mjs
npm --prefix backend/stylist-node run build
```

Tests cover real RPC-shaped omission, null/future dates, mismatched/deleted
products, duplicate/unexpected response rows, failed metadata pages,
untrusted SKU strings, bounded query sizes, absence of extra reads when
future RPC projections already contain observations, and unchanged quota
admission. The dedicated Stylist 6 CI workflow runs the observation tests.

## Remaining release constraints

- End-to-end hosted v20 verification is still required; shipping a READY
  Preview is not equivalent to staging being repinned to that Preview.
- Live price/size/stock/destination are NOT verified and should not appear
  as guarantees. There is no existing merchant-offer table in the current
  live public schema; the offline merchant snapshot validator is not a
  current shopping feed.
- Measure added read-only REST overhead on hosted staging before release.
  The pure tests and a 400-row SQL join prove correctness, not live p95.
- Product catalog still has only **115 active shoes**; supplier inventory
  freshness and authorized merchant feeds remain the dominant product
  bottleneck. The human independent Stylist 5 versus Stylist 6 blind quality
  gate also remains unpassed.
- Production Stylist 5 v209 remains untouched; **NO-GO**.
