# Stylist 6: Uzbekistan footwear candidate import — first verified tranche

**Date:** 2026-11-13
**PR:** #15 (`codex/stylist6-visual-first-20261110`)
**Owner brand scope:** Zara, Massimo Dutti, Mango Man (normalized `MANGO`), Tommy Hilfiger.
**Forbidden:** ECCO, Cabani, Basconi (never added to the new tranche).

## Result and honest boundary

This change **prepares and validates** a deterministic review-only footwear
candidate package; it **does not publish** shop_catalog products, reactivate
archived merchandise, run paid AI, or make a production deployment.

- **16 candidates** from 4 brands: 7 Tommy Hilfiger / 5 Zara / 3 MANGO /
  1 Massimo Dutti.
- **7** Tommy Hilfiger candidates were read **directly from live merchant
  product pages** (same-day structured JSON-LD), with exact SKU, current
  listed UZS price, product URL, vendor product image URL and listing-level
  `InStock` / UZ shipping claims.
- The other **9** are sourced from official Uzbekistan product pages
  surfaced through public search. Their exact current merchant price,
  photo, and variant stock still require direct merchant rechecks.
  They have `observedAt:null` (not a made-up last-seen timestamp).
- **3** Tommy Hilfiger URLs were found already archived as
  `is_active=false` in live Supabase. Do not add duplicates or simply
  reactivate them. Their stored prices differ from current merchant
  **listing** prices by -656,700 / -878,700 / -276,900 UZS. The correct
  action is a separately approved archived-row reconciliation *after*
  checking variant availability and image rights.
- **13** candidates did not match an existing shop_catalog URL during
  the read-only exact URL check. Only **4** of these have both a directly
  observed seller product and an image; the other 9 need a direct recheck.
  `FOOTWEAR_UZ_INACTIVE_PREVIEW_20261113.json` is a **four-row,
  `is_active=false` DB-shaped preview**, never an upsert instruction.
- **0** proven ready-to-publish purchase offers: all items lack verified
  size-specific stock and image display/inference permission. Listings and
  HTTP 200 are NOT enough. Country-wide delivery claims are not a
  confirmed checkout for a specific item, size, address or landed price.

The seven product images from FR Group returned HTTP 200 with `image/webp`
and were read via HEAD. **Bytes were not downloaded, hosted or supplied to
NVIDIA.** The seller's imaging availability is not a rights grant.

## Files

- `scripts/stylist6/curate_uz_footwear.mjs` — bounded, read-only merchant
  evidence collector; `node scripts/stylist6/curate_uz_footwear.mjs` updates
  the review package using fresh merchant detail pages (serial max 12).
  It deliberately has **no write/--apply path** to Supabase.
- `scripts/stylist6/curate_uz_footwear.test.mjs` — no-network CI tests for
  source host and currency allowlists, disallowed brands, scope honesty,
  SKUs, dedupe, inactive preview, offline slot feasibility, image rights and
  zero publication.
- `benchmarks/stylist6/FOOTWEAR_UZ_CANDIDATES_20261113.json` — candidate
  products and provenance, statuses, merchant evidence hashes only.
- `benchmarks/stylist6/FOOTWEAR_UZ_EXISTING_INACTIVE_20261113.json` —
  prior archived row equality matches from **read-only** Supabase.
- `benchmarks/stylist6/FOOTWEAR_UZ_INACTIVE_PREVIEW_20261113.json` —
  four prospective new inactive rows with merchant SKU, photo URL,
  observed date, native shoes role; **not imported**.

## Examples and source links

- FR Group: <https://frgroup.uz/ru/store/man/obuv/tommy-hilfiger>
- Zara: <https://www.zara.com/uz/ru/кожаные-туфли-p12492820.html>
- Mango: <https://shop.mango.com/uz/en/p/men/shoes/leather/gathered-leather-loafers/47091370/99/00>
- Massimo Dutti: <https://www.massimodutti.com/uz/rasmiy-oksford-poyabzali-l12701451>

Explicitly ignored:
- Generic Zara/Mango category listing with no stable variant ID.
- A Zara suede loafer page labeled `НЕТ В НАЛИЧИИ`.
- Massimo Dutti loafers labeled `notify me` without verified stock.
- Tommy Hilfiger `CHUKKA RUNNER` (sports runner masquerading as dress boot)
  and espadrilles (outside the shortage priorities).
- Any archived shoes auto-reactivated solely from old catalog data.

## Commands and staging gate

```bash
# Read-only first-party merchant sources (requires external network):
node scripts/stylist6/curate_uz_footwear.mjs

# No network, no store access, no mutations:
node --test scripts/stylist6/curate_uz_footwear.test.mjs
node --test backend/stylist6/catalog_adapter.test.mjs
```

Before any subsequent activation the review must establish **per exact
merchant SKU and color:** current correct seller price/currency, size
availability (not only listing-level InStock), checkout/delivery into UZ,
licensed product image display and separate image-inference rights, merchant
refresh cadence, stable ID reconciliation and zero duplication. Run a strict
staging dry-run against the full immutable catalog and human-reviewed
three-look scenarios. Production Stylist 5 and shared active shop_catalog
must not change without a separate owner-approved rollout/rollback plan.
