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
- **0** proven ready-to-publish purchase offers: seven Tommy Hilfiger listings
  now carry time-limited **merchant-reported size-specific quantities**,
  but *none* has checkout-reserved stock or image display/inference
  permission. Listings and HTTP 200 are NOT enough. Country-wide delivery
  claims are not a confirmed checkout for a specific item, size, address
  or landed price.

The seven product images from FR Group returned HTTP 200 with `image/webp`
and were read via HEAD. **Bytes were not downloaded, hosted or supplied to
NVIDIA.** The seller's imaging availability is not a rights grant.

## Files

- `scripts/stylist6/curate_uz_footwear.mjs` — bounded, read-only merchant
  evidence collector; `node scripts/stylist6/curate_uz_footwear.mjs` updates
  the review package using fresh merchant detail pages (serial max 12).
  It deliberately has **no write/--apply path** to Supabase.
- `scripts/stylist6/fr_variant_stock.mjs` — exact model-ID / variant-ID
  read from the public merchant `GET /ru/catalog/get-product-info?productId=`
  endpoint; validates price, status, per-size quantity and expiry.
- `scripts/stylist6/fr_variant_stock.test.mjs` — no-network malformed
  merchant JSON, stale size evidence and API allowlist tests.
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
node --test scripts/stylist6/fr_variant_stock.test.mjs scripts/stylist6/curate_uz_footwear.test.mjs
node --test backend/stylist6/catalog_adapter.test.mjs
```

Before any subsequent activation the review must establish **per exact
merchant SKU and color:** current correct seller price/currency, size
availability rechecked just before checkout (site-reported quantities are not
reserved), checkout/delivery into UZ,
licensed product image display and separate image-inference rights, merchant
refresh cadence, stable ID reconciliation and zero duplication. Run a strict
staging dry-run against the full immutable catalog and human-reviewed
three-look scenarios. Production Stylist 5 and shared active shop_catalog
must not change without a separate owner-approved rollout/rollback plan.

## Direct public stock API check (2026-11-13)

The FR Group product page sets a numeric `productId`. After normal page load,
its storefront issues a read-only `GET /ru/catalog/get-product-info?productId=…`
which returns `modelId`, `salePrice`, `canBePurchased`, `isOutOfStock`,
`skuSortedList`, and `sizes[]` with EU size, precise SKU-variant ID and a
merchant-reported quantity. All seven previously shortlisted Tommy colorways
passed **model ID AND listing price matching** against that public response.

| Merchant model | Family | EU sizes reported | Seller-reported units |
| --- | --- | --- | ---: |
| FM0FM06276 / 171935 | boat shoe | 45, 46 | 2 |
| FM0FM05387 / 171927 | boat shoe | 40–45 | 9 |
| FM0FM06085 / 174797 | chukka, black | 40–45 | 15 |
| FM0FM06085 / 174815 | chukka, brown | 42–44 | 3 |
| FM0FM05891 / 164544 | leather tassel loafer | 43–45 | 4 |
| FM0FM05940 / 171923 | suede tassel loafer | 41–44, 46 | 5 |
| FM0FM05782 / 152421 | suede loafer | 43 | 2 |
| **Total** | **7 colorways** | **26 size variants** | **40 reported units** |

**Do not treat 40 as reserved or guaranteed stock.** Evidence includes a
30-minute refresh deadline; after it expires `assessFrVariantSize()` returns
`unknown_or_expired` until the product is checked again. Five tests check
malformed/mismatched model IDs, prices, unavailable flags, out-of-domain
endpoints and negative quantities. The collector retains a listing as
`unverified` if this endpoint fails. There is **no cart action, payment, or
purchase request**.

### Other approved brands: access / provenance blockers

Read-only direct requests to the nine previously search-curated URLs were made
on 2026-11-13. Mango's three pages returned **HTTP 429** and a Vercel
Security Checkpoint. Zara's five pages served only non-product bootstrap HTML;
a normal, unauthenticated Chrome attempt on one Zara listing returned
**HTTP 403 Access Denied**. Massimo Dutti's one page served a non-product
bootstrap page; a normal Chrome attempt returned **HTTP 403 Access Denied**.
The nine records intentionally retain `needs_direct_merchant_recheck`, NULL
freshness, and NULL product images. Do **not** attempt to bypass merchant
protection, invent item sizes/prices/photos, or reactivate stock because a
search result was indexed. An authorized official product export or seller
feed is needed for those brands.

**Quality implication:** adding seven variant-bearing colorways does not fix
the absence of locally verifiable Derby/Oxford selections. The human visual
comparison gate and image use permissions are still missing. Stylist 6 remains
production NO-GO; Supabase `shop_catalog` is unchanged.

## Additional authorized-brand sourcing leads — Mango UZ collection

The **official Uzbek Mango men's formal shoes category** (not individual
product pages) displays eight distinct article+color collection entries. We
captured only the publicly indexed model reference, name, color, listed UZS
price and displayed EU size range. The strict, offline, review-only additions
are in:

- `benchmarks/stylist6/MANGO_UZ_COLLECTION_LEADS_20261113.json`
- `scripts/stylist6/collection_leads.mjs`
- `scripts/stylist6/collection_leads.test.mjs`

Eight article/color references across **five merchant article numbers**:

| Family | Color variants | Indexed UZS prices |
| --- | --- | --- |
| Oxford leather | black / brown | 2,299,000 each |
| Derby/Blucher | black / brown | 1,299,000 each |
| Double monk strap | black / brown | 2,999,000 each |
| Formal shoe | brown split leather / black leather | 1,999,000 / 1,799,000 |

These **eight are leads, not eight additional verified products**. A category
size range is not stock at a specific size. They lack deep product links,
photos, licensed use and seller variant confirmations. The current local
Mango product pages returned HTTP 429 security checkpoints, and Zara and
Massimo Dutti returned HTTP 403 in Chrome. Do not bypass those barriers.
Some leads may overlap previously searched models; the exact product/color
identity must be verified before joining them with shop_catalog.

This is useful sourcing evidence for **Derby/Oxford/monk**, missing in the
current locally linked footwear pool, without weakening the admission gate.
The original 16-product candidate batch and seven directly size-checked
Tommy listings remain unchanged. A combined 24 references is **not** a claim
that 24 items can be purchased or published.

Command: `node --test scripts/stylist6/collection_leads.test.mjs`. No seller
network, DB write, product photography or NVIDIA invocation is performed by
that test.
