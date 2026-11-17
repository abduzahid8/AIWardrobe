# Stylist 6 — licensed, exact-SKU original product photo cache

**Status: foundation deployed, merchandise media NOT backfilled, photo-first
styling NOT enabled.** Owner acceptance and 70% blind human win are pending.
This is a P0 infrastructure dependency of [issue #17](https://github.com/abduzahid8/AIWardrobe/issues/17).

## Observed root cause, 17 November 2026

Real authenticated iPhone runs of the creative AI on commits `dbefcdd`,
`6ae1c2c`, `45428a8` returned
`skuCompositionPhotoFirst:deadline` every time. Fall back to text
composition was source-safe, but the independent NVIDIA photo judge approved
only 1/6 looks in the last two attempts. More tweaks to live CDN timeouts
are NOT a fashion-quality solution. The cloud request is not the place to
download forty-eight independent merchant photos from twenty CDN hosts.

The active catalog has 8,554 items. Supabase Storage `shop-catalog`
contains 3,853 older objects, yet only **1,534 active image URLs point
to those assets, all SHOES**. No current TOPS/BOTTOMS have first-party
catalog URLs. Existing `vendored/` photographs cannot be presumed to be
present-day SKU images just from file names. Existing host network access
and NVIDIA inference allowlists are NOT legal hosting/inference licenses.

## New source and rights binding

The server-only `public.catalog_photo_cache_v1` metadata table is an
additional, secure opt-in. It does not mutate any original `shop_catalog`
row or the production Stylist5 generation function. Every READY row needs:

- EXACT original current `shop_catalog.id` and `image_url`;
- SHA256 of `JSON.stringify([id,image_url])` for immutable source binding;
- approved, documented `rights_basis` (`owner_upload`,
  `licensed_provider`, `merchant_contract`), `rights_reference`,
  hosting AND external-AI grants, granted date and optional expiry;
- authenticated bytes hash `image_sha256`, verified JPEG media,
  permitted first-party storage object path and verification within 30 days;
- `status:ready`, with ability to mark `revoked`.

RLS is enabled, and `anon` / `authenticated` have NO access. Only
server-side `service_role` can use this metadata. Public object storage
is acceptable **only with confirmed public-hosting rights**. Production
user uploads and private wardrobe photos are NOT candidates.

## Admin-ingestion workflow

No automatic bulk scraper, no user JWT access, no aggressive merchant retries.
The admin provides an explicitly authorized private manifest, e.g.:

```json
[
  {
    "productId": "EXACT_CURRENT_CATALOG_ID",
    "sourceImageUrl": "https://APPROVED_HOST/REAL_ORIGINAL_SKU_IMAGE.jpg",
    "rightsBasis": "merchant_contract",
    "rightsReference": "owner-audited contract or permission evidence",
    "rightsGrantedAt": "2026-11-17T00:00:00Z",
    "rightsExpiresAt": "2027-11-17T00:00:00Z",
    "hostingPermission": true,
    "externalAiInferencePermission": true
  }
]
```

The URLs/rights shown are **format examples, not permission claims**.
Never mark a merchant URL as licensed merely because it returns HTTP 200.
Use a private, uncommitted path to this manifest, and run the CLI only
on an authorized admin host with environment variables `SUPABASE_URL`,
`SUPABASE_SERVICE_ROLE_KEY`:

```bash
node scripts/stylist6/ingest_licensed_product_photos.mjs --manifest /private/licensed-grants.json
# Only after reviewing the dry-run and checking grants:
node scripts/stylist6/ingest_licensed_product_photos.mjs --manifest /private/licensed-grants.json --apply
```

Without `--apply` the tool NEVER downloads or uploads photos. With
`--apply` it checks original active SKU, seller and price, downloads
the approved original (no redirects, time- and byte-bounded), optimizes
a JPEG, stores the source-addressed path, then RECHECKS the current
source URL before recording the READY provenance row. Stale or
unlicensed data aborts. A failed upload or changed source is NOT ready.

To revoke permissions: authorized admin must mark the exact metadata
record `revoked` and remove its public storage object according to the
source rightsholder's instruction. This operation is NOT an automatic
scheduled task yet; do not use indefinite rights without review.

## Staging request integration

The existing cloud handler uses the privileged cache table only when
BOTH opt-ins are explicitly configured on the 72-second Render staging:

```
STYLIST6_PHOTO_FIRST_COMPOSER_ENABLED=true
STYLIST6_PHOTO_ASSETS_READY=source-hash-verified
```

Even with the flags, a request must discover at least three
source-and-rights-verified photographic products **per top, layer,
bottom and shoes role**. The new creative resolver reads ONLY the verified
first-party assets; it may not fan out to forty-eight merchants.
The independent quality critic still examines actual photographs and
may fall back to the original source image when cache validation fails.
No licensed media coverage => text AI-first composer; no fake visual
approval, no degraded production Stylist5.

The cache resolver checks source equality, provenance and rights before
contacting storage, then recomputes the actual JPEG byte checksum. A
nonmatching, revoked or tampered cached image is never shown as verified
visual evidence; the original source path remains available under the
existing photo authorization and full-outfit judge.

## Acceptance (NOT YET PASSED)

1. Valid signed rights / provenance for at least 3×4 real current source
   images in ONE discoverable inventory segment, then >=12 per slot.
2. Independent source-photo manual spot check of the exact merchandise
   pictured by each current source URL, no substituted stock shots.
3. Native authenticated iOS stage logs
   `skuCompositionPhotoFirst:applied`, photos ready, THREE distinct
   source-bound four-product candidates, independent photographic
   approvals, no unreviewed padding, <72s.
4. No new 4XX/5XX, no elevated quota, no Stylist5 production changes.
5. **World-class product requirement:** >=70% preference for Stylist6
   versus Stylist5 in 30+ held-out blind cases, three real human reviewers.
   Passing tests or seeing three cards is not fashion acceptance.
