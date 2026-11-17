# 2026-11-10 Stylist 6 pilot post-run audit — canonical archive and live-catalog parity

**Repository branch:** `codex/stylist6-visual-first-20261110`
**Pilot SHA inspected:** `bcd1b8007f48f00b9f560ca8a149c0b7e58ef816`
**Production & staging remain untouched:** Supabase `generate-outfits` v207, `aiw-stylist-node-staging` v4.
**Verdict remains NO-GO for staging, pending unblinded results and actual image critic.**

## 1. The canonical 46-image ZIP IS available in the owner's ChatGPT conversation

The prior Mac Codex run truthfully reported that it did not find this archive on **that Mac**. However, a separate ChatGPT-hosted copy was successfully checked directly (not just guessed from the archive title):

- Archive: `AIWardrobe_Style_References_46_Pilot.zip` (private user artifact, **NOT in public GitHub**).
- Complete ZIP SHA256: `3f78b1ce3912705ab2a3b74cb0f65b53a85235baa571c110237ec4cc1b7f3e8e`.
- ZIP manifest dataset_id: `image-teaste-2026-11-10`; schema_version 2; 46 images.
- Each of the 46 manifest `sha256` hashes matched the actual JPEG inside the archive; 46 unique photo hashes; no missing images; total raw image bytes: 12,972,713; compressed ZIP bytes: 12,897,696.
- The **private uploader dry-run** completed: `{"result":"verified_private_dry_run","photos":46,"distinct_sha256":46,"total_bytes":12972713,"uploaded":0,"model_trained":false}`.

**Do not interpret this as a completed transfer to the Mac or Supabase Storage.** The owner needs to download the ZIP from their current ChatGPT conversation to their Mac, choose its actual local path, and rerun the committed `scripts/style_references/upload_private_reference_bundle.py --archive <local_path>` dry run before any optional permissioned private upload.

File permission/Instagram ownership scope must still be verified for those specific images; the ZIP itself does not prove copyright ownership. Do not put these photos in a public Git repository or existing public Supabase bucket.

## 2. The pilot used a materially different catalog universe than production

The committed offline report says `catalogRows=40989`, `catalogAdmitted=38388`, snapshot ID hash `239c266503dd056e` (see `benchmarks/stylist6/pilot_12_report.json`). Pilot catalog adapter checks its snapshot's `is_active` flag, HTTPS-looking URLs and positive numeric price.

**Independent live read-only Supabase query now says:**
- `shop_catalog`: 7,912 total rows, 7,020 currently active.
- All 7,020 currently active rows have positive prices and HTTPS image/shop URLs.

The 38,388 admitted **snapshot rows are not equivalent to 7,020 currently active production rows** merely because they have IDs and HTTPS URLs. The snapshot's age, merchant visibility, actual offer validity, canonical image availability, currency and active membership must be reconciled explicitly. Offline A/B on one frozen snapshot is useful for comparing algorithms, but is **not live-catalog acceptance testing**.

**Required before claiming the new engine ready for staging:**
1. Export a read-only, minimally necessary, recent snapshot of the **actual 7,020 active production SKUs** from the authorized Supabase project, including source IDs, authoritative price/currency, image/source URLs, and garment/visual intelligence with hashes. Keep private user data out.
2. Produce an ID-level intersection / snapshot-drift report: offline 38,388 vs live active 7,020, distinct base-product IDs, stale/invalid variants, missing, price changes, unverified imagery. Reject stale SKUs from final shopping output. Do not silently combine two universes.
3. Rerun pilot A/B/C with **the same current active catalog**. Do not quietly claim historical snapshot speed or 99 compliant looks applies to current stock.
4. Specifically measure the 1.5M UZS budget case on the active **live** catalog. The previous `1,584,702 UZS` minimum is only proven under the frozen reference/slot constraints and the pilot's UZS slice; do not generalize to all possible shopping outfits.
5. Old v207 domain kernel locally bundled for controls is useful but **not** evidence of a live authenticated v207 endpoint response. Document missing transport/quota differences honestly.

## 3. Visual data availability is the next quality gate

The report records reachable product images as just `87/152` from the pilot host. With 65 HTTP 403 failures, whole-look visual evaluation would skip many candidates by design. Do NOT spend on API calls before building an authorized and reproducible way to access the original product photos (merchant-permitted server cache; approved Supabase private mirror where license allows; or select only currently accessible catalog images). Never bypass CDN restrictions by spoofing cookies, pretending a broken URL is reachable, or silently replacing a garment photo.

Other true blockers: no configured authorized hosted vision API credentials, no paid model spend approval, and no three-human blind comparison of the 12 pilot cases. No win-rate can be claimed.

## 4. Minimal next action for Mac Codex

1. Owner downloads verified ZIP from their ChatGPT conversation to Mac and passes the actual local path; Codex verifies full ZIP SHA256 above and executes local private uploader dry-run.
2. Codex ensures creator rights permit hosted visual analysis of these particular media. Upload only to private, isolated experimental storage with explicit permission and service-key handling.
3. Before opening 108 model calls, fix/reference image accessibility, compare active catalog against frozen snapshot, and run **one** authorized multimodal test with total spend cap. Use API model/version pinned to a supported current model, not a stale identifier.
4. Copy the existing **24 blind comparison grids** and blinded review instructions out of local `/tmp/s6-pilot-20261110/` to a portable private ZIP, preserving the scoring key separately. A temp folder inaccessible to reviewers cannot produce blind votes.
5. Request three independent blind reviews. Record wins/losses and critical style defects. Only if the 12-scenario gate passes should engineering proceed to Node/iOS staging.

The goal is better outfits. Do not spend another week adding one-off garment-name rules or deploying a backend on the strength of mocked green tests.
