# @sainttnoirclub — authorized Instagram source for AIWardrobe

**Status: OWNER PERMISSION REPORTED / IMAGE ARCHIVE NOT YET OBTAINED.**  
**Date:** 2026-11-10  
**Repository branch:** `codex/stylist6-visual-first-20261110`  
**Never claim any images were downloaded from this source until the manifest and stored files have been verified.**

## What has been verified externally

- Supplied Instagram profile URL: https://www.instagram.com/sainttnoirclub/
- Search index attributes to **this exact handle** at least two public Reel descriptions:
  - “Spring/summer old money outfits for men” — from https://pictame.com/en/discover/hashtag/old-money-fashion-style
  - “The type of man I’m trying to be” — from https://pictame.com/en/discover/hashtag/aesthetic-wallpaper-for-men
- This is evidence about captions/categories, **not media extraction, photo quality, number of posts, source ownership of reposted material or complete archival coverage.**
- Attempt to open the Instagram profile directly from this environment failed; no direct original images were obtained.
- Project owner reports explicit creator authorization to download/store/use account-owned images for model development. Metadata status: **user-attested, not independently verified**. A creator cannot grant rights to third-party copyrighted photographs if they do not own or control those rights.

## Obtain exactly the account's original media without passwords

The creator can request Instagram's official data export:
**Instagram → Settings and activity → Accounts Center → Your information and permissions → Download your information → Download or transfer information → select @sainttnoirclub → Some of your information / content / posts → Download to device**, choose all available dates and highest available media quality. Prefer JSON for captions/timestamps. Instagram menu wording may vary.

Ask for an export containing ONLY public media/posts and associated metadata, not passwords, DMs, account settings or security data. Do not paste session cookies, access tokens or login credentials into GitHub, ChatGPT, or public services.

Run locally after receipt:

```bash
python3 scripts/style_references/import_instagram_archive.py \
  --source /private/sainttnoirclub-posts.zip \
  --output /private/aiwardrobe-sainttnoirclub \
  --account sainttnoirclub \
  --owner-authorized
```

The importer:
- reads ZIP members without extracting untrusted paths;
- rejects ZIP path traversal, symlinks, unsupported file signatures and overly large archives;
- limits itself to post photos, excluding DMs and Stories by default;
- hashes original images and deduplicates identical bytes;
- preserves captions/creation timestamps when the export contains matching JSON paths;
- saves a private `manifest.json` and image files named by SHA256;
- marks all imported photos **pending review**, and approved count is 0 by default.

**Never commit raw photos into the public source repository.** Keep media outside Git and use a private Supabase Storage bucket only after reviewing bucket policies, rights, authorization and data retention. The module `backend/stylist6/reference_library.mjs` allows only curated/approved images as reference inputs. There is no live ingest to Supabase yet.

## Curation and intended ML use

Do NOT train a general-purpose image model by simply ingesting this entire account. For a reference-first stylist, mark each image:
- `rights_confirmed=true` only after confirming that owner controls the actual photograph, including reposts;
- `contains_full_outfit=true` only if a complete, visible outfit is present;
- `review_status=approved_for_reference_use` after curator review;
- `style_tags`, `occasion_tags` and `quality_score` (0–1) with human stylist evidence;
- optional descriptions of garment roles/silhouette/palette, with observations distinguished from guesses.

A dedicated evaluator then learns to **retrieve and compare** actual catalog products against the outfit aesthetic. The original reference image is not automatically used as a sale image or product listing. The catalog's true source links, SKU IDs, stock data and budgets remain authoritative.

## Acceptance gates before claiming integration

1. Manifest account equals `sainttnoirclub`; N images copied, N distinct SHA256 hashes, local files all exist; source photos clearly identified.
2. Curator-approved outfit references count recorded, distribution across styles/occasions/season and outfit completeness audited.
3. Real reference-to-product retrieval benchmark against current AIWardrobe catalog run offline. Never assume look-alike nearest neighbours yield complementary garments.
4. Blind 12-case reference-first versus creative-first versus Stylist 5 evaluation as defined in `docs/architecture/STYLIST6_DECISION_GATE.md`; expand to 30+ only if credible improvement.
5. No Supabase production `generate-outfits` change, no paid tests and no merge until security, quota, actual visual quality and human-review gates are met.

**Current actual import count: zero** (creator media ZIP not yet available in this environment). The ingestion pipeline and automated synthetic tests exist; these are preparation, not proof of recovered Instagram photos or a completed AI stylist.
