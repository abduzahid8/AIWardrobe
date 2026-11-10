# AIWardrobe Stylist 6 — 46-image fashion-reference intake and catalog coverage

Date: 2026-11-10. Data came **only from the user-uploaded `image-teaste.zip`**; no Instagram post links or account provenance appeared in the archive.

## Actual media audit

- ZIP: 93 entries including macOS resource forks and a directory; **46 full 1920×2560 JPG outfit collages**, photo_1 through photo_46; all SHA256 hashes distinct.
- The uploaded collages depict flat-lay **menswear complete looks** (top, trousers, footwear, generally a layer, with watches, eyewear or fragrance as decorative accessories). They are NOT model-worn outfit photographs. No on-body fit or garment SKU can be inferred from them.
- Reviewed and labeled every collage, with approximate visual top/layer/bottom/shoes, palette, seasons, purpose and silhouette. Manual annotations are *medium-confidence* and materials may only be visual guesses.
- Counts across the 46 annotated examples: **39 layered**, **7 without outer layer**, **35 with knit-like or roll-neck base tops**, **26 with explicitly high-neck/roll-neck/mockneck top**, **29 with visually depicted loafers**.
- Season tags are overlapping: **autumn 35**, **winter 23**, **spring 23**, **summer 9**. These are not mutually exclusive counts.
- Dataset is heavily concentrated in Old Money / Ivy / subdued smart-casual. Over-representation means we MUST NOT let its style proportions define all future styles. Add independent casual/streetwear/sporty/reference examples before considering universal training.

## Verified server-side shop catalog, read-only Supabase SQL

Supabase project `fyqpifmrsftsfqibhwhy` (production data read ONLY; no tables, functions, policies or storage buckets changed):
- shop_catalog historical rows previously 7,912; **7,020 currently active** with valid HTTPS product image and source link plus priced quote at time of this check.
- Active intelligence slots: **top 3,082; outerwear 2,117; bottoms 1,672; shoes 115** (unrecognized and outfit records explain difference to total).
- Active `shoes/loafer` **25**, `shoes/derby` **8**, `shoes/minimal-sneaker` **11**, `shoes/chelsea` **4**.
- Active `top/fine-knit` **15**, but `outerwear/pullover` **623**. An additional name-based check finds `turtleneck/roll-neck/mockneck` under top as 29 (25 subtype=unknown + 4 subtype=fine-knit) and 12 classified as outerwear/pullover. This indicates **role-versus-taxonomy mismatch** for reference construction; it is not proof that all pullovers are acceptable underneath a blazer. Design an outfit-wearing-role adapter backed by visual evidence and real compatibility QA, not an unverified mass reclassification.
- Active `outerwear/blazer` **218**, `outerwear/cardigan` **120**, `outerwear/coat` **111**, `bottom/tailored-trouser` **222**. These are adequate starting categories, but images/colors/silhouettes need human evaluation.
- 25 loafers include multiple color variants of the same silhouette and many with quoted prices above 1.6M UZS; this can limit low-budget Old Money lookbook recreation. It is a catalog-coverage problem, not one to fix by turning off price constraints.

## Experiment integration actually committed

- 46 human-reviewed **blueprints** in `benchmarks/stylist6/references/image_teaste_46.json`. No photos are publicly committed, and blueprint media_sha256 remains null until the local importer validates the exact bytes.
- `backend/stylist6/lookbook_briefs.mjs` converts visual outfit-level examples into 6–12 diverse structured briefs matched to requested style/occasion/season, without fabricated SKU IDs.
- `backend/stylist6/prototype.mjs` preserves `referenceId` from each selected brief through the visual critic payload and the selected outfit concept, so a provider adapter can resolve the validated full reference photo in a private vault.
- `scripts/style_references/import_flatlay_archive.py` verifies all 46 original JPEGs, the ZIP path shape, SHA uniqueness, and writes hash-addressed images + manifest outside Git. It does **not** contact Instagram/Supabase or enable any model training.
- `backend/stylist6/lookbook_briefs.test.mjs` runs end-to-end mocked visual-critic tests and checks the actual 46-record index. `scripts/style_references/test_import_flatlay_archive.py` uses synthetic 46-member archives to test safe import.
- CI workflow `.github/workflows/stylist6-visual-first.yml` verifies both new test suites, previous prototype tests and importer security contracts. **Mock success is not real fashion-quality evidence.**

## Files and rights

The actual original photos remain in the user's uploaded ZIP and a separately exported **private pilot package** in the conversation. They have not been copied into GitHub or any Supabase bucket. The user said they have permission from the Instagram account owner, but the ZIP does not prove it originates from that account or who owns the photographs inside. Source account and post URLs are therefore **unverified**. Evaluate privately first; confirm rights for these individual assets before commercial training, hosting or public re-distribution.

## State and next objective (no false GO)

**Done:** 46 visual analyses, checksum-backed local media package, 46-record style blueprint index, prototype adapter and tests, catalog feasibility diagnosis.

**Not done:** private remote media upload, real hosted image-to-image/reference critic, user-authorized positive authenticated API generation, actual V5 vs S6 blind outfit scoring, automated cross-catalog outfit validation with exact market prices. The app is NOT yet using this reference collection.

**Pilot next:** Replay 12 fixed cases with the 46 looks as example briefs and actual catalog products; separate 'not enough appropriate products' from 'style model chose badly'; have human judges compare resulting ensembles to real V5 outputs before further architecture investment. Do not deploy/modify `generate-outfits` v207 or existing `aiw-stylist-node-staging` v4.

**Release state: NO-GO for production, GO for isolated offline reference-first research only.**
