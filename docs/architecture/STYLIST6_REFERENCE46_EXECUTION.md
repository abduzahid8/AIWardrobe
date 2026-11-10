# UPDATE 2026-11-10 — SOURCE FOLDER `image-teaste` EXISTS ON OWNER'S MAC

**THIS OVERRIDES THE "MUST DOWNLOAD CANONICAL ZIP" STEP BELOW.** The user showed a Finder/IDE screenshot of a directory named `image-teaste` next to `i18n`. Do NOT assume it is in Downloads; locate the directory in the actual AIWardrobe checkout first. The owner's Mac is NOT directly mounted into this ChatGPT session, but Codex on that Mac can inspect it.

The repository now contains `scripts/style_references/verify_image_teaste_folder.py` with a canonical **SHA-256 aggregate over all 46 original JPEG filename + SHA-256 pairs** independently derived from the verified ZIP in this conversation:

`8b7b47845d678d15952dce6b1b16a42aeb4d94955e4ba08b8d0ede828167f04f`

Run from the actual local checkout:

```bash
git switch codex/stylist6-visual-first-20261110
git pull --ff-only
python3 scripts/style_references/verify_image_teaste_folder.py --folder ./image-teaste
# If passing repo root instead:
python3 scripts/style_references/verify_image_teaste_folder.py --folder .
```

A successful result MUST report `verified_canonical_46_photo_folder`, `count:46`, `total_bytes:12972713`, and `uploaded:false`. If any source byte differs, stop — do not silently accept unknown Telegram photos.

If the source folder matches, prepare a private upload-compatible ZIP **without downloading or recreating the exact old ZIP bytes**:

```bash
python3 scripts/style_references/verify_image_teaste_folder.py \
 --folder ./image-teaste \
 --output-private-bundle "$HOME/Downloads/s6-reference46-private-repacked.zip"
python3 scripts/style_references/upload_private_reference_bundle.py \
 --archive "$HOME/Downloads/s6-reference46-private-repacked.zip"
```

The newly repacked ZIP has a different *ZIP-level* hash by design; all 46 original JPG hashes are preserved and validated. **This is dry-run only.** Supabase upload still needs explicit owner authorization, reviewed image rights and private server credentials. Do not upload into a public bucket or publish photos into the public GitHub repository. The `.gitignore` explicitly excludes `image-teaste/`.

Once the original images are verified, continue the actual **live-catalog drift comparison** with the existing `scripts/stylist6/catalog_drift.mjs` and the model-backed, authorized, single-request image test; then gather three independent human blind reviews. Do not skip the 12-scenario quality gate. The hosted critic defaults to `gemini-3.8-flash`; its generation config has been made compatible with the current Google migration requirements (no legacy sampling temperature in 3.8 requests).

---

# CODEX HANDOFF — complete a real Reference-First Stylist 6 pilot

**Updated:** 2026-11-10
**Repo:** https://github.com/abduzahid8/AIWardrobe
**Branch:** codex/stylist6-visual-first-20261110
**Status:** Private image upload prepared but NOT EXECUTED; fashion evaluation NOT PROVEN.
**Safety:** Production Supabase \`generate-outfits\` v207 and existing \`aiw-stylist-node-staging\` v4 must remain unchanged. This supersedes any mandate to ship before measuring quality.

## User goal (non-negotiable)

Build a genuinely useful stylist that produces 3 fashionably convincing, visually distinct, shop-able outfits tailored to style, occasion, budget and weather, grounded in actual merchandise. No more isolated regex bans. Prioritize outfit aesthetic quality and real human evaluation, not green code tests.

## Current handoff facts verified by chat and GitHub

- The user's **private, user-supplied** \`AIWardrobe_Style_References_46_Pilot.zip\` is available as a download **in their ChatGPT conversation**. It is NOT publicly accessible or available inside the repository: user or local agent must obtain it from the user's private Downloads folder, without importing photos into public Git.
- The ZIP's 46 unique images + exact SHA256 are in \`manifest.json\`; \`reference_manifest.json\` and \`reference_vectors_ready.json\` contain manual visual role/season/color annotations. The images are menswear flat-lay full-outfit collages, not worn-on-human photos. Source Instagram ownership/post URLs remain **unverified**, even though the user reported permission from @sainttnoirclub's owner. Respect permission scope for these particular uploaded files; no commercial retraining or public hosting until evidence covers their actual owners.
- In the repo, \`benchmarks/stylist6/references/image_teaste_46.json\` has 46 structured *blueprints* without original binary bytes. \`backend/stylist6/lookbook_briefs.mjs\` maps the reference concepts to a distinct set of possible outfit roles. \`backend/stylist6/prototype.mjs\` is an **offline unprotected prototype**, never an app endpoint; its injected visualCritic is **MOCKED** in existing tests.
- Catalog at last inspection: 7,020 active priced HTTPS products: 3,082 tops, 2,117 layers, 1,672 bottoms, 115 shoes. Of these, 25 catalog loafer SKUs and only 15 \`top/fine-knit\` are identified; there are 623 \`outerwear/pullover\`. Treat subtype/role misclassification and sparse footwear as potential bottlenecks, not fabricated evidence about actual item colors. See \`benchmarks/stylist6/references/catalog_subtype_coverage_20261110.json\`.
- A private-only upload tool has been committed: \`scripts/style_references/upload_private_reference_bundle.py\`. Default is **DRY-RUN** and verifies all 46 hashes, never logs secrets; opt-in upload requires environment credentials and confirmation. Its test is \`scripts/style_references/test_upload_private_reference_bundle.py\`, exercised by \`.github/workflows/stylist6-visual-first.yml\`.
- The public GitHub repo ignores original archives, private vaults and receipts. Never push photos/raw service credentials to GitHub.

## Required execution — perform in this order

### 1. Materialize a private source of truth

On owner's local Mac (not a public CI runner), locate the private ZIP in Downloads. Do not invent its path. Verify the exact SHA dataset:

\`\`\`bash
python3 scripts/style_references/upload_private_reference_bundle.py \
  --archive "/private/path/AIWardrobe_Style_References_46_Pilot.zip"
\`\`\`

Expected: 46 valid images, 46 unique SHA256, zero uploads and no model training.

Confirm original author rights/permission scope for all 46 images (not just the Instagram handle). If the archive cannot be legally used by hosted ML providers, stop upload, keep metadata-only local evaluation.

For a permitted private research upload, set **server-only** \`SUPABASE_URL\` and \`SUPABASE_SERVICE_ROLE_KEY\` through secure local environment or secret manager; never commit or paste them. Use the uploader with:

\`\`\`bash
python3 scripts/style_references/upload_private_reference_bundle.py \
  --archive "/private/path/AIWardrobe_Style_References_46_Pilot.zip" \
  --upload --confirm-private-research --create-private-bucket \
  --receipt "/private/path/stylist6_upload_receipt.json"
\`\`\`

It must create/check a **PRIVATE** bucket \`stylist6-private-research\`, upload the 46 original JPGs, then download each privately and verify hashes. The default project URL must point to \`fyqpifmrsftsfqibhwhy\` only if that is the explicitly authorized staging target. Reject public bucket, mismatched project, stale image hashes, unsupported URLs and partial uploads. If production project isolation cannot be guaranteed, create an isolated Supabase branch/project first.

**Never claim upload succeeded without the signed receipt and object verification.** A positive upload is not permission to train.

### 2. Validate fashion labels against actual images

Re-review each reference visually at full resolution. Current text captions are **approximate annotations**, not verified SKU/material/fitting facts. Produce a private audited manifest with:
- verified image hash, outfit completeness and media source/right basis
- top/second layer/trousers/shoes, neckline, silhouette, seasonal weight, distinctive textures, accent/palette
- formal register and occasion fit; realistic layer/temperature combinations
- human-reviewed \`usable_for_reference\` vs unsuitable image
- license/reuse status per media, not a blanket IG-handle assumption.

Never promote all 46 to positive training examples automatically. Hold out some references for quality evaluation to avoid leakage.

### 3. Build a small real Reference-First experiment before full rewrite

Using the existing code in \`backend/stylist6/\`:
- Implement an offline staging-only reference matcher which retrieves actual catalog items for each *garment role* and image.
- Fix the top-versus-pullover **wearing-role projection** (a fine-knit mockneck that acts as a base top is not the same as a thick outerwear pullover). Inspect actual photo and materials; do not bulk reclassify 623 pullovers.
- Use image embeddings/visual facets only when matched to corresponding image SHA/model revision. Image-to-image similarity may retrieve analogous garments **within the same role**, but must not be assumed to predict cross-item compatibility.
- Ground all selected products in real catalog IDs/images/original store URLs and price/FX quote; reject fabricated products, stock data or unavailable links.
- Run an actual hosted multimodal **whole-outfit visual critic**, comparing private reference image with a composited grid of real SKU photos. Verify the provider supports image inputs and data handling; if not available, state blocker instead of substituting text-based mock scores. CPU/latency/cost bounded; no local GPU.
- Preserve original authentication, user anchors, budget and quota through existing trusted adapters. Offline experiment is **not** authorized to modify an account's paid quota.

### 4. Prove or disprove style quality (12-case gate)

Read \`docs/architecture/STYLIST6_DECISION_GATE.md\` and \`benchmarks/stylist6/scenarios.json\`.

Produce actual products for the **same 12 selected briefs** using:
A. V5 baseline, B. creative-first S6, C. reference-first S6.

Make anonymous reference and candidate image grids with 3 complete outfits per system, capture raw JSON and source links with version/build SHA. Ask at least 3 human reviewers to vote blindly on style/occasion fit, cohesion, individuality and desire to wear; store dissent and reject reasons, not just average model score.

Required architecture-choice criteria (pilot heuristic, not final product approval):
- at least 8/12 majority wins against V5
- zero critical clothing/occasion mismatch
- 100% real-source SKU, quote/budget, anchor/identity compliance
- three genuinely different looks where inventory permits, or honest inventory-limited result
- bounded inference cost and measured latency.

If the reference-first approach underperforms, STOP THE REBUILD; audit images, retrieval, taxonomy and the evaluator. No hundreds of garment bans.

### 5. Only after the successful gate, integrate isolated staging iOS

Do NOT alter production v207 or existing staging v4. Add a **new** secure Node preview and \`aiw-stylist6-staging\` Supabase Edge proxy (signed HMAC, user JWT, exact-once quota admission). Mobile dev builds may opt in; release always stays on production. Test a real positive account ONLY with explicit approved quota/spend and exact debit delta. Test 30+ held-out inputs and blind preferences; collect latency (cold/warm p50/p95), CPU, model provider, receipts, any errors and iOS screenshots, and report GO/NO-GO.

## Security and completion contract

- No login cookies/sessions, Instagram passwords or Supabase service-role/NVIDIA keys in prompt, screenshots, commits, logs or chat.
- No public image bucket, public Git LFS photo commits, silent model training, or quota-consuming unauthorized tests.
- No source/hash guessing. No claim that the mocked visualCritic is a production vision model.
- No app/production rollout until real image critic and positive authenticated end-to-end proof.
- Separate commits with CI; maintain branch provenance and exact execution evidence.

**First deliverable:** local receipt of 46 privately validated images and a 12-case reference-first pilot with real catalog links and visual blind-comparison artifacts; if blocked, finish all available code/tests and report the precise missing credential or local file path. Do not stop after writing a plan.
