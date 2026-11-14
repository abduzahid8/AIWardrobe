# Offline visual fashion enrichment

Use a separate local Python environment. No inference keys or paid APIs are needed. The model's weights download once (~1.4 GB); batches use local CPU or Apple MPS. This never runs in the synchronous outfit request.

```sh
python3 -m venv .visual-venv
.visual-venv/bin/pip install -r scripts/visual/requirements.txt
# Credentials are read from a local file, never committed.
export SUPABASE_ACCESS_TOKEN_FILE=/path/to/local/token
node scripts/visual/export_catalog.mjs --output /tmp/catalog-images.json --slot shoes --limit 32
# macOS optional: OCR emits only text geometry/counts, never recognized words.
swiftc scripts/visual/image_text.swift -o /tmp/catalog-image-text
.visual-venv/bin/python scripts/visual/enrich_catalog.py /tmp/catalog-images.json /tmp/catalog-visual.jsonl \
  --revision 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 --limit 32 --ocr-command /tmp/catalog-image-text
node scripts/visual/import_catalog.mjs /tmp/catalog-visual.jsonl
```

Export returns an `after` cursor and skips current source/image evidence. Each batch is explicitly limited to 1–256 products. Prioritize shoes and sparse source descriptions; expand deliberately after checking sampled images. Failed downloads remain unknown. Retry the same export to revisit failed images. A source edit invalidates its evidence by hash; changed images need a new export. Source image URLs that change content without changing URL require an explicit re-export of that product.

SigLIP similarity scores are relative to the label set, not calibrated probabilities. Only strong score/margin observations are exposed. Fiber composition, exact proportions/measurements and weight are not inferred by this classifier. Source category remains authoritative; a base/layer role ambiguity is allowed, while footwear/bottom-versus-upper conflicts are quarantined. Diagram and promotional-image exclusion needs corroborating OCR geometry or human evidence. With no OCR binary the classifications remain advisory. Schema supports model/evidence versioning; a new taxonomy or confidence policy should increment its version with a compatible migration.

Profiles, embedding caches and optional downloaded catalog images remain local until the explicit import. The import transaction rechecks the current catalog hash and image URL, writes derived metadata/embeddings only, and preserves product identities, source links, image URLs, currency and authoritative prices. Never commit tokens or model caches. Production reads retrieve compact observations; embeddings remain service-only.

STYLIST 5 adds an append-only resume journal, per-source failure journal, bounded
transport retries and a 20-second streaming deadline. Failures remain unknown;
no host blacklist is learned. The model embedding cache is keyed by image SHA
and pinned model revision. Facets are re-derived for taxonomy changes and source
metadata revisions. Resume checks expire after 24 hours (`--recheck-hours 0`
forces re-fetch); export also revisits unchanged URLs after seven days, detecting
content changes without a URL change. Retain journals privately; console output
contains counters, never product IDs or URLs. Accent hues, hem shape and shoe
purpose are confidence-gated observations; exact cloth weight and fiber content
remain unknown unless verified independently.

The `scripts/stylist5` workflow freezes the catalog, prepares a priority queue,
validates cache imports in an isolated database, and runs the one fixed quality
benchmark. The capsule marks machine evidence explicitly, never human fashion
approval. Production cache/schema changes are part of the guarded release, not
individual batch deployments. Do not run the old import script independently
while preparing that release.

The v3 facet pass also measures a primary chromatic hue from pixels when a
uniform background and a dominant foreground hue make that defensible. Mixed
hues, ambiguous masks, neutral garments and non-uniform backgrounds abstain.
`refine_pixels.py` can augment an existing checksum-verified journal without
repeating model inference. Transient re-fetch failures retain the previous
verified observations; they do not mark a host or product permanently bad.
Human photo-quality annotations use the same checksum/source-revision guard.
Measurements and on-body fit remain unknown.
