# Stylist 6 — category-integrity release gate (2026-11-13)

## Why this was needed

The shop catalog is not a clean product taxonomy. Read-only checks in live
Supabase project `fyqpifmrsftsfqibhwhy` showed **7,020 active rows, only 115
active footwear rows**, with 62 of those shoes missing `last_seen_at` and
**none observed in the previous 14 days** as of 2026-11-13. This timestamp
is a catalog observation, not proof of stock or purchase eligibility.

Separate category-name probes found real mistakes, for example:

- `lacoste-ch1917_f6z`: **Oxford Shirt** classified as
  `bottoms/lower_body`. Its official product URL contains
  `button-down-shirts`.
- `md-uz-61729790`: **Tapered Jeans** classified as
  `tops/upper_body`. Its official product URL contains `tapered-fit-jeans`.
- `md-uz-62899803`: **Wool Blazer** classified as
  `bottoms/lower_body`; official product URL contains `blazer`.

Raw keyword searches are **NOT** reliable verification: `Oxford` is also a
shirt fabric and `bootcut` is a trouser cut. `Tommy Jeans` can be a brand
label on outerwear. A query with broad name matching generated ~101 suspicious
live rows, *not* 101 proven misclassifications.

## Change: fail closed, do not silently relabel

`backend/stylist6/catalog_category_integrity.mjs` is a pure, local classifier.
Only when the **unambiguous garment family in the title** and the
**merchant product URL path** both contradict the native category does the
adapter return `skipped: catalog_category_conflict`. Contradictory native
`category`/`garment_type` fields are also quarantined. Titles that mention
multiple families, generic product URLs and mismatched evidence are left
unchanged, while the audit reports the uncertain rows for manual review.
No AI-generated descriptions, prices, FX rates or availability are used.

This happens *before* live taxonomy hints or the native-slot projection, so
a mislabeled shirt cannot fill the trousers slot just because the metadata
says `lower_body`. `CATALOG_ADAPTER_VERSION` was bumped to
`catalog-adapter-v2-category-evidence` for honest benchmark provenance.
This is a data-integrity check, **not** a new styling veto or shoe ban.

The read-only catalog auditor is:

```bash
node scripts/stylist6/catalog_category_audit.mjs
node scripts/stylist6/catalog_category_audit.mjs --details
```

It scans committed `data/shop_catalog_*.json` **without reading credentials,
calling merchants, changing files or updating Supabase**. `last_seen_at` is
missing from the historical JSON snapshots and must not be confused with
the live SQL catalog's timestamps.

## Evidence

| Check | Before | After |
| --- | ---: | ---: |
| Historical JSON files scanned | 180 | 180 |
| Snapshot rows (includes variants and duplicates) | 40,989 | 40,989 |
| Active snapshot rows | 38,658 | 38,658 |
| Unambiguous category-conflicting *entries* | Not quarantined | 199 |
| Unique SKU IDs represented by those entries | Not quarantined | 172 |
| Ambiguous entries for manual review | Not tracked | 24 |
| Admitted candidates in offline 8-scenario comparison | 38,244 | 38,047 |
| Price implausibility skips | 144 | 142 |
| Produced retrieval looks (8 scenarios, two arms) | 42 | 42 |
| Scenarios losing at least one look | — | 0 |

The two-arm **offline** comparison used eight fixed scenarios
(`S6-01,S6-02,S6-04,S6-05,S6-07,S6-08,S6-B01,S6-B03`) with identical
committed source files, disabled V5 and **no external calls**. One low-budget
scenario `S6-B01` already yielded zero looks in both builds. Both runs
generated the same look counts for every scenario. They are retrieval decks,
not NVIDIA-reviewed outfits or new stock assertions.

Commands used:

```bash
node --test backend/stylist6/catalog_adapter.test.mjs \
  backend/stylist6/catalog_category_integrity.test.mjs \
  scripts/stylist6/catalog_category_audit.test.mjs
node scripts/stylist6/pilot_12.mjs --out /tmp/pilot \
  --scenarios S6-01,S6-02,S6-04,S6-05,S6-07,S6-08,S6-B01,S6-B03 \
  --skip-v5 true
```

Note: `pilot_12.mjs` updates the tracked historical report as a side effect;
its output was captured separately and the tracked historical report restored.
No previous benchmark was rewritten in this change.

## Remaining release blockers

This patch **does not** add shoes, correct live catalog rows, refresh merchant
observations, establish SKU+size availability/delivery, or prove Stylist 6
visually outperforms Stylist 5. The shopping gate still requires a
current authorized merchant feed with exact variant, size, stock, landed
currency/price, destination and separately licensed product images. The human
blind gate remains unpassed (previous NVIDIA-family judge: 5–5 on ten comparable
held-out scenarios).

Do **not** bulk-update production category fields based on this audit,
reactivate archived shoes, treat HTTP 200 as stock, or promote Stylist 6 to
production. Source correction requires owner-controlled verification and a
staging migration with a rollback plan. Production remains Stylist 5 v209.
