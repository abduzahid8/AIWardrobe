# Stylist 6 — genuine active-catalog footwear A/B, 2026-11-13

Project: Supabase fyqpifmrsftsfqibhwhy public RLS, Stylist 6 Draft PR #15.
Mode: anonymous SELECT-only, no database writes, no product photo HTTP
requests, no NVIDIA or provider calls, no quota debit, no deployment.

## Source integrity

The frozen_catalog.mjs script pulled the public shop_catalog,
catalog_garment_intelligence and catalog_visual_fashion tables through
anon/RLS GET requests using existing local owner-managed .env. It
did not read any private wardrobe, auth, payment or user tables.

- Live active shop_catalog: **7,020** SKU rows.
- Eligible native UZS/outfit-category shop records: **2,983**.
- Other currencies: **4,014**, left out without currency conversion.
- Missing outfit slot: **23**.
- Four-slot frozen sample: **384** (96 each top, outerwear, bottom, shoes).
- Exact-current cached visual profiles: **379 of 384**.
- Frozen file checksum (SHA-256):
  b4bb7ae09ffef1a5c44c2df309236beb739b2c9e1e49bc4291e4d63b91a933d0.
- Fresh read returned the same sample checksum as earlier 2026-11-13
  audit; this is a snapshot of actual catalog rows, NOT proof of
  refreshed seller prices or actual sizes.

The curated four **inactive preview** Tommy Hilfiger rows were appended
ONLY to temporary JSON comparison files, where is_active:true is a
**hypothetical testing assumption**. No SQL upsert, actual catalog
activation or merchant-offer authorization.

## Eight-scenario retrieval results

Scenario IDs: S6-01, S6-02, S6-04, S6-05, S6-07, S6-08,
S6-B01 and S6-B03 (Old Money / Classic / Semi-Classic, work, casual,
travel, dinner and dates). Reference-first and creative-first
retrieval ran, visual verdicts NOT_RUN. S6-B01 with a 1.5m UZS
full-look budget produced zero outfits in all runs.

| Test pool | Admitted before/after | Looks before/after | Unique shoes before/after | Changed looks | New Tommy shoe picks |
| --- | --- | --- | --- | ---: | ---: |
| Native-UZS sample, all existing brands | 382 / 386 | 42 / 42 | 17 / 17 | **1** | **1** |
| Same sample, exclude ECCO by evaluation flag | 372 / 376 | 42 / 42 | 18 / 17 | **7** | **1** |

The --exclude-brands ECCO flag is an **offline research choice**, NOT
a change to production catalog filtering. The owner declined ECCO as a
new-brand import; whether to permanently hide existing ECCO items
requires a separate product decision.

**Conclusion:** four Tommy listings change one shoe in selected looks
but do not improve unique footwear diversity; excluding ECCO
makes more looks change without increasing distinct shoes. Need more
relevant locally available loafers, Derby, Oxford and Chelsea, with a
lawful source for photos and size-specific live stock data.
Do not artificially boost new items to inflate a metric.

The one newly selected product is a brown Tommy Hilfiger
WARM HYBRID CHUKKA, in S6-01 Old Money/Work reference-first look;
it replaced a black waterproof Geox ankle boot. That one selection is
not evidence of a more elegant Old Money outfit, nor does it validate
any additional local Derby/Oxford stock. The alternate study without
ECCO is a controlled offline filter, not a change to the user's real
catalog.

## Repeatable, non-mutating commands

With Node 22, npm ci, and existing local .env containing the Supabase URL
and **public anon** key:

    mkdir -p .private/stylist6-live
    AIW_S6_ENV_FILE="$PWD/.env" node scripts/stylist6/frozen_catalog.mjs \
      .private/stylist6-live/catalog.json --per-slot 96
    node scripts/stylist6/live_footwear_replay.mjs \
      --catalog .private/stylist6-live/catalog.json \
      --out .private/stylist6-live/comparison
    node scripts/stylist6/live_footwear_replay.mjs \
      --catalog .private/stylist6-live/catalog.json \
      --out .private/stylist6-live/comparison-no-ecco \
      --exclude-brands ECCO

The orchestrator validates four inactive candidate rows and uniqueness,
uses private nontracked outputs, passes --report-out to the pilot to
avoid overwriting the tracked benchmark, compares complete outfit
identities, and prints summary counts. Detailed URLs and deck contents
remain in .private, mode 600. Never directly upsert a preview.

## Actual remaining gates

1. Verify each exact SKU/color/size, seller price, country and delivered
   total; size snapshots are time-limited, not reservations.
2. Image rights: display and provider upload are separate grants.
3. Hosted staging pinned to exact tested compute SHA, independent
   human fashion judgments on three distinct full looks.
4. Quota, latency, billing, rollback review and owner-approved release.
   Stylist 5 v209 remains untouched.
