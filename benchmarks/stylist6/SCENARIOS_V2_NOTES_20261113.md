# Scenarios v2 (2026-11-13) — taxonomy, split, preferences, run commands

`benchmarks/stylist6/scenarios.json` is now **version 2**: 68 scenarios split
into `development` (45) and `heldOutEvaluation` (23). Every version-1 scenario
(S6-01..S6-30, S6-B01..S6-B05) is byte-identical in content to version 1 and
derives the identical replay inputs (weather + preferences rotation); new rows
are appended after the originals in each section. The replay also always runs
its four hardcoded probe rows S6-F36..S6-F39, so a full replay covers 72
scenario executions.

## 1. Frozen catalog snapshot

Built 2026-11-13 with the public anonymous reader (no service credentials,
no provider calls, no image requests):

```
node scripts/stylist6/frozen_catalog.mjs .private/catalog-uz-20261113.json --per-slot 96
```

| stat | value |
|---|---|
| snapshotSha256 | `b4bb7ae09ffef1a5c44c2df309236beb739b2c9e1e49bc4291e4d63b91a933d0` |
| rows | 384 (96 per slot) |
| per slot (pool -> chosen, with visual profile) | top 1162->96/95, outerwear 989->96/95, bottom 729->96/93, shoes 103->96/96 |
| withVisualProfiles | 379 / 384 |
| distinctBaseProducts | 384 |
| sourceRows / eligibleRows | 7020 / 2983 |
| skipped | not_uzs 4014, no_outfit_slot 23 |
| output permissions | `.private/` 0700, file 0600 |

Price floor evidence (calibrates the budget tiers): cheapest feasible four-slot
outfit = **1,424,702 UZS**; three-slot without outerwear = **1,166,702 UZS**.
Brand mix: Zara 173/384, Geox 46, JACK & JONES 46, Selected Homme 28,
Massimo Dutti 23, Canali 20, MANGO 19, GANT 11, ECCO 10, Tommy Hilfiger 5,
Armani Exchange 3.

## 2. Scenario taxonomy

* **Styles (7):** `old_money` 12, `casual` 11, `minimalist` 11, `classic` 9,
  `semi_classic` 9, `business_casual` 9, `smart_casual` 7.
  The engine vocabulary (`normalizeStyleId` in `src/domain/styleInference.ts`)
  is exactly `old_money | semi_classic | minimalist | business_casual |
  casual | classic`; `smart_casual` normalizes to `casual` (see limitations).
* **Occasions (11):** Work 14, Date night 11, Travel 11, Weekend 8, Casual 7,
  Dinner 7, Business meeting 3, Wedding 2, Conference 2, Brunch 2,
  Gallery opening 1. Occasions not in the reference lookbook index
  (Wedding, Dinner, Weekend, Brunch, Conference, Gallery opening) run without
  shared reference metadata; new rows carrying that condition are labelled
  `reference_metadata_not_guaranteed`.
* **Budget tiers (10 distinct):** 400k (2), 800k (2), 1.2M (3), 1.5M (1),
  1.8M (11), 2M (1), 2.5M (13), 3M (1), 6M (1), 8M (1) UZS; 32 scenarios are
  open-cap (`budget: null`). The 400k -> 2.5M span required by the mission is
  covered by 7 tiers (400k, 800k, 1.2M, 1.5M, 1.8M, 2M, 2.5M).
* **Personal preference payloads (12):** see section 4.
* **Sparse probes (18, assertion `honest_sparse_deck_or_budget_infeasible`):**
  all 400k/800k/1.2M rows plus all 1.8M rows. Observation on the 2026-11-13
  snapshot: <=1.2M always yields 0 looks in all arms; 1.8M yields 0-3 looks
  depending on style gates (e.g. semi_classic Work at 9 C -> 0, smart_casual
  Travel -> 3). 2M+ rows keep only the v1 contract
  `hard_total_budget_or_honest_422`, but partial decks there are legitimate
  too (S6-B04 2M -> 1 look, S6-D15 2.5M -> 2/3/2, S6-H11 and S6-D10 2.5M ->
  0/0/0).

### ID scheme

* `S6-01..30` — v1 development rows (unchanged, first in `development`).
* `S6-B01..B05` — v1 held-out budget rows (unchanged, first in
  `heldOutEvaluation`).
* `S6-D01..D15` — new development rows (seeds 2001..2015).
* `S6-H01..H18` — new held-out rows (seeds 3001..3018).
* `S6-F36..F39` — replay-internal probe rows (not in this file; hardcoded in
  `scripts/stylist6/fashion_replay.mjs` and always executed).

## 3. Development / held-out split

* `development` (45 = 30 v1 + 15 new): iteration feedback allowed; engine
  authors may read per-scenario results and adjust.
* `heldOutEvaluation` (23 = 5 v1 + 18 new): run after feature freeze for the
  gate decision; includes S6-B01..B05 unchanged. Treat per-scenario outcomes
  as single-shot; do not iterate against them.
* Index-preservation guarantee: the development block is exactly 45 rows (a
  multiple of 15 = lcm of the rotation periods), so every v1 row keeps the
  same derived weather temperature and preference rotation as under the v1
  single-array layout. Verified programmatically for all 35 rows.

## 4. How preferences are expressed

New rows may carry a top-level `preferences` object; the replay passes it
through verbatim (`scripts/stylist6/fashion_replay.mjs`:
`preferences: scenario.preferences ?? preferences[i % 3]`). Fields the engine
actually consumes:

| field | consumed by | semantics |
|---|---|---|
| `favoriteColors` | `backend/stylist6/personal_taste.mjs` | soft ranking signal (+0.04/item, bounded) |
| `favoriteBrands` | `personal_taste.mjs` | soft ranking signal (+0.06/item) |
| `avoidBrands` | `personal_taste.mjs` | hard item exclusion in the Stylist 6 fashion arm (brand match, normalized lowercase) |
| `likedPatterns` | `personal_taste.mjs` | canonicalized to `stripe/check/texture/print`, soft (+0.02) |
| `fitPreference` | `personal_taste.mjs` | only `fitted` (-> slim) and `loose` (-> wide) move the needle; other values are neutral |
| `avoidColors` | `src/domain/garmentContext.ts` via `stylingContext` | hard garment-context rejection |
| `dislikedPatterns` | `src/domain/garmentContext.ts` | hard garment-context rejection |

Personal taste is applied only AFTER all hard quality gates and never rescues
an invalid composition (bounded adjustment <= 0.085 per look). The v1 rows
have no `preferences` field and keep the replay's 3-payload rotation
(favoriteColors navy/cream, avoidColors black + relaxed, dislikedPatterns
graphic + regular). New rows without distinctive payloads also get the
rotation, deterministically by index. Payload colors/brands were chosen
against snapshot evidence so signals can actually fire (tokens present at
confidence >= 0.8: black 65, brown 46, navy 27, blue 25, grey 23, white 10,
burgundy 8, olive 5; brands as listed above).

Preference rows carry the `preferences_engine_passthrough` assertion. There is
no assertion that favorites must appear — taste is a bounded tie-breaker, and
claiming guaranteed favorites would be dishonest.

## 5. Exact commands

```bash
# 1. Freeze the catalog snapshot (public anon reads only; output 0600 in .private/)
node scripts/stylist6/frozen_catalog.mjs .private/catalog-uz-20261113.json --per-slot 96

# 2. Full 72-execution replay (68 file scenarios + S6-F36..F39), offline.
#    Output file must not exist yet (wx flag).
node scripts/stylist6/fashion_replay.mjs .private/catalog-uz-20261113.json .private/fashion-replay-20261113.json

#    Fast subset (schema/pipeline validation or debugging); ids must exist:
node scripts/stylist6/fashion_replay.mjs .private/catalog-uz-20261113.json .private/fashion-replay-validate.json --only S6-01,S6-B01,S6-D04,S6-D10,S6-H18

# 3. Developer gallery HTML from the replay report (arms side by side,
#    preferences/budget captions, explicit "No feasible look" cells):
python3 scripts/stylist6/fashion_gallery.py .private/fashion-replay-20261113.json .private/fashion-replay-20261113.html

# 4. Blinded pilot contact sheets — NOTE: consumes the OLDER pilot_12 flow,
#    not the fashion replay report. pilot_12 writes <dir>/pilot-private.json:
node scripts/stylist6/pilot_12.mjs --scenarios S6-01,S6-02,... --out /tmp/s6-pilot-run    # also reads scenarios.json v2 (additive read)
python3 scripts/stylist6/contact_sheets_12.py /tmp/s6-pilot-run                           # arg = OUT dir; writes blind-<id>.jpg + blind-key.json
#    (reference grounding panels look for lookbook files in ~/Downloads/Telegram Desktop)

# 5. Blind review package + scoring — belongs to the retail organizer flow
#    (retail_replay.mjs writes <organizer>/replay.json next to inventory.json,
#    scenarios.json, image-rights.json, references/hashes.json):
python3 scripts/stylist6/blind_review.py build <organizer-dir> <output-dir> [--owner-local] [--v5-arm baseline|v5Seed] [--v6-arm v6Retrieval|v6Hosted]
python3 scripts/stylist6/blind_review.py score <output-dir>/<key>.json <submission1.json> <submission2.json> <submission3.json>
```

`pilot_12.mjs` and `fashion_replay.mjs` both read this file additively
(`scenarios` / `development` / `heldOutEvaluation` union), so v1 manifests in
copied bundles keep working.

## 6. Engine limitations discovered (2026-11-13 snapshot)

1. **Style vocabulary is closed.** `smart_casual` (and anything else) silently
   normalizes to `casual` for coherence gates; the reference lookbook does
   contain `smart_casual`, so those rows still get reference metadata. Scenarios
   using it remain valid inputs, but arm differences for them measure the
   `casual` path.
2. **Hard budget floors are style-dependent.** Raw slot floors (1.42M four-slot,
   1.17M three-slot) underestimate the real floor: style/occasion/climate gates
   push it above 2.5M for some combinations (semi_classic Business meeting at
   6 C, classic Wedding at 19 C -> 0 looks even at 2.5M). All 0-look outcomes
   observed were pure `budget` rejections — no hard-gate or coherence failures —
   i.e. the engines abstain honestly instead of violating the budget.
3. **Partial decks are honest.** `recommendFashionDeck` returns
   `complete | partial | no_feasible_looks`; the retrieval deck reports
   `distinct_inventory_limited | no_verified_looks`. Deck sizes 0-3 were all
   observed across the calibration runs; metrics record `looks` per arm and
   every produced deck had 0 budget and 0 constraint violations.
4. **Shoes pool is thin.** Only 103 eligible shoe rows existed; the 96-row
   quota takes 93% of the pool, so shoe diversity is a snapshot limitation.
5. **avoidBrands is asymmetric between arms.** The Stylist 6 fashion arm hard-
   blocks avoided brands; the V5 control path ignores unknown preference keys
   (its `StylingPreferences` interface has no `avoidBrands`). Arm deltas on
   avoidBrands rows (S6-D10, S6-D14, S6-H13) partially measure this asymmetry.
6. **Replay scenario coverage contract.** The replay reads only
   id/style/occasion/budget plus the v2 additions (weather/preferences/brief
   overrides); `mode/region/currency/seed/assertions` are fixture
   documentation and flow to blind-review tooling, not the engines.
