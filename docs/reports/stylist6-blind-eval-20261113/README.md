# Stylist 6 vs Stylist 5 — blinded comparison methodology and results (2026-11-13)

## What this is

A blinded, reproducible comparison between the Stylist 5 production kernel and
the Stylist 6 fashion-arm engine over the frozen 2026-11-13 UZS catalog
snapshot (sha `b4bb7ae0…`, 384 rows), using the v2 scenario set
(`benchmarks/stylist6/scenarios.json`: 45 development + 23 held-out).

**The ratings below are AI-rater preferences, NOT human acceptance.** The rater
is `nvidia/nemotron-3.5-super-vl-120b-a12b` — the same model family the S6
visual critic uses (provider- and model-independence from the *deterministic
scorer* is guaranteed; model-family independence from the S6 critic is NOT).
Uncertainty is reported; no human votes were available.

## Pipeline

1. `AIWARDROBE_DETERMINISTIC_PLANNER=1 node scripts/stylist6/fashion_replay.mjs
   .private/catalog-uz-20261113.json <out>.json` — offline 3-arm replay
   (stylist5 control kernel / stylist6Previous retrieval / stylist6Fashion).
2. `node scripts/stylist6/blind_pair_judge.mjs <replay> <catalog> --out-dir …`
   — per held-out scenario with feasible decks from BOTH arms: product photos
   are downloaded from the audited allowlist hosts, PNG→JPEG via sips, and each
   deck is scored by the VL model under a strict JSON rubric (look_quality,
   deck_cohesion, deck_diversity, occasion_fit, footwear_register; 0-10 each).
   Arms are renamed ALPHA/BETA with fresh per-scenario randomization; the
   token→arm key is written to a separate `key.json`. Verdicts are never shown
   arm names. 2 independent samples per deck are averaged.
3. Unblinding + aggregation happen only at analysis time.

## Deterministic results (tuned engine, 72 scenarios)

| arm | looks | coherent | coherence rate | uniqueSkus | budget violations |
|---|---|---|---|---|---|
| stylist5 | 155 | 149 | 96.1% | 543 | 0 |
| stylist6Previous | 134 | 126 | 94.0% | 441 | 0 |
| stylist6Fashion | 161 | 161 | **100%** | 524 | 0 |

Three qualified directions (signature/refresh/discovery) in 50/72 scenarios;
honest `alternative`/partial decks where inventory or budget is insufficient.
Zero hard-constraint violations in every produced deck.

## Blinded AI-rater results (10 comparable held-out scenarios)

Both arms returned ≥2-look decks in 10 of 23 held-out scenarios (the rest are
honest budget/inventory abstentions on one or both sides).

| evaluation round | S6 wins | S5 wins | ties | win rate | 95% CI (ties=0.5) |
|---|---|---|---|---|---|
| look-level rubric (first look only, 1 sample) | 5 | 4 | 1 | 50% | 24–86% |
| post-tuning, same rubric | 4 | 5 | 1 | 40% | 14–76% |
| **deck-level rubric, 2 samples/deck (final)** | **5** | **5** | **0** | **50%** | **19–81%** |

Mean criterion deltas (final round): look_quality +0.05, deck_cohesion +0.25,
deck_diversity −0.35, occasion_fit +0.25, footwear_register +0.05 (all ≈0 on
the 0-10 scale).

## What the evaluation taught us (and what changed because of it)

- **Rater noise floor ±3-4 points per scenario**: S6-H02's decks are
  byte-identical across engine revisions, yet its delta moved −4 → −7 → −1
  between rounds. Scenario counts this small cannot separate 40% from 60%.
- The first rubric judged only the first look's photos (a design flaw), which
  inflated variance; the final rubric judges the full 3-look deck.
- Tuning round (brand-variety tie-break + footwear-register calibration) was
  validated on the DEVELOPMENT set only (deterministic metrics equal-or-better;
  110→110 looks, coherence +0.0006, uniqueSkus +2, layered 48→49) — the
  held-out set was touched exactly once, after tuning.

## Honest conclusion

Deterministic quality measures favor Stylist 6 (100% vs 96.1% coherence, more
looks served, zero violations). The blinded AI rater shows **parity, not
superiority: 50% (5W/5L), CI 19–81%** — below the 65% target, and with n=10
comparable scenarios + this noise floor the target cannot be *demonstrated*
in either direction. The gate is NOT met and is not claimed. Path to a
decisive measurement: (a) widen catalog/budget coverage so both arms return
decks on more held-out scenarios; (b) raise samples per deck; (c) add a
rater model from a different provider family; (d) human blind votes remain
the gold standard and are the owner's call.

## Reproduce

```
node scripts/stylist6/frozen_catalog.mjs .private/catalog-uz-20261113.json --per-slot 96
AIWARDROBE_DETERMINISTIC_PLANNER=1 node scripts/stylist6/fashion_replay.mjs .private/catalog-uz-20261113.json .private/replay.json
node scripts/stylist6/blind_pair_judge.mjs .private/replay.json .private/catalog-uz-20261113.json --out-dir .private/blind-judge
```
Raw artifacts: `.private/blind-judge-decklevel/{blind,key,verdict}.json` (mode 600).
Judge spend: 40 provider calls / 669,170 tokens for the final round.
