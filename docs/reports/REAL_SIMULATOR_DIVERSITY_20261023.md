# Real Simulator Audit — Personal Stylist Deck Diversity

**Date:** 2026-10-23
**Device:** iPhone Simulator `AIWardrobe-Login-Audit-20261020` (`A803D6CC-0840-40F3-95EF-70546B2E0ED0`), iOS 26.5
**App:** `com.aiwardrobe` 1.4.3 (56), **Debug** build rebuilt from the working tree
**Runner:** Maestro 2.8.0, `.maestro/run_personal_stylist_20_sim.sh`
**Flow:** `.maestro/sim_personal_stylist_20_native.yaml`
**Mode:** Personal stylist → New clothing (live shop catalog) → Generate outfit

> The simulator was previously running a **release** build with an embedded
> `main.jsbundle`, so it could not exercise the uncommitted diversity changes.
> The app was rebuilt with `npx expo run:ios --configuration Debug` and
> installed on the same simulator before the audit. Metro is served on
> `localhost:8082`; port 8081 belongs to a different project on this machine.

## Gates

| Gate | Result |
|---|---|
| TypeScript (`npx tsc --noEmit`) | pass |
| Jest — diversity/deck/slot/stylist suites | **101/101 tests pass** (8 suites) |
| Deterministic 20-scenario stylist simulation | **20/20 strict**, worst pairwise similarity **0.55** (threshold 0.65) |
| Real simulator 20-scenario suite (first pass) | **16/20** |
| Real simulator 20-scenario suite (after retry) | **17/20** |

## Real simulator results

| # | Scenario | Result | Notes |
|---|---|---|---|
| 01 | work, old money | PASS | Deck telemetry: RELAXED; valid pool had only one outerwear family (catalog limitation) |
| 02 | work, classic, under $250 | PASS | |
| 03 | job interview, business casual | PASS | |
| 04 | corporate conference, semi-classic | PASS | |
| 05 | date night, old money | PASS | Deck telemetry: RELAXED; three knitwear-based cards despite coat/cardigan candidates |
| 06 | romantic dinner, minimalist | PASS | |
| 07 | date night, casual | PASS | |
| 08 | weekend brunch, classic | PASS | |
| 09 | everyday casual, minimalist | PASS | |
| 10 | formal wedding, old money | PASS | Deck telemetry: STRICT; three distinct archetypes |
| 11 | black tie wedding, classic | **FAIL** | 3/3 attempts: edge `503 ai_generation_unavailable` |
| 12 | formal gala, semi-classic | PASS | |
| 13 | gym workout, casual | **FAIL** | 3/3 attempts: edge `422 no_matching_clothing` (no sport catalog) |
| 14 | sport run, minimalist | **FAIL** | 3/3 attempts: edge `422 no_matching_clothing` (no sport catalog) |
| 15 | travel airport, classic | PASS | |
| 16 | flight travel, semi-classic | PASS | Recovered on attempt 2 |
| 17 | quiet luxury weekend, old money | PASS | |
| 18 | modern professional office | PASS | Deck telemetry: RELAXED; three blazer-based cards despite `supportsTargetConcepts=true` |
| 19 | офис, олд мани | PASS (retry) | Transient edge failure; passed on retry attempt 2 |
| 20 | sayohat uchun, oddiy | PASS | |

The three deterministic failures (`11`, `13`, `14`) match the previously
documented catalog/provider gaps and are not caused by the diversity change.
Scenario `19` failed all three attempts in the main run with the generic AI
banner and passed on retry, confirming a transient provider failure.

## Deck telemetry (real edge candidates)

Four representative scenarios were re-run with verbose deck logging:

| Scenario | Mode | Archetypes selected | Inventory |
|---|---|---|---|
| 01 — work, old money | RELAXED | `blazer-polo-tailored-boot`, `blazer-polo-tailored-boot`, `blazer-shirt-tailored-loafer` | `conceptCapacity=27`, `outerwear families=1` |
| 05 — date night, old money | RELAXED | `knitwear-shirt-tailored-loafer`, `knitwear-shirt-chino-loafer`, `knitwear-shirt-tailored-loafer` | `conceptCapacity=144`, `supportsTargetConcepts=true` |
| 10 — formal wedding, old money | STRICT | `knitwear-shirt-tailored-loafer`, `blazer-shirt-chino-boot`, `blazer-polo-tailored-dress-shoe` | similarities 0.30 / 0.22 / 0.32 |
| 18 — modern professional office | RELAXED | `blazer-polo-chino-loafer`, `blazer-polo-tailored-loafer`, `blazer-shirt-tailored-loafer` | `conceptCapacity=81`, `supportsTargetConcepts=true` |

### Finding

The gate works as intended when the valid inventory truly lacks alternative
formulas (`01`: one outerwear family). However, `05` and `18` relaxed with
`diversity_unavailable_despite_varied_inventory` even though the pool
supported the target number of concepts and the replacement builder produced
different alternatives (`coat-*`, `cardigan-*`). Both decks therefore still
showed repeated formulas:

- `05`: card 1 and card 2 are both a white shirt under a knit layer with
  tailored/chino trousers and loafers (visual near-duplicates).
- `18`: all three cards are blazer + top + tailored/chino + loafer.

Root cause is in the absolute floor of `assembleDiverseDeck`: when no strict
combination is found and `supportsTargetConcepts` is true, it sets the
`diversity_unavailable_despite_varied_inventory` reason and then falls through
to the `product-only` rung, which still permits shared archetypes. The
combination score then prefers the edge candidates because they have lower
colour/material similarity, so the genuinely different fill candidates are
not selected.

## Evidence

All screenshots from this audit are committed under `.maestro/verify-shots/`
(force-added despite the blanket `.maestro/verify-shots/` ignore rule):

- Full suite screenshots (16 passing scenarios + 4 failure captures):
  `.maestro/verify-shots/sim20_diversity_20261022/`
- Retry evidence for scenario 19:
  `.maestro/verify-shots/sim20_diversity_20261022_retry/19/`
- Telemetry-run screenshots (scenarios 01, 05, 10, 18):
  `.maestro/verify-shots/sim20_telemetry/`
- Maestro run log: `.maestro/verify-shots/sim20_diversity_20261022/run.log`
- Verbose deck telemetry: `.expo/dev/logs/start.log` (search `stylist deck`)
- Deterministic report output: `/tmp/stylist-diversity-report.md`

Key cards:

| Evidence | Path |
|---|---|
| 05 — near-duplicate knitwear cards 1 and 2 | `.maestro/verify-shots/sim20_telemetry/05/sim20_05.png`, `sim20_05_c2.png` |
| 18 — three blazer-based cards | `.maestro/verify-shots/sim20_telemetry/18/sim20_18.png`, `sim20_18_c2.png`, `sim20_18_c3.png` |
| 10 — strict, three distinct archetypes | `.maestro/verify-shots/sim20_telemetry/10/sim20_10.png`, `sim20_10_c2.png`, `sim20_10_c3.png` |
| 11 — black-tie 503 failure | `.maestro/verify-shots/sim20_diversity_20261022/11/sim20_11_fail.png` |
| 13 / 14 — sport 422 failures | `.maestro/verify-shots/sim20_diversity_20261022/13/sim20_13_fail.png`, `.../14/sim20_14_fail.png` |
| 19 — transient failure, then retry pass | `.maestro/verify-shots/sim20_diversity_20261022/19/sim20_19_fail.png`, `.maestro/verify-shots/sim20_diversity_20261022_retry/19/sim20_19.png` |

## Reproduction

```bash
# rebuild the debug app from the working tree
npx expo run:ios --device A803D6CC-0840-40F3-95EF-70546B2E0ED0 \
  --configuration Debug --no-bundler
xcrun simctl install booted \
  ~/Library/Developer/Xcode/DerivedData/AIWardrobe-*/Build/Products/Debug-iphonesimulator/AIWardrobe.app
xcrun simctl spawn booted defaults write com.aiwardrobe RCT_jsLocation localhost:8082
xcrun simctl launch booted com.aiwardrobe

# full real-simulator suite
FLOW=.maestro/sim_personal_stylist_20_native.yaml \
OUT=.maestro/verify-shots/sim20_diversity_20261022 \
./.maestro/run_personal_stylist_20_sim.sh

# retry a subset
FLOW=.maestro/sim_personal_stylist_20_native.yaml \
OUT=.maestro/verify-shots/sim20_diversity_20261022_retry \
ONLY=19 ./.maestro/run_personal_stylist_20_sim.sh
```

## Follow-up

Fix the absolute-floor selection so a pool with
`supportsTargetConcepts=true` selects the least-similar combination (including
fill candidates) instead of relaxing to a product-only deck. Add a regression
test that feeds five same-archetype edge candidates plus diverse fill
candidates and asserts that the final deck has three distinct archetypes.
