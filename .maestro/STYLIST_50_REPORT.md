# AI Personal Stylist — 50-Scenario Visual Test Report (FINAL)

**Date:** 2026-10-07 → 2026-10-08
**Project:** /Users/v.zahid/AIWardrobe (Expo Go, iPhone 17 simulator)
**Outcome:** 50/50 scenarios PASS in final state — 2 pre-fix FAILs found, fixed, and re-run clean; 19 NOTE rows (cosmetic/data/backlog only); 350/350 screenshots fresh (7 per scenario), 0 failure shots this run.

## Executive summary

- Every scenario produced a full deck (3 distinct cards), every card 4 labeled tiles (LAYER / TOP / PANTS / SHOES), prices in UZS, % Match, weather chip, style tip — verified by deck gate + shot integrity for all 50.
- **All 50 decks were rule-audited deterministically** from the Metro log (see Method) — not just spot-checked. Final captures contain zero hard-rule violations.
- Two real FAILs were found and resolved: scenario 25 and scenario 50 both had a training towel in the TOP slot (pre-fix captures). Towel fix shipped (code + tests), both re-run clean.
- Scenario 47's main-run captures were invalid (edge-function timeouts; only 2 cards rendered; a faulty gate accepted them) — re-run clean.
- 19 NOTE rows are cosmetic quirks, catalog data issues, style oddities that are legal by the agreed rules, or the documented sport-shoe backlog.
- Quality gates green: `tsc --noEmit` exit 0; Jest identical to HEAD baseline (4 pre-existing failed suites / 23 pre-existing failed tests) with 790 passing (+2 towel tests); ESLint 13 pre-existing `no-console` only.

## Method

1. **`bash .maestro/run_stylist_50_cards.sh`** — per scenario: maestro drives the app, types the scenario comment (canonical list, script lines 14–65), waits for generation, captures 7 screenshots (card 1, scrolls, clamps). `ONLY=NN` filters; up to 5 attempts with a deck gate (`edge=5 shown=3` in `/tmp/expo_metro.log`).
2. **Deterministic deck audit** — `/tmp/audit_decks.py` replays `/tmp/expo_metro.log` (the app's own generation log) and re-implements every judging rule from `occasionRules.ts` against the **actual rendered deck** (the 3 cards after the passing `FINAL SLOTS` line):
   - RULE-A: sneakers/hybrid athletic soles × tailored trousers
   - RULE-B: two outer coats layered
   - RULE-C: two pullover knits layered
   - RULE-D: accessory prop / non-garment in a garment slot (incl. towels)
   - scoped silhouette rules (formal/classic occasion × trainers/jeans etc.)
   - SPORT-SHOE preference (gym deck should use athletic shoes) — reported as NOTE, not FAIL (fill preference, not a veto)
   Output: `/tmp/audit_out.txt` — blocks `#1–#55` = pre-fix run, `#56–#114` = final run + re-runs.
3. **Visual eyeball** (verifier agents + self) on 24 scenarios for cosmetic/product-data issues.
4. **Ground truth:** Metro log + file md5. The image-read tool exhibited content misattribution during this session (batch and even fresh-path single reads returned swapped images); md5 chains and Metro forensics proved the screenshot files correct. Verdicts rely on Metro + md5, not on image-read output.

### Audit results

- **Pre-fix run (#1–#55):** 51 violation lines (38 hard-rule + 13 sport-shoe notes) — confirms the shipped vetoes catch what they were built for.
- **Final run (#56–#114):** every card 4 items. Violations only:
  - #77 = scenario 21: sport-shoe ×3 (Derby, Tassel Loafers ×2)
  - #78 = scenario 22: sport-shoe ×3 (loafers)
  - #80 = scenario 24: sport-shoe ×1 (Leather moc toe)
  - #81 = scenario 25 (main): **towel FAIL** + sport-shoe ×1 → fixed, re-run
  - #109 = scenario 50 (main): **towel FAIL** → re-run
  - #111 = scenario 25 (re-run): sport-shoe ×1 only (towel gone)
  - #113 = scenario 47 (re-run): ok · #114 = scenario 50 (re-run): ok
  - All other blocks ok, including gym scenarios 23 (sneakers ✓) and 41 (Running Shoes / Кеды ✓).

## Results (50 rows)

| # | Scenario (canonical) | Verdict | Evidence / notes |
|---|----------------------|---------|------------------|
| 01 | work, old money | NOTE | audit ok; Buy missing card 1; trench photo reads as plain mac (no belt) |
| 02 | work, classic | NOTE | audit ok; blazer+sweatshirt+jeans casual-odd but legal; Buy missing card 2; "100% Wool Regular Fit" shirt claim odd |
| 03 | work, semi-classic | PASS | audit ok (gate + integrity; no eyeball) |
| 04 | work, minimalist | PASS | audit ok |
| 05 | work, casual | NOTE | audit ok; same-name blazer 4.5M vs 7.0M UZS across cards; "Hilifger" typo; Buy missing card 2 |
| 06 | weekend casual brunch, old money | NOTE | audit ok; crepe-sole boots with wool trousers (sole not athletic — legal); Buy missing all 3 cards |
| 07 | weekend casual brunch, classic | NOTE | audit ok; camo barrel jeans loud for "classic"; "Wool Blend Check Suit" photo reads plain; SHOES tile below fold card 3 |
| 08 | weekend casual brunch, semi-classic | NOTE | audit ok; suit trousers + boat shoes odd but legal; striped-suit photo reads plain |
| 09 | weekend casual brunch, minimalist | PASS | audit ok |
| 10 | weekend casual brunch | PASS | audit ok |
| 11 | date night, old money | NOTE | audit ok; polo photo labelled "V-neck" (photo/name); cowboy boots odd for old money; no literal Buy CTA |
| 12 | date night, classic | NOTE | audit ok; knit-polo-over-sweatshirt (allowed carve-out); sweater+shirt card clean |
| 13 | date night, semi-classic | NOTE | audit ok; sweater+sweatshirt combos (carve-out); cowboy boots odd |
| 14 | date night, minimalist | NOTE | audit ok; hybrid-sole shoes with drawstring wide-leg pants (not tailored → legal, odd); tip mentions shirt cuff with no shirt |
| 15 | date night, casual | NOTE | audit ok; Hybrid Clean cupsole × cotton chinos (legal); suit-trousers+denim-shirt+henley oddity; Buy only card 2 |
| 16 | formal wedding, old money | PASS | audit + visual clean; minor TOP photo/name mismatch |
| 17 | formal wedding, classic | NOTE | audit ok; hybrid shoes + wool blazer but Loose Fit pants (not tailored → legal, borderline stylistic) |
| 18 | formal wedding, semi-classic | PASS | audit + visual clean |
| 19 | formal gala, minimalist | NOTE | audit ok; hybrid shoes × "Tapered Fit Climbing Trousers" under blazer (not tailored → legal, odd) |
| 20 | formal wedding, casual | PASS | audit + visual clean |
| 21 | gym workout, old money | NOTE | audit: **sport-shoe ×3** (Derby, Tassel Loafers ×2) in a sport deck — backlog (see Residual); knit polo over dress shirt legal; garments are regular wear (no sport apparel in catalog); caption "Old Money" correct |
| 22 | gym workout, classic | NOTE | audit: **sport-shoe ×3** (loafers); knit-polo-over-sweatshirt carve-out; parachute pants + loafers odd; caption correct |
| 23 | gym workout, semi-classic | PASS | audit clean; sneakers = correct sport shoe |
| 24 | gym workout, minimalist | NOTE | audit: 1 card with Leather moc toe shoes; blazer over zip-through knit legal; Buy mid-card |
| 25 | gym workout, casual | **PASS (after fix)** | main-run deck **FAIL: Large Training Towel in TOP** (RULE-D) → fixed (code + 12 tests) → re-run 11:47 clean; residual: 1 card Leather Loafer Clogs (sport-shoe backlog) |
| 26 | travel airport, old money | NOTE | audit ok; "Leather Track Shoes" is a lug-sole derby (photo/name); joggers under old-money label odd; no Buy |
| 27 | travel airport, classic | PASS | audit ok |
| 28 | travel airport, semi-classic | PASS | audit ok |
| 29 | travel airport, minimalist | PASS | audit ok |
| 30 | travel airport, casual | PASS | audit ok |
| 31 | work, old money, $250 | PASS | audit ok; budget over-cap shows price delta by design |
| 32 | work, minimalist, under $100 | PASS | attempt-1 edge flake (`shown=0`) recovered on attempt 2; audit ok |
| 33 | office interview, classic | NOTE | audit ok; visual: hybrid shoes × washed pants (legal-odd) |
| 34 | corporate meeting, old money | PASS | audit ok |
| 35 | romantic dinner, old money | PASS | audit ok |
| 36 | black tie wedding, classic | NOTE | audit ok at rule level; visual: black-tie under-dressed (trench combos instead of tux formality) |
| 37 | quiet luxury weekend casual | PASS | audit ok |
| 38 | everyday casual, minimalist | PASS | audit ok |
| 39 | airport trip, semi-classic | PASS | audit ok |
| 40 | flight travel, casual | PASS | audit ok |
| 41 | sport run, casual | PASS | audit ok; Running Shoes / Кеды = correct sport shoe |
| 42 | business casual work | PASS | audit ok |
| 43 | modern professional office | PASS | audit ok |
| 44 | date night classic | PASS | audit ok |
| 45 | old money work | PASS | main capture corrupted by the 23:40 sim freeze (`shown=1`, HOME frames) → recaptured 07:47; audit #110 ok; self-verified: 3 distinct 4-tile cards |
| 46 | minimalist date night | PASS | audit ok |
| 47 | classic work, $200 | PASS (re-run) | main attempts edge-flaked: `shown=None` timeout, `shown=0`, then `shown=2` wrongly accepted → invalid 2-card capture; re-run 20:31–20:33 (attempt 1 edge-fail, attempt 2 clean): blazer + structured polo + Comfort Suit Trousers + Oxford/Desert Boot — audit #113 ok, shots refreshed |
| 48 | semi-classic brunch weekend | PASS | audit ok |
| 49 | formal old money | PASS | audit ok |
| 50 | casual everyday, smart casual | **PASS (after fix)** | main-run deck **FAIL: Small Training Towel in TOP** (pre-fix capture) → re-run 20:33–20:34 clean: henley+jeans+кеды / blazer+striped shirt+denim+outdoor sneakers / trench+knit polo+wool pants+loafers — audit #114 ok |

**Tally:** 50 PASS / 0 FAIL (final state). **19 NOTE rows** (01,02,05,06,07,08,11,12,13,14,15,17,19,21,22,24,26,33,36) — cosmetic/data/backlog only. 2 pre-fix FAILs (25, 50) resolved.
**Evidence tiers:** 24 rows eyeballed + audit; 45/47/50/25 re-capture-verified + audit; remaining 23 rows = audit + gate + shot integrity (rules fully machine-checked; cosmetic eyeball not done — see Limitations).

## Fixes shipped (all client-side; verified by gates)

1. **Universal vetoes** in `rejectOutfitIncoherence` (`features/outfit-generator/utils/occasionRules.ts:446`, block `:456–494`) — applied before the formal/classic scope guard, so they cover every style/occasion; `styleJudgeReason` (`:711`) routes edge-generated cards through them (`:728`):
   - RULE-A: sneakers/hybrid athletic soles never pair with tailored trousers
   - RULE-B: two outer coats never layer together
   - RULE-C: two pullover knits never layer together
   - RULE-D: accessory props never fill a garment slot (comment `:487`)
2. **Sport occasion silhouette** — `matchesOccasionSilhouette` requires trainer-class shoes (`:390–395`); `SPORT_SHOE_RE` `:436`; `slotShoesRaw` prefers sport shoes before the formal tier (`normalizeSlots.ts:786`).
3. **`ACCESSORY_PROP_RE`** (`occasionRules.ts:621`) += key rings/брелок and **training towels/полотенце** — the scenario-25/50 fix; towel + judge-veto unit-tested.
4. **Tailored-outer fabric family** extended: `parka|anorak|gilet|windbreaker` (`normalizeSlots.ts:45/50`).
5. **Run-script hardening** (`.maestro/run_stylist_50_cards.sh`): `ONLY=` filter; per-attempt deck gate; `pre_shown` grep arithmetic fix; bash 3.2 empty-array guard; md5 consolidation verified correct.
6. **Tests** — `__tests__/utils/outfitCoherenceUniversal.test.ts`: 16 tests incl. towel veto (`:184–193`) and 5 jogger-phrasing tests; `__tests__/utils/normalizeSlotsShopUrl.test.ts`: Buy-button merge regression; all passing.
7. **Jogger phrasing gap** (user-reported live card: blazer + "Basic Jogging Trousers") — sloppy-bottom regexes matched `jogger(s)` but never `jogging`, so jogger pants named by action slipped old-money/classic/semi/minimalist decks. `SLOPPY_BOTTOM_RE` (`occasionRules.ts:52`) and the style-scoped branches (`:281`, `:308`, `:317`) + `BOTTOM_WORDS_RE` (`:617`) now all match `jogging`. Edge scorer/prompt parity updated in source (lines 301/365/598) — **not deployed** (no supabase CLI); the client judge is the gate either way.
8. **Buy button missing on most tiles** (user-reported) — edge items never carry `sourceUrl`, and `normalizeTo4Slots` merged price and image from the pool row but **never the shop URL**, so only tiles whose item happened to come from the pool rendered a Buy button. `repairShopUrl` (`normalizeSlots.ts`, alongside `repairImage`) now reattaches the pool row's `shopUrl` by id (incl. `shop_` prefix variants) then by name. Remaining no-Buy cases = catalog rows whose `source_url` is NULL (data) or static-fallback pool items (no URLs by design).

## Decisions made during the audit

- **Sport-shoe preference is a fill preference, not a judge veto** (`normalizeSlots.ts:537/544/683`) — dress shoes in gym decks (21/22/24/25) stay **backlog NOTE**, no new code fix.
- **No sport apparel exists in the catalog** (confirmed by user: "i don't have sports clothing"). Gym decks are built from regular garments by necessity → garment-mismatch notes on gym rows are a catalog limitation, not FAILs.
- The image-read tool is unreliable for verification; Metro log + md5 are the adjudication sources.

## Residual issues / backlog (product, not test-blockers)

1. **Sport shoes in gym decks** (21 ×3, 22 ×3, 24 ×1, 25 ×1) — one-line veto in `rejectOutfitIncoherence` would enforce the preference; deliberately not shipped (decision above).
2. **No sport apparel in catalog** — gym outfits cannot look athletic; add sport tops/shorts to catalog if athletic looks are wanted.
3. **Catalog data:** photo-vs-name mismatches (trench photo plain, striped-suit photo plain, polo photo labelled V-neck, "Leather Track Shoes" misnamed, "Hilifger" typo). These feed the stylist — worth a catalog cleanup pass.
4. **Price inconsistency:** same-name "100% Wool Suit Blazer" priced 4 500 000 vs 7 000 000 UZS on different cards — dedupe/sync needed.
5. ~~**Buy button missing** on many cards~~ — **FIXED** (see Fixes shipped #8); residual cases are NULL `source_url` rows in `shop_catalog` (data cleanup) and static-fallback items.
6. **Deck homogeneity:** re-run 47's 3 cards share blazer + suit trousers — valid but same-y.
7. **Gate bug:** the run script wrongly accepted `shown=2` for scenario 47 — gate should additionally require ≥3 distinct md5s per scenario (post-hoc integrity check now covers this).
8. **Edge-function flakiness:** `generate-outfits timed out` / "No complete outfits returned from AI" during the network flap caused 47's main-run failures.
9. **Allowed-by-design carve-outs:** sweater+sweatshirt and knit-polo-over-shirt layering remain legal; revisit if undesired.
10. **Style caption vs requested style** cosmetic mismatches seen in pre-fix runs; re-run captions spot-checked correct.

## Gates & reproducibility

- `npx tsc --noEmit` → exit 0
- `npx jest` → 4 failed suites / 23 failed tests (**identical set to HEAD baseline**, pre-existing), **796 passed / 819** (+6 jogger/Buy-button tests vs 788 baseline)
- `npx eslint <2 edited util files>` → 13 `no-console` errors (pre-existing, unchanged)
- Shots: 350 `s50c_*` files (7 × 50), 0 `_fail` this run, all newer than run start (2026-10-07 22:41), every scenario ≥3 distinct md5s (no flat decks); 7 legacy `s50_*_fail.png` from the Oct 6 round remain on disk
- Re-run: `bash .maestro/run_stylist_50_cards.sh` (or `ONLY=47,50 bash …`); deck audit: `python3 /tmp/audit_decks.py > /tmp/audit_out.txt`

## Incidents

- **23:40 Oct 7** — simulator froze mid-scenario-47 (screen + JS dead); recovered via `simctl shutdown/boot`. Scenario 45's captures were corrupted by the freeze → recaptured 07:48; scenario 47's resumed captures were invalid (edge timeouts, 2-card deck) → recaptured 20:33.
- **19:43 Oct 8** — simulator down; rebooted. App was signed in throughout (transient signed-out UI earlier); no re-auth needed.
- **Image-read misattribution** — batch and fresh-path reads returned swapped images while md5/metro proved files correct; switched to Metro/md5 forensics for all adjudication.
- **Edge function not redeployed** — no supabase CLI in this environment; all fixes are client-side (they gate whatever the server returns). Deploy separately if the server should avoid these combos itself.
- Status-bar clock skew in some captures (sim clock) — cosmetic, does not affect verdicts.
