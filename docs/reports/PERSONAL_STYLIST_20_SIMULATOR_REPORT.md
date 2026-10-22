# Personal Stylist — 20-Scenario Simulator Report

**Date:** 2026-10-22
**Device:** iPhone Simulator `AIWardrobe-Login-Audit-20261020` (`A803D6CC-0840-40F3-95EF-70546B2E0ED0`), iOS 26.5
**App:** `com.aiwardrobe` 1.4.3 (56.0), signed-in account `dasdsa@gmail.com`
**Flow:** `.maestro/sim_personal_stylist_20_native.yaml`
**Runner:** `.maestro/run_personal_stylist_20_sim.sh`
**Mode:** Personal stylist → New clothing (live shop catalog) → generate outfits

## Executive summary

| Metric | Run 2 result |
|---|---|
| Scenarios executed on simulator | **20** |
| Passed | **17** |
| Failed | **3** |
| Simulator pass rate | **85%** |
| Deterministic product failures | 3 (`11`, `13`, `14`) |
| Retry-recovered scenarios | 4 (`05`, `16`, `19`, `20`) |
| Result screens visually inspected | 17 |
| Visually clean result screens | **13 / 17** |
| Screens with blank product image tiles | **4 / 17** (`04`, `07`, `08`, `09`) |
| Jest suite (same date) | 20/20 passed |

Run 1 and Run 2 both finished at **17/20** with the same three deterministic failures. Run 2 adds a fresh screenshot set and a full visual inspection. Every passing scenario reached **“Your Outfits”** and rendered **3 outfit cards** with `90% Match`, `TOP` / `LAYER` / `BOTTOM` / `SHOES`, UZS prices, `Buy`, `Add to Calendar`, and `Save Outfit`.

## Run history

| Run | Passed | Failed | Retry-recovered | Evidence |
|---|---|---|---|---|
| Run 1 | 17/20 | `11`, `13`, `14` | `02`, `07`, `09`, `17`, `19` | `.maestro/verify-shots/sim20native/` |
| Run 2 | 17/20 | `11`, `13`, `14` | `05`, `16`, `19`, `20` | `.maestro/verify-shots/sim20native_rerun/` |

The three failures are stable across runs. Transient edge/AI failures vary by run; they eventually recover on retry.

## Run 2 results

| # | Comment entered | Outcome | Attempts | Visual | Notes |
|---|---|---|---|---|---|
| 01 | work, old money | PASS | 1 | Clean | Old Money deck |
| 02 | work, classic, under $250 | PASS | 1 | Clean | Budget/CBU reference visible |
| 03 | job interview, business casual | PASS | 1 | Clean | Modern Professional deck |
| 04 | corporate conference, semi-classic | PASS | 1 | **Blank BOTTOM** | Semi-Classic deck |
| 05 | date night, old money | PASS | 2 | Clean | Old Money deck |
| 06 | romantic dinner, minimalist | PASS | 1 | Clean | Minimalist deck |
| 07 | date night, casual | PASS | 1 | **Blank BOTTOM** | Casual deck |
| 08 | weekend brunch, classic | PASS | 1 | **Blank TOP + BOTTOM** | Classic deck |
| 09 | everyday casual, minimalist | PASS | 1 | **Blank LAYER** | Minimalist deck |
| 10 | formal wedding, old money | PASS | 1 | Clean | Old Money deck |
| 11 | black tie wedding, classic | **FAIL** | 3 | n/a | Edge `503 ai_generation_unavailable` |
| 12 | formal gala, semi-classic | PASS | 1 | Clean | Semi-Classic deck |
| 13 | gym workout, casual | **FAIL** | 3 | n/a | Edge `422 no_matching_clothing` |
| 14 | sport run, minimalist | **FAIL** | 3 | n/a | Edge `422 no_matching_clothing` |
| 15 | travel airport, classic | PASS | 1 | Clean | Classic deck |
| 16 | flight travel, semi-classic | PASS | 2 | Clean | Semi-Classic deck |
| 17 | quiet luxury weekend, old money | PASS | 1 | Clean | Old Money deck |
| 18 | modern professional office | PASS | 1 | Clean | Modern Professional deck |
| 19 | офис, олд мани | PASS | 2 | Clean | Russian intent → Old Money |
| 20 | sayohat uchun, oddiy | PASS | 5 | Clean | Uzbek intent → Casual; recovered after 4 failures |

Scenario 20 failed 4 consecutive times with the generic AI banner, then passed. A direct call with the same session returned `200` with 5 outfits, confirming a transient provider failure rather than a deterministic catalog gap.

## Visual inspection

**Method:** four screenshots per passing scenario were combined into a 2×2 contact sheet (`<id>_contact.png`), plus an all-scenarios first-card overview and a failures overview. Each result screen was checked for:

- header/back button, `90% Match` badge
- four product tiles with images
- `TOP` / `LAYER` / `BOTTOM` / `SHOES` labels, names, UZS prices, `Buy`
- three distinct outfit cards
- `Add to Calendar` / `Save Outfit` actions
- style summary and styling tip
- clipping, overlap, blank tiles, and error visibility

### Blank image tiles

Four passing scenarios had at least one blank product image in the first card; the other cards loaded normally:

| Scenario | Slot | Item | Price |
|---|---|---|---|
| 04 | BOTTOM | 100% Linen Striped Pants | 1,900,000 UZS |
| 07 | BOTTOM | WASHED STRAIGHT-LEG CHINO TROUSERS | 779,900 UZS |
| 08 | TOP | Regular fit striped Oxford shirt | 1,400,000 UZS |
| 08 | BOTTOM | SUIT FLARE FIT TROUSERS | 1,099,900 UZS |
| 09 | LAYER | RELAXED FIT COLLARLESS SUIT BLAZER | 1,899,900 UZS |

The blanks are rendered as the empty gray image placeholder while the item name, price, and `Buy` button remain correct.

All five image URLs returned `200 image/jpeg` with non-zero payloads (25 KB–474 KB), so the assets themselves are valid. A 20-second-wait spot check of scenario 08 produced a fully loaded deck (different randomized items), which points to **slow image loading / placeholder timing** rather than permanent broken URLs. The 5-second capture window is enough to expose the placeholder state.

### Layout and content checks

- 13/17 result screens were visually clean: every tile rendered, no overlap, no clipped names or prices.
- Card totals, `90% Match` badges, and buttons align consistently across all 20 runs.
- No incorrect currency symbols; UZS prices are readable. Scenario 02 correctly shows the USD budget reference (`$194.69 / $250`, `$236.49 / $250`, `$232.30 / $250 · CBU reference estimate · 2026-10-22`).
- `Add to Calendar` sits just below the initial viewport in some first-card screenshots; it is fully visible after the first scroll. This is expected scrolling behavior, not clipping.
- Product names are sometimes localized by the catalog (for example `Лоферы ...`, `Свитер ...`) inside an English UI. The UI strings remain English; this is a catalog content-consistency issue.
- Scenario 20 (`sayohat uchun, oddiy`) generated a deck whose summary reads **“A Casual look”**; the requested travel occasion is not reflected in the copy.

### Failure screens

`overview_failures.png` confirms all three deterministic failures remain on the stylist screen with the sticky `Generate outfit` footer. The red error banner is partially hidden behind that footer, truncating the message mid-sentence:

- `11`: “The AI couldn't finish your outfit. Please try…”
- `13` / `14`: “No clothing matches your selections. Add…”

## Failure analysis

### 11 — black tie wedding, classic (503)

- 3/3 attempts failed in Run 1 and 3/3 in Run 2.
- Direct call with the same session, prompt, style, occasion, and `limit: 5` returned:
  ```json
  {"status":503,"error":"ai_generation_unavailable","success":false,"providerAttempts":[]}
  ```
- `providerAttempts` is empty; no usable model attempt is reported.
- Likely cause: the formal/black-tie path cannot assemble a complete outfit from the current live shop pool and fails at the AI-generation stage instead of returning a structured no-match response.

### 13 — gym workout, casual (422)

- Direct call returned:
  ```json
  {"status":422,"code":"no_matching_clothing","error":"The selected clothing pool has no complete outfit suitable for the requested style, occasion and weather."}
  ```
- The shop catalog has tailoring, knitwear, trousers, and loafers, but no complete athletic outfit. The edge correctly rejects the pool; the app does not explain the missing category.

### 14 — sport run, minimalist (422)

- Same `422 no_matching_clothing` response as scenario 13.
- Deterministic and catalog-driven, not a provider outage.

### 20 — transient AI failure (recovered)

- Four consecutive attempts failed with the generic AI banner (`503` path in the app), then a retry passed.
- A direct call for `sayohat uchun, oddiy` / `casual` / `travel` returned `200`, `success: true`, 5 outfits, and 1 provider attempt.
- This is the strongest evidence that the AI provider is intermittently unavailable and the client should retry.

## Evidence

- Run 2 screenshots (4 per passing scenario):
  `.maestro/verify-shots/sim20native_rerun/<scenario>/sim20_<scenario>*.png`
- Visual contact sheets:
  `.maestro/verify-shots/sim20native_rerun/contact/<scenario>_contact.png`
- Overview sheets:
  - `.maestro/verify-shots/sim20native_rerun/contact/overview_first.png`
  - `.maestro/verify-shots/sim20native_rerun/contact/overview_failures.png`
- Failure screenshots:
  - `.maestro/verify-shots/sim20native_rerun/11/sim20_11_fail.png`
  - `.maestro/verify-shots/sim20native_rerun/13/sim20_13_fail.png`
  - `.maestro/verify-shots/sim20native_rerun/14/sim20_14_fail.png`
- Maestro logs:
  - `.maestro/verify-shots/sim20native_rerun/run_01_02_03_04_05.log`
  - `.maestro/verify-shots/sim20native_rerun/run_11_12_13_14_15_16_17_18_19_20.log`
  - `.maestro/verify-shots/sim20native_rerun/run_20.log`
- Direct edge probes:
  `.maestro/verify-shots/sim20native/failure_probes.log`
- Run 1 evidence:
  `.maestro/verify-shots/sim20native/`

## Observations

- **Blank image tiles are a real first-render UX issue:** 4/17 passing scenarios showed at least one gray placeholder in the first card at the 5-second capture point. The URLs are valid and a longer wait loaded images, so the fix is likely prefetch/retry/placeholder handling rather than catalog repair.
- **Error banner is obscured:** all failure messages are truncated behind the sticky `Generate outfit` footer.
- **Sport has no catalog path:** gym/sport requests consistently dead-end in shop mode.
- **Black-tie has no fallback:** it returns `503` instead of a structured no-match response.
- **Transient AI failures:** 4 scenarios needed retries in Run 2; scenario 20 required 4 failed attempts before passing.
- **Weather was unavailable** in the inspected simulator runs; the app used its weather fallback.
- **Localization:** Russian intent works; Uzbek travel intent generates a Casual-labelled deck, and catalog product names mix Russian/English inside the English UI.

## Recommended fixes

1. Prefetch or retry product images and keep the placeholder until the image actually renders; consider delaying the visual-ready state until all card images settle.
2. Move the error banner above the sticky footer or add bottom padding so the full message and suggested fix are visible.
3. Return a structured `no_matching_clothing` error for black-tie requests instead of `ai_generation_unavailable` when no formal candidates exist.
4. Add an activewear/sport capsule to the live shop catalog, or show a sport-specific empty state with category guidance.
5. Add client retry/backoff for transient `ai_generation_unavailable` responses with a visible retry action.
6. Review the Uzbek travel→copy pipeline and catalog product-name localization.

## Commands

```bash
# Run all 20 scenarios against the native simulator app
FLOW=.maestro/sim_personal_stylist_20_native.yaml \
OUT=.maestro/verify-shots/sim20native_rerun \
./.maestro/run_personal_stylist_20_sim.sh

# Run a subset
FLOW=.maestro/sim_personal_stylist_20_native.yaml \
OUT=.maestro/verify-shots/sim20native_rerun \
ONLY=11,13,14 \
./.maestro/run_personal_stylist_20_sim.sh
```

The deterministic service-level suite remains at `docs/reports/PERSONAL_STYLIST_20_TEST_REPORT.md` (20/20 passing).
