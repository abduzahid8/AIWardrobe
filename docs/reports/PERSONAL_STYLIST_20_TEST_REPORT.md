# Personal Stylist — 20-Test Acceptance Report

**Date:** 2026-10-22
**Project:** AIWardrobe (`/Users/v.zahid/AIWardrobe`)
**Suite:** `__tests__/services/personalStylist20.test.ts`
**Command:** `npx jest __tests__/services/personalStylist20.test.ts --runInBand --verbose`

## Executive summary

| Metric | Result |
|---|---|
| Tests written | 20 |
| Tests passed | **20** |
| Tests failed | **0** |
| Suite runtime | 289 ms |
| Stability check | 20/20 consecutive runs passed (0 flakes) |
| TypeScript | `npm run typecheck` — clean |
| ESLint (new file) | clean |
| Live services required | none (deterministic) |

The 20 tests cover the personal stylist end-to-end decision path: intent parsing, persona voice, localization, style knowledge, system-prompt construction, outfit generation, budgets, shop-only decks, and the outfit quality judge.

## Test results

### A. Intent understanding (1–5)

| # | Test | Protects | Result |
|---|---|---|---|
| 1 | English work request with style and budget | `old_money` + `work` + `$200` parse correctly | PASS |
| 2 | Russian work request with style and budget | `классика` + `работ` + `150$` parse correctly | PASS |
| 3 | Uzbek travel request with a budget | `sayohat` maps to `travel`, `$300` retained | PASS |
| 4 | Smart-casual occasion separated from budget | `weekend casual brunch under $200` → casual/casual/$200 | PASS |
| 5 | Empty and whitespace input | No crash; returns null style/occasion/budget | PASS |

### B. Persona voice and localization (6–9)

| # | Test | Protects | Result |
|---|---|---|---|
| 6 | Named stylist with wardrobe-first greeting | Identity is Ava; greeting promises wardrobe-based help | PASS |
| 7 | Russian work reply | Russian text, Russian action chips, 3 suggestions | PASS |
| 8 | Uzbek budget reply | Uzbek text, budget guidance, 3 suggestions | PASS |
| 9 | Regional locale fallback | `ru-RU` → ru, `uz-UZ` → uz, unsupported → en | PASS |

### C. Style knowledge and prompt engineering (10–13)

| # | Test | Protects | Result |
|---|---|---|---|
| 10 | All six style pillars complete | Every style has label, vibe, palette, fabrics, pairing, avoid | PASS |
| 11 | Style alias normalization | `OLD MONEY`, `old-money`, `semi classic` resolve; unknown → casual | PASS |
| 12 | System prompt identity and pipeline | Ava identity, style direction, all 7 pipeline steps included | PASS |
| 13 | Contextual prompt constraints | Occasion, weather, favorite/avoid colors, and budget injected | PASS |

### D. Outfit generation and commerce rules (14–17)

| # | Test | Protects | Result |
|---|---|---|---|
| 14 | Empty wardrobe | Returns one shopping-path suggestion with top/bottom/shoes gaps | PASS |
| 15 | Three distinct complete looks | 3 looks, each complete, each keeps at least one owned piece | PASS |
| 16 | Budget enforcement | Shop spend ≤ $100; no over-cap luxury pieces | PASS |
| 17 | Shop-only Team Collaboration deck | Zero wardrobe pieces even with a full wardrobe available | PASS |

### E. Outfit quality judge (18–20)

| # | Test | Protects | Result |
|---|---|---|---|
| 18 | Workout formality | Athletic outfit accepted; office clothing rejected for gym | PASS |
| 19 | Black tie formality | Real tuxedo accepted; ordinary navy suit rejected | PASS |
| 20 | Coherence and climate guardrails | Sneakers + tailored trousers rejected; linen at 10°C rejected; cold outfit without warm layer rejected | PASS |

## Coverage map

| Stylist subsystem | File under test | Tests |
|---|---|---|
| Intent parsing | `features/outfit-generator/utils/stylistIntent.ts` | 1–5 |
| Persona identity / voice | `src/services/ai/stylistPersona.ts` | 6–9, 12–13 |
| Style knowledge | `src/services/ai/stylistPersona.ts` | 10–11 |
| Outfit generation | `src/services/outfitGenerationService.ts` | 14–17 |
| Suggestion engine | `src/services/suggestionEngine.ts` | 14–15 |
| Quality judge | `backend/supabase/functions/_shared/occasionRules.ts` | 18–20 |
| Budget rules | `src/utils/outfitBudget.ts` | 16 |

## Verification

```text
PASS __tests__/services/personalStylist20.test.ts
  personal stylist — 20 acceptance tests
    ✓ 1. understands an English work request with style and budget
    ...
    ✓ 20. enforces coherence and cold-weather guardrails

Test Suites: 1 passed, 1 total
Tests:       20 passed, 20 total
Snapshots:   0 total
Time:        0.562 s
```

Additional checks:

```text
npm run typecheck        -> exit 0
npx eslint <new test>    -> exit 0
20 repeated runs         -> 0 failures
```

## Scope and limitations

- These are deterministic React Native service/pipeline tests. They do **not** launch a simulator or call the live Supabase edge function, NVIDIA, or a paid AI provider.
- They verify the stylist's local contract: parsing, prompt context, deck construction, budget handling, localization, and rule-based judging.
- Device-level visual behavior (layout, image rendering, animations) remains covered by the existing Maestro suites, including `.maestro/STYLIST_50_REPORT.md`.
- The 20 scenarios were also executed on the iOS simulator; see `docs/reports/PERSONAL_STYLIST_20_SIMULATOR_REPORT.md` for the 17/20 pass result and the three live-edge failures.
- Live model quality is not asserted; the suite checks the guardrails and contracts around the model, which is the intended boundary described in `docs/ARCHITECTURE.md`.

## Recommendations

1. Keep this suite as a fast regression gate in CI before the Maestro runs.
2. Add one live-edge smoke test separately (mocked credentials are already used in `stylistCanonicalRequest.test.ts`).
3. If stylist copy changes, update tests 6–13 together with `stylistPersona.ts`; they intentionally pin identity, language, and prompt structure.
