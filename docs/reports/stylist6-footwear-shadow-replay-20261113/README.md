# Stylist 6 — footwear selection impact, controlled offline A/B

Date: 2026-11-13. Branch: PR #15, stylist6-visual-first.

## Scope

Read-only offline engineering experiment. Historical merchant catalog files,
no NVIDIA, no product photo downloads, no live Supabase writes, no migration,
no user-facing activation. The four *inactive* Tommy Hilfiger review entries
were temporarily marked active **only in a private copied JSON file** for
retrieval diagnostics. This is not authorization to publish those entries.

Two identical 40,989-row historical snapshot imports were compared; the
second included only four extra hypothetical shoes. Both ran the same eight
scenarios, retrieval constraints, budgets and style decisions, with V5 skipped
and visual reviews marked NOT_RUN. Original committed benchmark report was
restored; detailed deck results were kept in private /private/tmp.

## Actual result

| Measurement | Original snapshot | Original + 4 Tommy shoes |
| --- | ---: | ---: |
| Raw catalog entries | 40,989 | 40,993 |
| Catalog items admitted by Stylist 6 | 38,047 | 38,051 |
| Full retrieval looks across 8 scenarios, 2 arms | 42 | 42 |
| Unique shoes selected among those looks | 22 | 22 |
| New Tommy shoes in selected looks | 0 | 0 |
| Changed generated looks | 0 | 0 |

Scenario IDs: S6-01, S6-02, S6-04, S6-05, S6-07,
S6-08, S6-B01, S6-B03. S6-B01 (1.5M UZS full-look
budget) produced no outfits in either run.

Caution: historical source expands variants to 38k rows, **not** the live
Supabase catalog of 7,020 active rows. These retrieval tests are neither
independent human fashion reviews nor a claim about future live selection.

## Concrete defect and bounded fix

The imported Tommy footwear titles often contain no English color.
The Stylist 6 catalog adapter therefore projected color UNKNOWN for every
one of the four new shoes, despite merchant product URLs carrying stable
black, brown and other transliterated Uzbek/Russian color suffixes.

- A pure, explicit FR Group color mapping now reads only exact HTTPS
  frgroup.uz/ru/store product path suffixes.
- Explicit color in the product title takes priority; unsupported and
  multicolor suffixes remain UNKNOWN, no guesses.
- The adapter version is now catalog-adapter-v3-merchant-color.
- Tests cover legitimate black/brown/beige/blue and spoofed domains,
  unrelated routes, malicious query parameters and title conflicts.

The same eight-scenario A/B replay after the color fix **still produced
zero newly selected shoes and zero changed looks**. This establishes a
correctness improvement, NOT a fashion quality win.

## Next release gates

1. Better locally shoppable loafers, Oxford, Derby and Chelsea variants,
   not simply more duplicate casual shoes.
2. Explicit currently selected size, verified seller price and merchant
   shipping/checkout evidence; FR Group stock observations are time-limited
   and NOT reservations.
3. Product photo display and NVIDIA inference rights; HTTP 200 and indexed
   category page photos are not authorization.
4. Verify real staging catalog retrieval and independently rate complete
   outfits against Stylist 5; production remains NO-GO.

No database, production function, active catalog rows, or paid provider
quota was modified in this experiment.
