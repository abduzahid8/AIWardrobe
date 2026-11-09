# iOS Stylist 5 screenshot QA — 2026-11-10

Status: **NOT ACCEPTED**. Evidence: 15 user-provided iPhone simulator screenshots, captured 00:00–00:03 on 2026-11-10 (Uzbekistan local time). Binary screenshots remain in the user conversation; no personal media is committed to the repository.

## Observed facts

- The mobile result view renders 4-slot shop outfits with images, titles, UZS prices, buy buttons and result actions. This is evidence of an app rendering outfits, **not proof of the PR #14 staging Node route**. The production mobile URL remains `/functions/v1/generate-outfits` and Supabase production is v207.
- Result descriptions and sparkles/tip box sometimes display the **same sentence** (e.g. Uzbek sentence beginning “Klassik poyabzal bichimli shim siluetini to‘ldiradi.”). Also seen: generic “To‘liq obraz: asosiy ustki kiyim, ikkinchi qatlam, shim va poyabzal.” It does not distinguish individual outfits.
- Multiple different product sets are headed by concept titles equivalent to `pijak · oksford`, `pijak · derbi` and `pijak · lofer`, making variants feel repetitive.
- One Semi-Classic + Date night outfit includes a blue short-sleeved all-over patterned shirt, dark tailored blazer, dark pleated trousers and dress derbies. This is a **human-fashion concern**, not yet a formally reproduced backend policy violation.
- Old Money + Work and Classic + Work screenshots show similar blazer/shirt/trouser/derby or loafer families, with some color/product variation. Visual distinctiveness remains insufficiently proved.
- Partial localization: Uzbek explanatory copy/buttons coexist with English category labels (TOP/LAYER/BOTTOM/SHOES), style labels and the form text (“New outfit”, “Weather unavailable”). Product catalog names should remain original, but app-owned UI text should be localized when available.
- Product tile names are truncated; this is a UX shortcoming. Price tiles and item totals appear in UZS.
- The “1 500 000 UZS” string shown in the generation form appears to be placeholder example text. **Do not infer that a budget was set or that displayed looks violate a hard budget** without the captured outbound request or visible active budget indicator.

## Code inspection and hypotheses

- Current `screens/AIOutfitmaker.tsx` calls `outfitDescription` and `distinctStylingTips` in `wrapLook` and re-localizes on language change. `src/services/stylingCopy.ts` filters exact duplicate normalized tips. The exact repeated copy in screenshots is therefore potentially caused by an **older installed JS bundle**, a different rendering path, or differences between the original and translated strings. Investigate the actual client revision/version first; do not assume a current-source regression solely from screenshot evidence.
- Current `backend/supabase/functions/generate-outfits/index.ts` exposes generationVersion `stylist-5-editorial-v2-rc-20261109-r2`, while PR #14 isolates protected Node compute with a separate staging URL.
- Verify server and client source fingerprints, generationRequestId and engineVersion, without dumping user access tokens or product lists to logs.

## Acceptance tasks (must be tested, not merely coded)

1. **Build provenance:** capture installed iOS JS revision, backend generationVersion, source/provenance and request correlation ID for each screenshot scenario; identify whether any is an older mobile bundle. No secret-bearing logs.
2. **Copy QA:** ensure the tip card is hidden for exact/semantic restatements of the description in en/ru/uz. Each displayed tip must add at least one supported fact, otherwise omit it. Test description/tip for each provided screenshot scenario and language switch.
3. **Fashion QA:** evaluate Old Money+Work, Classic+Work, Semi-Classic+Date night with human reviewers and actual catalog garment evidence. Flag conspicuous vacation prints under tailored jackets in date looks; do not silently lower quality thresholds or introduce hard rules based only on appearance.
4. **Concept diversity:** measure silhouettes, formality, top/layer patterns, footwear family and repeated product identity across all 3 looks and repeated generations. Repeated generic header should not substitute for a meaningful visual concept.
5. **Localize UI:** app-owned category labels, weather-unavailable text, headings, buttons and occasion summaries should follow the selected locale; don't translate store-authored product names. Preserve current en/ru/uz support.
6. **Budget truth:** verify whether a real active `preferences.maxShopBudget` is present on the actual request. Under a real UZS 1,500,000 cap, every generated shop outfit must fit (or an honest no-valid-outfit result must appear). Placeholder does not activate budget.
7. **Transport/no-go:** PR #14 staging positive authenticated smoke, exact quota deltas and latency remain unproven. No production cutover based on these screenshots.

## Done means

Reproduced evidence, minimal source changes, tested mobile build and exact commit SHA, screenshots of corrected scenarios, localized copy nonduplication, catalog/source/price checks, and a separately authenticated staging smoke report including quota+1/request and Node CPU/wall. Do not mark release GO until production rollback artifacts, security and performance are validated.

Production Supabase `generate-outfits` must stay v207 during this investigation.
