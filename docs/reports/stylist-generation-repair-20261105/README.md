# Stylist generation repair — November 5, 2026

Base: `885e9f581b60018f3470dde4724ca6cc11b49d72`. Isolated branch: `codex/stylist-generation-repair-20261105`.

## Findings and changes

- **Formal HTTP 200 → client failure:** independently downloaded active Supabase `generate-outfits` v179. In a local run of that source, the server returned three successful Formal outfits containing chinos; the latest client rejected all three with `formal_requires_tailored_bottom`. The new client response boundary repairs labels, records rejection stages and searches returned products plus loaded live catalog inventory for complete compatible replacements before budget/diversity selection. Validation remains mandatory. An exhausted pool returns the localized `noMatchingClothing` shortfall. This reproduces a concrete mismatch; the exact two historic simulator requests were not recovered.
- **Compatibility gaps:** Interlock/jersey bottoms without tailoring evidence were treated as generic trousers, utility shirts could accompany formal blazers, and bulky biker layers were not evaluated against refined tailoring. Shared rules now catch those combinations. Knit construction is not labelled athletic without evidence; explicitly tailored jersey remains distinguishable. Coherent casual uses of these garment families remain allowed.
- **Fallback bypass:** closest-match fallback previously validated structure/budget but only penalized incompatible style, dress code and weather. It now uses the existing bounded quartet search and full style/context judge, including preferences, source, anchor and budget.
- **Travel ranking:** a 0–1 activity score was overwhelmed by raw style points in retrieval. Retrieval and trimming now combine normalized style/context scores. Everyday travel favors removable layers and supported walking features, while retaining business-travel exceptions. The global pool stays capped at 40; per-slot reserves follow ranking before catalog arrival order, retaining cheapest floors and scarce-slot coverage.
- **Repeated text:** description and tips received the same generic explanation. Descriptions now identify all four actual products; separate English/Russian/Uzbek tips explain layering, occasion and uncertainty. Product descriptions/material/pattern and shopping links survive the candidate mapping.
- **Usage:** `consumeOrPaywall` invokes `check_and_increment_usage`; it is not just an access check. Paid tiers currently use `aiOutfits: -1` (unlimited), so purchased-credit loss was not established. The button now checks access first and records usage only when validated results are ready for display. Display-history storage failure no longer clears a successful result.

## Read-only deployment evidence

Project: `fyqpifmrsftsfqibhwhy`. Function status: ACTIVE; version: **179**; updated: **2026-10-30 15:06:58.229 UTC**.

Management bundle SHA-256: `7249e5b3d639789bee75c0117de382286d19acf656fbd12e37c47e82879967db`.

Downloaded entry SHA-256: `3149017bbe4018a5b0786aaf7ad5f9be3af33b0bff5fda1e31304fd1a5e354d3`.

The deployed bundle embeds older `_shared/occasionRules.ts`. It lacks the current domain garment-context Formal requirements, directional layering/whole-look coherence, and recent semantic look history used by the latest mobile source. It already includes four-slot validation, product rotation and overlap metadata; those were not assumed absent simply because its deployment is older.

## Verification

- Final focused Jest run: **12 suites, 155 tests passed**. Includes Classic + Formal, Casual + Formal, complete four-item contracts, recovery with budget/source/anchor/preferences, reported bad combinations, acceptable casual counterexamples, localization, usage handling, catalog ranking and transport/auth behavior.
- Additional hosted-handler and pool regression run: **2 suites, 69 tests passed** (pool tests also included in final run).
- TypeScript `tsc --noEmit`: passed.
- Deno check of `generate-outfits/index.ts` with the frozen lock: passed.
- ESLint of modified mobile/domain modules: **0 errors**, 173 warnings; no unrelated cleanup performed.
- Local integration exercised both the downloaded v179 handler and the updated handler through response repair, current client validation, budget checks and diversity selection. Provider/auth/database calls were local mocks; no hosted generation was invoked.
- A separate local smoke used **126 current catalog rows fetched read-only** from Supabase (40 tops, 40 layers, 40 formal-trouser candidates, 6 dress-shoe candidates). It returned HTTP 200 and three current-client-valid four-item Formal looks. Public item IDs, names, original prices/currencies and shopping links are retained in [live-catalog-smoke.json](live-catalog-smoke.json). This live-catalog run had no spending cap; budget regression coverage uses explicit caps separately.

No full Jest run or iOS Simulator run of this branch was performed. The existing running simulator/Metro instance belongs to the separate dirty working tree. Full native rendering and real provider behavior therefore remain unverified for these changes.

## Deployment boundary

No deployment, hosted data/configuration/billing mutation, migration, merge or force push was performed. The original working tree and its uncommitted work were preserved. Main is unchanged. The shared backend fixes require a separately authorized Edge deployment to affect production; pushing this branch does not update Supabase. Client recovery can handle compatible older-server outputs only when authentic suitable companion products are available in the loaded inventory.
