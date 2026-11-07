# Bounded quality floor — verified candidate, not deployed

Base: `d9d7ffc779069b352dca3e55badf29dd6f0d5332`, the published styling refinement. User reports production v187 as the current working release. No production source/log API calls or deployments were made in this task.

Correction: `4a8c3ef619e7e3139a138af6b5ff168febffca4d`, branch `codex/stylist-quality-floor-20261107`.

## Trace and correction

The complete/fresh-deck shortcut considered only deck length and recency. Soft coherence penalties influenced ranking but never made a hard-valid look ineligible for that shortcut. `assessOutfitQuality` also averages six largely full-score dimensions with coherence: the weak Work fixture still had aggregate score above 0.85 and no hard rejection. `signalsFor` had no acceptance verdict, so no reserve replacement was requested.

The new request-memoized acceptance verdict uses raw compatibility, before recency penalties: a 0.75 floor for strict Formal/refined Work and a 0.65 floor for Minimalist Casual. Refined looks also fail if layering, footwear, or style compatibility is below 0.5. A business blazer with positively identified severe/shredded denim receives a scoped incompatibility penalty for Minimalist Casual. Clean denim, casual/unstructured blazers, minimal sneakers and coherent casual looks remain permitted; Formal rules are not imposed on Casual.

Weak cards are removed before the fast path. The existing reserve admits only hard-valid, above-floor candidates. When repairing a short deck caused by a quality rejection, the retained good cards are protected by physical-product pruning, a retention objective, and a mandatory retention check. Their shoes are pinned during the existing shoe reassignment. Unused weak candidates do not pin an otherwise complete deck or change its recency search. All-weak decks can enter the same reserve. The final response is checked again after shoe substitution.

No acceptable replacement means HTTP 422 with an empty deck, never an eligibility waiver or two-card success. `quality_replacement_unavailable` is classified outside the screen's closest-match fallback, so it cannot silently rebuild a weaker deck. Current clients' unknown-code default also fails closed. Memo keys now include full row evidence: identical IDs/names with changed construction descriptions cannot reuse a stale acceptance verdict.

## Focused verification

- **97 focused tests passed**: 75 domain/CPU/error regressions, 7 new real-handler quality-repair cases, 8 selected recency/anchor/deck cases, and 7 selected budget/auth/source cases. No full suite.
- Full-handler tests parse three hard-valid provider looks and replace only the weak one, preserving both good product sets including shoes; an owned anchor remains in every card. Budget, previous-use, duplicate-photo and missing-reserve cases return no outfits. An all-weak provider deck is replaced from valid inventory.
- **12 offline post-provider handler replays**, 96 garment rows each, HTTP 200 with three outfits. Maximum process CPU **134.176 ms** under the **500 ms** ceiling. New Work, anchored Work and Minimalist Casual repairs took **40.024 / 44.844 / 37.449 ms**. Replays assert retained good cards, floor acceptance, source identities, four slots, budget, recency and anchors. Module compilation/network/provider latency are excluded.
- Quartet enumeration, shoe rotation and diversity selector modules remain byte-for-byte unchanged. Reserve bounds remain **384 attempts / 24 admitted looks / 12 rows per slot**, with **512 diversity steps** per selection. Request admission is unchanged. Hashes and raw CPU evidence are alongside this file.
- The exact compiled release bundle passes offline OPTIONS 200, GET 405 and unauthenticated POST 401 startup checks with zero network attempts.

## Production evidence limit

Explicit `tropical-pattern short-sleeve shirt` plus `drawstring trousers` already fails the existing hard checks. So does explicit distressed/ripped denim under the existing Minimalist rules. Tests retain those rejections; representative missing-synonym/soft-coherence cases demonstrate the new floor. Exact product IDs/catalog metadata from the reported failed card were not supplied, so this work does not establish why those visual attributes were absent from, or bypassed, the live card's metadata. A metadata-only gate cannot identify a tropical print from a generic product title/photo. No live visual signoff is claimed.

## Prepared Edge artifact

`aiwardrobe-generate-outfits-v3-quality-floor-20261107.tar.gz`

- Archive SHA-256: `229eef03cfedcd9286f99b817eea6014a31bab348f5cba2e5ddaf4879e75196e`.
- Bundle SHA-256: `6da017677bd166f65c3adbbf61021a4dc1b80d19c8424e500b7f185ea48400d1`, 365,345 bytes, esbuild 0.28.2; all 38 source-input hashes recorded.
- Marker: `stylist-runtime-repair-v3-quality-floor-20261107`; this is not an assigned hosted version.

Original release artifacts are unchanged. No deployment, main merge, subscription, billing, database or secret changes. Any later deployment must recheck the then-current production baseline and preserve existing JWT settings.
