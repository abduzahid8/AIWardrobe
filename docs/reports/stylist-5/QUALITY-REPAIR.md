# Autonomous stylist quality repair — 2026-11-08

Starting branch: `codex/stylist-5-integrated-20261108`, fetched and fast-forwarded to `bcda5f1da2d2745b192686e8fbff384b646e12fa`. The unrelated dirty primary checkout was preserved. GitHub implementation commit: `01d01b07b1b45910b2939bb3c72ece792799f3e3`. Its Git tree `e1bae89058fa03fedf4d4fde9e3866e8f55f885f` exactly matches the locally tested implementation commit `a3137e359fd9f112721f563a72d0d1667cc367e2`. Runtime identifier: **`stylist-5-editorial-v2`**, emitted in successful responses and privacy-safe request logs. No production deployment, merge, Vercel setting change, or paid inference was performed.

## What was already implemented

The starting PR already had concept-first planning, cached visual descriptors and hydration, compact per-slot SQL retrieval, physical identity deduplication, bounded NVIDIA complete-look ranking, deterministic fallback, optional-layer negotiation, explicit budget enforcement, and fresh-slot/shoe rotation. Production v203 screenshots cannot establish a defect in those unreleased changes. This investigation reproduced the latest branch rather than treating the screenshots as current-code results.

## Root causes and implementation

1. **A prose parser contradicted the structured style policy.** Final validation split `Avoid`/`NEVER include` sentences on commas and treated every fragment as a forbidden substring. A sentence recommending navy, charcoal, and brown alternatives therefore rejected charcoal. Removed this duplicate executable interpretation of prompt prose. Canonical item eligibility, source/slot validation, preference exclusions, climate/activity checks, and whole-outfit quality acceptance remain in force. New hosted tests exercise charcoal outfits in Classic, Old Money, and the shop-only Business Casual contract.
2. **Concept reservation compared different garment purposes.** `stylingCandidateRepresentatives` compared every reserved concept role to the globally highest-scoring garment. A Minimalist overshirt preference could remove every cardigan before a knit-layer outfit was evaluated. The existing score band now compares candidates within their compatible concept role. Caps, alias collapse, eligibility and downstream quality floors are unchanged. Regression coverage retains a valid cardigan while rejecting a much weaker candidate within that same role.
3. **A partially feasible legacy brief hid usable inventory.** Classic/date with locked Oxfords had three bottoms, but its feasible office brief admitted only one tailored bottom and discarded chinos. A legacy-only role reserve now includes available roles when the feasible briefs cannot cover the requested physical choices. Every combination still passes the same caller evaluator. Updated clients retain semantic diversity requirements; this reserve does not claim three editorial concepts from identical legacy products.
4. **Localized explanations were unreachable for normal versioned responses.** Both mobile mapping paths preferred nonempty English server fallback text before checking `explanationVersion`. A shared description resolver now renders version-1 grounded facts in the active locale and preserves legacy free-form prose. Product names remain authentic. Added missing concept title translations, localized variant-title fallback and slot labels in English, Russian and Uzbek.
5. **Result imagery cropped the actual products.** The results grid used `cover`; it now uses `contain` with its existing background and cache behavior. This preserves the entire source frame, including tall trousers and shoes, without changing product identity. It cannot restore a product already cropped inside its source photograph.
6. **Budget status referenced generation-local state from render.** The existing banner referenced out-of-scope `shopBudget`. The validated result budget now lives alongside result state. Translated copy labels look total versus budget cap; unavailable display FX falls back to the known source currency instead of an invented UZS quote. Enforcement remains based on the original cap and approved FX snapshot.
7. **Checks included archived deployment source as application code.** TypeScript now excludes `docs/reports` artifacts, while retaining the application and tests. Fixed two actual implicit-any errors in the active Edge handler. App request admission logging uses the existing privacy-safe logger; server request logs retain the generation version.

## The original six hosted failures

The starting `fourSlotHosted.test.js` result was reproduced exactly: **30 passed / 6 failed**.

| Failure | Finding and treatment |
| --- | --- |
| Snowy Work | Generic `Leather boots` had no verified dress profile and was rejected by the legacy office admission contract. The positive climate fixture now identifies Chelsea boots and a charcoal overcoat. This also exposed the real charcoal prose-parser bug. No climate or quality waiver was added. |
| Piqué anchor / paged layers | Query-builder fixture modeled a retired API and emitted no `fashion_slot`. Updated it to the actual compact RPC and per-slot offset contract. Positive cardigan fixtures now supply explicit color/fine-knit evidence rather than relying on unknown metadata to meet the quality floor. |
| Paged compatible tops | Same obsolete fixture contract. The corrected test proves page two supplies six shirts and six layers, but **still expects quality rejection** for the weak pullover/jersey combination. It asserts the precise quality-stage code, candidate counts, and no outfits; it no longer demands that any complete quartet be accepted. |
| Classic locked Oxfords | Real concept-role coverage gap; fixed with the legacy role reserve and a direct planner regression test that also verifies a rejecting evaluator cannot be bypassed. |
| Piqué companions | Unspecified merino cardigan scored below the existing refined-date floor. The positive fixture supplies explicit fine-knit/color evidence. A separate sparse-data regression asserts that the original under-supported combination still fails the unchanged 0.75 floor. |
| Misfiled coated jeans | Retired query-builder fixture; now verifies actual compact-RPC pagination past 100 mislabeled rows and insists on a valid blazer. |

No tests were skipped. Three explicit charcoal regressions were added. The unrelated quality-floor test's aggregate assertion now compares against the actual required minimum rather than an obsolete 0.85 incidental score; rejection assertions remain intact.

## Real catalog replay and image review

Replayed all six fixed scenarios with PostgreSQL 16 against the existing immutable **7,020-row** catalog snapshot, SHA-256 `dc4853cdb301b55cb8828bdd9e78a778bfc2de64a5da98c93dbe27dd8cafcfa6`. Applied the compact migration only to the isolated local fixture and exercised compact/hydration privileges and contracts. The benchmark and contact sheets made **zero provider/inference calls**.

`benchmark.json` compares the cumulative current PR to the preserved v200 kernel, not to production v203. `quality-repair-role-comparison.json` isolates the additional role-reservation repair using identical frozen inputs and retrieval. Its "before" is this repair before the role-comparison change, not the untouched PR head.

| Scenario | Selected concepts before → after role repair | Minimum semantic distance before → after |
| --- | --- | --- |
| Casual / Travel | 2 → 3 | 0.660 → 0.719 |
| Semi-Classic / Work | 3 → 3 | 0.591 → 0.591 |
| Old Money / Work | 3 → 3 | 0.680 → 0.680 |
| Classic / Work | 3 → 3 | 0.458 → 0.458 |
| Semi-Classic / Date night | 3 → 3 | 0.446 → 0.446 |
| Minimalist / Everyday | 1 → 2 | 0.435 → 0.544 |

All six return three validated looks. These computed diversity measurements support improved candidate access; they do **not** establish human preference or three genuinely different concepts in every request. Minimalist still selects only two concepts. Quality acceptance thresholds, weights and evaluation limits were not lowered.

Regenerated six private real-product comparison sheets and inspected images, including Minimalist and Date night. The Minimalist candidate now includes a visible knit/cardigan direction alongside the overshirt looks, rather than three tee/overshirt variants. Some fabric/proportion judgments remain uncertain: a photograph may reveal heavier knit construction than its text descriptor supports. **15 of 82 unique image downloads were unavailable** in this local review. This is a review-fetch observation, not proof that all those images fail in the iOS app. Private products, embeddings, prompts, account data and image sheets were not uploaded to GitHub.

Reproduction, with the original private frozen fixture provisioned locally:

```sh
node scripts/stylist5/benchmark.mjs /path/to/private/frozen
python scripts/stylist5/contact-sheets.py /path/to/private
node --test scripts/stylist5/compact-contract.test.mjs
```

## Validation

- Hosted four-slot integration: **39/39 passed**, including the original rotation, source, anchored discovery and new charcoal cases.
- Focused policy/planner/visual/copy/NVIDIA/compact-contract checks: **11 suites, 82/82 passed**.
- PostgreSQL compact/hydration security and budget contract smoke: **1 passed, 0 skipped**.
- `npm run typecheck`: **passed**.
- Deno check of the active `generate-outfits/index.ts` with the frozen lockfile: **passed**.
- Repository `npm run lint`: **0 errors**, existing warning backlog remains (1,496 warnings in this run).
- `git diff --check`: **passed**.
- Broader legacy `outfitBudgetWire.test.js`: **70 passed / 25 failed**, both on untouched `bcda5f1` and final implementation `a3137e3`; **the 25 failing test names match exactly, with zero new failures and zero skipped tests**. See `quality-repair-legacy-wire.json`. Many assert the retired product-generation/correction protocol or old catalog fixtures; the active rank-only protocol is covered by the passing NVIDIA tests. Authentication, entitlement, malformed-budget and pricing rejection tests in the legacy suite continue to pass. The repository is not claimed universally green.

## Remaining acceptance and release limitations

- No current production v203 invocation, authenticated hosted Auth/RLS smoke, production latency percentile measurement, or real NVIDIA ranking run was performed. Auth/budget behavior was exercised with mocks; the local database smoke does not substitute for hosted authorization checks.
- Three validated physical outfits do not universally mean three concepts. Minimalist has two in this frozen replay; legacy four-slot clients retain their physical-card compatibility behavior.
- Without an explicit budget, the grid still shows individual prices; no unsupported cross-currency grand total is invented.
- Explanation language is fixed at generation/mapping time. Authentic product names may remain English in Uzbek/Russian prose. Unknown visual/fit/fabric facts remain unknown.
- Results image containment, translated headers/narratives, budget banner, long names and optional-layer layouts require real iOS Simulator review at the final client revision. Current production screenshots cannot validate this client bundle.
- The existing protected Vercel project-setting issues and separate deployment approvals remain outside this repair. No production Supabase or Vercel change was made. Release readiness still requires hosted verification and the owner's visual evaluation.
