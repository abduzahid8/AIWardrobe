# Stylist-5 Launch Candidate — pinned reconciliation report

Date: 2026-11-09
Reconciled base (reviewed PR tip): `962bd102` (`origin/codex/stylist-5-integrated-20261108`)
Candidate branch: `codex/stylist-5-launch-candidate-20261108`
Preserved local snapshot: `2a74913` (tag `backup/local-integrated-20261108`), bundle and
pre-reconcile patch kept under the session temp directory. The original worktree
`/Users/v.zahid/AIWardrobe` was not modified, reset, cleaned or re-pointed.

## Commits

| SHA | Subject | Scope |
| --- | --- | --- |
| `d359b2a` | `fix(stylist): reconcile catalog identity, image admission and Edge feasibility` | catalog identity move, image URL repair/admission, Node ingestion mirrors, Edge retrieval + rotation feasibility, quiet-palette red gate, focused tests |
| `b3b453d` | `fix(stylist): port mobile totals, concept titles, tips and feedback guards` | screen card total, concept-title fallback, styling-tip dedupe, feedback gating, locales, focused tests |

Candidate code tip: `b3b453d`. This report is committed on top of it as a
documentation-only commit. The branch is a fast-forward over `962bd102`; no history was
rewritten and the reviewed branch and its commits are untouched locally.

## What was reconciled

### Catalog images (was: 961 Zara rows excluded)
`catalogImageAdmission` / `normalizeCatalogImageUrl` live in `src/domain/catalogIdentity.ts`
now (compat re-export kept at `src/utils/catalogIdentity.ts`). The verified production
construction `.jpg&w=1024` is repaired by restoring the `?` separator; only structurally
unusable URLs (blank, control characters, non-HTTP schemes, unrepaired glued query,
multiple query separators) are refused. The repair is mirrored for Node ingestion in
`scripts/lib/catalogImageUrl.js` and used by
`scripts/sync-product-only-downloader-feeds.js` and `scripts/sync-zara-men.js`, so the
Zara source keeps ~961 products instead of being dropped. The Edge candidate fetch
(`fetchShopItemsForAllSlotsEdge`) admits rows with the same helper and records
`imageAdmissionSkips` / `imageUrlsNormalized` counters in the trace. The catalog rows
already stored in production are not deleted; the existing verified backfill script
(`scripts/validation/backfill-zara-image-urls.mjs`, dry-run default) remains available
for a separate maintenance pass.

### Candidate feasibility (rotation starvation)
`rotationPool` now also returns the recently-shown products it pruned (`pruned`) while
fresh supply still met the target. In the Edge planner, if the fresh-only pool cannot
produce a complete deck, planning is retried once with the pruned rows appended and the
affected slots marked as forced repeats. The retry is a controlled feasibility
expansion: pruned rows already passed every hard gate, the audit records
`attempted/baseLooks/expandedLooks/prunedSlots/outcome`, and a rejected retry restores
the base maps, evaluation cache and gate counters so indices cannot leak. Deck-overlap
diagnostics (`previousDeckOverlapBySlot`, `shoeRotationAudit`) exclude exactly the
expanded rows that survived selection, so the rotation audit stays truthful.

### Visual/color evidence
The Edge `inferShopColor` now reads the merchant `color_name`/`color`/`colour` slug
before free-text descriptions (descriptions often name companion garments). Quiet
classic styles (`old_money`, `classic`, `business_casual`, `semi_classic`, `minimalist`)
reject vivid scarlet/electric red after pattern rejection while keeping muted reds
(burgundy/wine/maroon/oxblood) and respecting explicit favorite colors.

### Mobile client
- Card total row: shown only when every shop piece resolves a quote and all quotes share
  one currency; UZS conversion only with loaded reference rates, otherwise the native
  amount with its own currency label. Reuses the tile quote resolver.
- Concept title: server `concept.title` is kept when the concept id has no localized
  copy; only an unidentifiable look falls back to `Styled look`.
- Styling tips: restated/case-duplicate tips are suppressed and an empty server tip
  array no longer renders an empty tip card.
- Feedback controls: gated on the exact `stylist_look_feedback` constraints
  (`look_id ~ '^outfit_[1-3]$'`, concept slug, bounded engine version, request id) so the
  UI never offers a silently dropped write.
- `outfitMaker.outfitTotal` added to en/ru/uz.

## Validation

| Check | Result |
| --- | --- |
| Focused Jest (presentation, price total, feedback eligibility, admission, palette, rotation, shop color, hosted safety/recovery) | 10 suites / 80 tests passed |
| `npm run typecheck` (app) | clean |
| Edge typecheck (`deno check` with the repo lockfile, `--frozen=false`) | clean (all functions, including `generate-outfits`) |
| Consolidated backend Jest (`__tests__/backend`, once) | see table below |

### Backend suite — candidate vs reviewed PR tip

Run once on the candidate (`__tests__/backend`, 40 suites / 599 tests) and once on a
pristine detached worktree at `962bd102` with the same lockfile/dependency snapshot:

| Metric | PR tip `962bd102` | Candidate (`d359b2a`+`b3b453d`) |
| --- | --- | --- |
| Failed suites | 10 / 40 | 8 / 41 |
| Failed tests | 67 | 64 |
| Passed tests | 532 | 538 |
| Total tests | 599 | 602 |

All candidate failures are a subset of the failures already present at the reviewed PR
tip; no suite or test that passes at the tip fails on the candidate. The two earlier
candidate regressions (`stylist3Safety`, `fourSlotHosted`) were caused by the moved
catalog identity being imported without a `.ts` extension through the repo's custom
Edge loader and are fixed in `d359b2a`. The two hosted suites
(`stylingV193Hosted`, `stylingV194Hosted`) failed the baseline full run on timeouts but
passed the candidate full run and isolated reruns; treat them as timeout-sensitive.

### Known pre-existing failures (identical or worse at `962bd102`)

These are stale/rotten harnesses relative to the reviewed PR's RPC-first retrieval and
rebuilt planner. They are recorded, not hidden, and were not rewritten beyond import
shims:

- `outfitCandidateCurrency.test.js`, `generationCandidatePool.test.js` — harness mocks
  the removed `from()/pagination` retrieval; production now uses
  `catalog_stylist_candidates_compact` RPC. Import shims updated; contract rewrite not
  part of this reconciliation.
- `generateOutfitsFallback.test.js` — returns the removed `localFallback` symbol.
- `paidWorkCoverage.test.js`, `curatedSelectionQuality.test.js`,
  `stylistRepairPath.test.js` — assert older planner/provider contracts.
- `qualityFloorHosted.test.js`, `outfitBudgetWire.test.js` — long hosted simulations
  with timeout-sensitive runs.

## Blockers / follow-ups

1. **Edge deployment is not performed.** Production still runs Stylist v204. Deploying
   `generate-outfits` (candidate changes included) is a separate, explicitly approved
   step.
2. **`deno.lock` drift.** `npm run typecheck:edge` (frozen lock) fails identically at
   `962bd102` because a transitive `esm.sh/@supabase/supabase-js@2.95.3` fingerprint
   changed upstream. Type checking itself is clean with `--frozen=false`. The lockfile
   needs one reviewed refresh before CI can enforce the frozen check.
3. **Stale backend harnesses** listed above should be rewritten against the RPC-first
   retrieval contract in a follow-up (not a launch gate: production code is covered by
   the focused suites and the Deno check).
4. **Catalog backfill** of the stored `.jpg&w=` rows is optional; ingestion and Edge now
   repair at read time, so coverage is restored without a write.
5. **Production budget/feedback data** requires the existing reviewed migrations; no new
   migrations are introduced by this branch.

## Decision

**GO for review/push to `codex/stylist-5-integrated-20261108`** — the candidate contains
the preserved local fixes on top of the reviewed PR tip, passes app typecheck and Edge
typecheck, adds focused coverage for every behavior change, and introduces no new
backend-suite failures versus the reviewed tip.

**NO-GO for production deploy until:** the Edge function deploy is reviewed/approved,
the lockfile refresh is merged, and a post-deploy smoke of Stylist generation is run
against the pinned revision.
