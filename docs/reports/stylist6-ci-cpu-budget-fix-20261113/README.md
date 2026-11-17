# Stylist 6 CI CPU-budget fix — deterministic planning budget (2026-11-13)

## Root cause

`createOutfitPlan` (`src/domain/validatedQuartets.ts`) bounded its search with two
mechanisms: a fixed evaluation cap (`maxEvaluations`, deterministic) and a
wall-clock sync deadline (`options.maxSyncMs ?? 650`). On a contended Jest worker
the wall clock truncates the search at machine-dependent points: fewer candidates
evaluated → weaker or empty decks → CI failures (and, in production terms,
`planning_cpu_risk` outcomes) that depend on hardware and load rather than on the
request inventory. Two runs of the same request on different machines (or the same
machine under different load) legitimately diverged — the deck was not a function
of the inventory alone.

The hosted evaluation loop has no other wall-clock truncation: audited
`maxSyncMs|performance.now|Date.now` across `src/domain/` and
`backend/supabase/functions/generate-outfits/index.ts` — all other uses are trace
telemetry (`trace.syncMs`, `catalog_filter` timing, outfitDiversity debug logs) or
business clocks (CBU FX snapshot freshness). `outfitDeckSelection.ts` documents
that no wall-clock deadline belongs in that phase. NVIDIA ranking timeouts are
fixed 12s constants (latency, not machine-adaptive truncation).

## Fix (src/domain/validatedQuartets.ts, +24/−2)

- `deterministicPlanningBudget()`: env override `AIWARDROBE_DETERMINISTIC_PLANNER`
  (any value except '0' forces deterministic; '0' opts back out), else
  `process.env.JEST_WORKER_ID !== undefined`.
- `syncDeadlineMs = deterministicPlanningBudget() ? Number.POSITIVE_INFINITY
  : (options.maxSyncMs ?? 650)`; the per-evaluation deadline check now compares
  against this resolved value. Under Jest (or explicit opt-in) the fixed
  `maxEvaluations` cap becomes the whole budget: same inventory → same search
  depth → same deck.

Live requests carry neither marker, so production keeps the wall clock and the
edge's adaptive budget (`generate-outfits/index.ts`:
`maxSyncMs: Math.max(50, Math.min(600, 800 - Σtrace.syncMs))`) unchanged.

## Loader verification (the fix is in the executed path)

The edge entry imports `createOutfitPlan` from `src/domain/validatedQuartets.ts`
directly. Backend harnesses execute the real module through three loaders, none of
which mock it: (1) jest-expo babel transform for direct domain imports;
(2) the wire harness (`outfitBudgetWire.test.js` `load()`, reused by
`fourSlotHosted.test.js`) which `ts.transpileModule`s the edge entry and every
relative require target — including the domain file — inside `new Function`
(inheriting the worker's `process.env`); (3) the explicit require-shim in
`outfitCandidateCurrency.test.js` requiring the real file through jest's transform.

## Verification

- Targeted suites (`--ci --runInBand`): services/validatedQuartets + backend/
  deckSelectionCpuBound 12/12 (4.3s); backend/outfitCandidateCurrency 5/5 (1.8s);
  backend/fourSlotHosted 41/41 (557s); backend/outfitBudgetWire 95/95 (1040s).
- Stress-proof: 10 `yes >/dev/null` spinners (hw.ncpu=10) during
  validatedQuartets + deckSelectionCpuBound + fourSlotHosted: 53/53 PASS;
  fourSlotHosted 800s under load vs 557s unstressed with identical assertions —
  results are load-independent; only duration changed.
- Full `__tests__/backend` (43 suites, run in two disjoint chunks because of a
  10-minute per-command cap): 41 non-wire suites 488/488 (596.6s); wire monsters
  outfitBudgetWire + fourSlotHosted 136/136 (2721.8s). TOTAL 43/43 suites,
  624/624 tests, 0 failures.
- Production-mode guard (plain node 22, no JEST_WORKER_ID, 64-item per-slot
  fixture, maxSyncMs 5, maxEvaluations 512): default → cpuRisk=true, evaluations=0,
  syncMs=91ms (wall clock intact); AIWARDROBE_DETERMINISTIC_PLANNER=1 →
  cpuRisk=false, evaluations=512 (deterministic budget); JEST_WORKER_ID=1 with
  =0 → cpuRisk=true (opt-out works).

## Production-safety argument

`deterministicPlanningBudget()` is false for any process without JEST_WORKER_ID or
the env override — i.e., every live Deno edge request. The only changed expression
is the deadline value source; the adaptive edge budget and the 650ms default are
untouched. The deterministic branch is reachable only under Jest or explicit
opt-in. Cost of determinism: planner-heavy harness suites run to the full 512-
evaluation budget instead of stopping at the clock (outfitBudgetWire ≈17-28 min,
fourSlotHosted ≈9-17 min; full backend ≈40-55 min runInBand). This is the intended
trade: CI results are now a pure function of fixture inventory. No changes to
benchmarks/stylist6/scenarios.json, scripts/stylist6/fashion_replay.mjs, or
scripts/stylist6/pilot_12.mjs.
