# CPU-bounded v3 — prepared, not deployed

Correction source: `7154dfa932c5ab6e71ed796cf303c60151b97a85`, based on published P1/P2 `74e41c5c5ff72fd5f0f4755bc2a249e81aa6cf07`.

Production baseline is the user's confirmed v185 rollback, bundle `ff4b5f02772d999c3ec57ccb973599ec7fbe75044a9e0f7e4c30ced3ba3570df`. No production API calls or deployments were performed here.

## Measured cause

The post-provider handler unconditionally expanded a validated three-look deck with `completeChoices()`. Its quartet enumerator could try 6,000 combinations and score whole outfits before rejecting their shopping spend. Successful reserves were then scored/ranked again by the diversity selector. A shoe-audit field also called `completeChoices()` solely to populate diagnostics, defeating lazy expansion. Diversity debug logging ran inside the combination loop, including repeated anchor-allowance messages.

An offline replay executes the actual handler suffix from enrichment through its real Response, plus its actual complete-choice validator and domain imports. Every fixture passes the provider's style/compatibility validators 3/3 before timing. The 96-row budget-pressure Work and anchored Formal fixtures reproduced 4,025.871 ms and 3,937.167 ms of process CPU. `validatedQuartets` consumed 3,898.997 ms and 3,876.024 ms respectively, with over 6,000 style-judge calls. Ordinary anchored selection emitted 6,107 log calls. These are measured source-path reproductions with representative fixtures, not the unavailable raw production request payloads or hosted CPU profiles.

## Correction

- Accept a complete fresh provider deck without enumerating alternatives. Expand only a conflicting, short, or recycling deck; the audit reads only an already-computed cache.
- Before expensive styling validation, apply the same budget, source, and recent-product constraints. Cache complete-choice verdicts by immutable row references.
- Cap Edge reserve enumeration at 384 attempts, 24 admitted looks, and 12 shortlisted rows per slot. Stop as soon as the admitted reserve is full. The client/default quartet reserve behavior is unchanged.
- Give each Edge diversity selection a shared 512-step budget. Cache normalized product/strict identities, rotation objectives, quality/recency signals, and deck scores. An exhausted search never relaxes hard eligibility or identity rules; it may return the existing incomplete-deck error.
- Remove production per-candidate/combination debug logging and large product/history log dumps. Keep one compact selection summary. Shoe assignment retains its existing bounded search and full caller validation.

## Focused verification

- Five focused suites, **62 passed**: `fourSlotHosted`, `deckSelectionCpuBound`, `validatedQuartets`, `shoeRotation`, and `formalityCoherence`. Hosted-handler coverage includes budget/source isolation, partial provider decks, physical aliases, worn items, five-generation recency, all four anchor slots, and minimum shoe overlap.
- Nine post-provider replays, all HTTP 200 with three four-slot looks, under a **500 ms process CPU ceiling**. Work/Formal budget-pressure cases fell to **41.655 / 41.956 ms**; worst total across cases was **126.480 ms**. Fresh valid decks made **zero quartet-enumerator calls**, including audit construction. Logging fell to one summary, or two including the existing anchor-enforcement message.
- CPU timing excludes module transpilation and provider/network latency; nested stage timings overlap and must not be summed. Fixtures are deterministic representative catalog pools with prices, metadata, history, anchors, and intentional provider product conflicts.
- Run: `CPU_BUDGET_MS=500 node scripts/validation/post-validation-cpu.cjs`. Pass a worktree path as the first argument to replay the baseline without enforcing the corrected budget. Raw before/after evidence is alongside this file.
- The compiled release bundle passes offline OPTIONS 200, GET 405, and unauthenticated POST 401 checks with zero network attempts. The handler prefix through request admission/pool preparation matches the baseline byte-for-byte; its hash and all 37 build-input hashes are recorded in `manifest.json`.

## Exact artifact

`aiwardrobe-generate-outfits-v3-cpu-bounded-20261106.tar.gz`

- Archive SHA-256: `8297382f0edf3b3d6f3873e33d5a3c7907f50d782f1863c91a7ebf6a2f40ba11`.
- Bundle SHA-256: `d232008818f59220475d9f04da92bfc243077fdb2bc826dffc62ecce38401637`, 357,072 bytes; esbuild 0.28.2.
- Generation marker: `stylist-runtime-repair-v3-cpu-bounded`. This is a release label, not an assigned hosted function version.

Deployment requires separate explicit authorization, verification of the then-current production baseline, and preservation of `verify_jwt: false`. No main merge, subscription, billing, database, or secret changes were made. The original P1/P2 release archive was not replaced or modified.
