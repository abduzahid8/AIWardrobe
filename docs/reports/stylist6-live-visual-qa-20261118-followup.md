# Stylist 6 — 18 November 2026: actual native fashion-quality canary and photo editor follow-up

**Release verdict: Stylist 6 is not yet an excellent, validated personal stylist. PRODUCTION NO-GO.**
This checkpoint is an exact hosted compute/iOS QA record, not a simulated aesthetic win.
Keep production Stylist 5, authentication, billing, merchant rights, rollback and the original independent full-photo veto unchanged.

## Implemented in feature PR #15

- `d5611542`: the independent second original-SKU fashion creator sees bounded, source-current photographed garment observations and separately labeled merchant text clues for the first creator's already-proposed outfit architecture, not just subtype names. Its one malformed SKU no longer discards other individually valid real four-item compositions; compatible first-studio remainder can fill unused photo-audition positions. Exact source identity, across-look nonreuse, weather/budget/merchant and independent original-photo checks remain authoritative.
- `dc3d2e7e`, `2d44a18c`: regression test and syntax correction for genuine source-bound partial salvage, 4 actual products per look, separate slot arrays, privacy, and unchanged final hard-gate replay.
- `92c3a37c`: allow the separately trained, real-photo Kimi F-token fashion jury to review a two-look-only independently photo-approved menu too. Before this, even a ready JPEG photo board with two valid looks had only a slow 120B photo editor; if it timed out, the final selection reverted to text-only. The second jury is staged only for existing >=72s server-owned visual-review flow, can only select a previously photo-verified portfolio, cannot add any SKU or override the original critic, and respects the original request deadline.

## Real authenticated native iOS QA: deployed `2d44a18c` on Render

Staging Render service `srv-dbtd336i0phs73fd0sbg`, deploy `dep-dbul9mafngtc73d4sbvg` **LIVE** from exact source SHA `2d44a18c9081eddaa55d5b46b656d09fa97e5fb0`; Supabase staging JWT/HMAC gateway unchanged. Booted real iOS 26.5 Simulator `AIWardrobe-S6-QA-20261115`, app `com.aiwardrobe`, actual Maestro flows complete with exit 0. Results below are structured Render production-style quality receipts, not mocked evaluations.

| Real scenario | UTC result | Source AI proposal | Independent original-photo review | Displayed | Editor outcome | Wall compute |
| --- | --- | --- | --- | --- | --- | --- |
| Classic / New / Date night | 07:35:45 | Nemotron: 4 proposed; 3 source-admitted | 2 approvals / 3 attempts; third `model_veto` | **2** | text fallback applied; photo contact sheet `ready`, primary vision jury `timeout`, independent visual jury `not_run`, eligible triples 0 | **33,872 ms** |
| Classic / New / Work | 07:39:05 | Nemotron + independent Kimi: 8 proposed | 3 approvals / 11 attempts across waves; Kimi photo-critic revision 1/3 approved | **2** | actual photo jury `applied`, second visual jury `unavailable`; one feasible 3-look portfolio existed, final director still chose 2 | **40,935 ms** |

Both showed 4 real source items per published look (`catalogFourItemContract=true`, `catalogFourItemDropped=0`), with photo proof and no fake third look. The Work initial two-studio path actually ran. The Date initial two-studio enhancement did not run in that particular model result. **These are not proof that the new partial-salvage branch improved subjective fashion**: unanchored live catalogue windows vary between native trials.

Direct visual inspection of the Work iOS cards showed a coherent but insufficiently varied tailoring capsule: **"Refined Tailoring"** with white shirt, navy blazer, tailored dark trousers and dark shoes; **"Blazer & Tailored Trousers"** with pale-blue shirt, charcoal blazer, dark trousers and dress shoes. Two different SKU sets; still essentially the same blazer/shirt/trouser architecture. This FAILS the owner's demand for three genuinely different, compelling directions. Screenshots remain private on macOS; no unlicensed source product photos or QA identifiers are committed.

### Engineering verification

- `node --import tsx --test backend/stylist6/*.test.mjs`: **542/542 PASS** after the two-look visual jury correction.
- `npm run build --prefix backend/stylist-node && npm test --prefix backend/stylist-node`: **PASS; 42/42** server tests; TypeScript verification PASS.
- Stylist 5 bundle authentication/quota source SHA-256 still `f90ced33ca3b0034c61cd2dfa289b189728dc75398cd72329f0b7c1854efc5e1`.
- No new original-image hosting or external AI inference rights granted. Live Supabase snapshot: 8,554 active catalog items, 7,556 visual fashion profiles, 7,550 succeeded visual jobs, 999 `unknown`, zero READY licensed photo-cache rows.
- The rights-validated first-pass photo composer remains OFF (`skuCompositionPhotoFirst=not_run`) even though the independent per-look photo critic and final authorized photo jury run. Do not confuse processed SigLIP2 garment features with legal photo use rights or full-photo visual-first generation.

## Exact post-fix deployment and REAL native Date-night verification

- Render deployment **`dep-dbuldc2jnfac73djv2dg` LIVE**, exact source SHA **`92c3a37c007d9d0b804bcf3ca4c1ea57f3fd0cc0`**.
- Authenticated booted iOS 26.5 `Classic / New / Date night` completed successfully using the same isolated Supabase/Render chain, one Generate action; Maestro exit 0. Render `stylist6_safe_visual_quality` around **2026-11-18 07:43:37 UTC**, compute elapsed **55,019 ms**, delivered wall **55,549 ms**.
- Real source AI **8 proposals**, initial source fashion studio `nemotron_plus_kimi_initial`; 9 original-photo critiques across waves, **2 photo-approved**, **2 shown**, no source-invalid or synthetic fourth item (`catalogFourItemContract=true`, `catalogFourItemDropped=0`). Photo-critic revision tried 2 replacements and approved 0 (artist recorded `nemotron_fallback`). Reasons included `style_mismatch`, `below_quality_floor` and `model_veto`.
- **Crucial correction confirmed:** photograph board **`ready`**, primary vision editor **`applied`**, independent different-lab true-photo jury **`agreed`** with vote size 2, original photo-verified two-look portfolio chosen. `photoEditorialEligibleTriples=0`, so the previous gate would have skipped Kimi completely; this run **did not**. The correction has a proven hosted execution effect, NOT a proven fashion win.
- The actual Date-night iOS card was inspected: fine dark knit with warm brown tailored layer, dark trousers and matching dark footwear; again a conventional polished tailoring direction, not evidence of three independent outstanding styles. No merchant pictures copied into the report.
- **Interpretation:** better final real-photo editorial coverage is now demonstrated, but 2/9 individual approvals still expose initial creative-composition failure. The live random source inventory is NOT frozen across old/new runs, so do not claim statistical or causal fashion improvement. Production remains **NO-GO**.

## Required next evidence (not a success claim)

1. **Done:** deploy and authenticated photo-jury Date-night canary on `92c3a37c` verified the independent true-photo verdict for two approved looks. Next, make the first creative fashion decisions stronger on the same frozen real SKU catalogue; visually judge both individually and as a capsule, and retain photo failures. An image-model `applied` status alone is not human fashion approval.
2. Improve actual FIRST-PASS fashion decisions with rights-cleared exact SKU images where commercially and legally authorized. Hard source menus that ban a frequently used subtype had a live counterexample of 0/7 photographic approvals; do not repeat that tactic or lower the visual threshold.
3. Run a *fixed same-context, same-item* held-out visual benchmark and >=30 independent human blind Stylist 6 vs 5 comparisons with >=3 reviewers, >=70% wins. Without that, there is no basis to merge or move production.
