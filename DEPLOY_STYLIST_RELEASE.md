# Stylist Integrated Release — production v206 (r2 release f7b4b54)

Current production status (2026-11-09): **Stylist 5 r2 DEPLOYED as Supabase Edge Function v206 — ACTIVE.**
Source commit: `f7b4b54909985d96d01809adb59cab5872201e93`.
Deployed generation identifier in source:
`stylist-5-editorial-v2-rc-20261109-r2`.
Deployment fingerprint:
`9bb8efb4580760691d9df7f95cdf3aa8f203e743604d369e5402eaee62ec650d`.
All **48 runtime source files** retrieved from the active Edge deployment match
the approved r2 source composition byte-for-byte (six new source files,
42 unchanged v205 runtime files). Type-only source files were submitted
for Deno dependency resolution but omitted from the retrieved runtime graph.
`verify_jwt=false` is preserved; bearer-token authentication remains in
the function's own code. **Authenticated end-to-end r2 generation and iOS
tap-through are still PENDING**, and must not be inferred from the ACTIVE
deployment state or source checks.

Previous production v205 is preserved for emergency redeployment in the
protected `public.stylist5_release_rollback` record
`stylist5-v205-backup-before-r2-20261109`. Backup contains the exact 48
deployed runtime files plus pinned compile-only `src/types/shop.ts`,
with verified manifest SHA-256
`d987b204e1b8bf42616737f61ce0462ed73c99d2aacc328170d9b25383c70a12`.
Metadata records original v205 fingerprint
`89fd3872ff0f9c29905e9938de69efd4b7458cdd56be8b52b8e2d09ad5cf1fc6`.
Older v204 backup remains available separately.

Historical v205 release details below are retained for traceability.

## 1. Preconditions

- Branch `codex/stylist-integrated-rebuild-20261109` at `f1d864e` (code
  release; pushed, remote SHA verified). No force-push; merge to the
  release branch via PR.
- Local gates green on that SHA:
  - `npx tsc --noEmit` (clean)
  - `deno check --frozen backend/supabase/functions/generate-outfits/index.ts` (clean)
  - `npx jest __tests__/services/stylistFashionRegression.test.ts`
    `__tests__/backend/generationVersion.test.js` (12/12)
  - Targeted concept-affected suites: 58/58
    (`outfitCandidateCurrency`, `generationCandidatePool`,
    `shopColorProvenance`, `stylist4`)
- iOS Release artifact: arm64-only verification build from the release tree
  succeeded, installed on simulator `AIWardrobe-RC-Local-20261103` as
  `com.aiwardrobe` 3.0.3 (66), launch + home render screenshotted
  (`/tmp/aiwardrobe-rc3-app.png`). Full dual-arch dSYM build is blocked
  by host disk (~1 GB free); run it on a host with headroom before TestFlight.

## 2. Production deployment — COMPLETE

The owner explicitly approved the production deployment on 2026-11-09.
Supabase MCP deployed only `generate-outfits` from pinned code release
`f1d864ec395a9361f03eba07e2422f4bbb4cf9a4`. The 51-file source graph
was complete; Supabase returned v205 ACTIVE. All 48 runtime files returned
by `get_edge_function` matched the pinned source byte-for-byte; three TS
files used only for types were excluded from the runtime bundle.
`verify_jwt=false` remains unchanged; the function still performs its own
bearer-token authentication.

Do not deploy again unless there is a new explicitly approved release.
The release's source is available at the pinned GitHub commit.

## 3. Post-deploy smoke — PENDING (authorized test account only)

Repeat the read-only generation probes (same canonical `catalogOnly` body
the client sends; each call consumes paid generation quota for the test user):

- `minimalist` + `work` → expect 3 looks, no loud/buffalo checks.
- `semi_classic` + `work` → expect 3 looks, no checked-on-checked.
- `old_money` + `date` → expect 3 looks, coherent register, no hooded layer.
- `minimalist` + `work` + `maxShopBudget {1500000 UZS}` →
  expect `no_outfit_within_budget` with `minimumShopSpend` (honest failure).
- Same + `{20000000 UZS}` → expect 3 looks, all server-side capped.
- Confirm every response reports the NEW `generationVersion`
  (anything other than `stylist-5-editorial-v2`).

Then run one in-app pass on the 3.0.3/66 build: budget control +
currency renders, totals match, images load, Buy opens real store URLs,
Save Outfit / Add to Calendar / thumbs feedback all persist.

## 4. Rollback

- Edge: the exact previous v204 bundled `index.ts` is saved in the protected
  Supabase table `public.stylist5_release_rollback` with release_id
  `stylist5-v204-backup-20261109`, 309104 source characters, SHA-256
  `ab3d9232541f8c355e2224a2fcb1119996fb4b59fb2cd1ab122d7fd73dc1ac69`.
  Retrieve the source via authorized SQL, verify its checksum, then use the
  authenticated Supabase Edge deployment tool with `entrypoint_path=index.ts`
  and `verify_jwt=false`. Verify the restored version and function health.
  This is an emergency-only recovery procedure; the target is v205.
- App: 3.0.3/66 is additive (dedicated budget control, response-integrity
  checks); older builds remain compatible with both edge versions because
  the request contract is unchanged.

## 5. Known remaining blockers (do not present as done)

- In-app tap-through of Generate/Buy/Save/Calendar/feedback was not
  possible in this environment (no scripted touch input: no `simctl` touch
  API, AppleScript GUI control TCC-denied). Manual pass required on the
  installed 3.0.3/66 build: open AI Stylist → New clothing →
  Minimalist + Work → budget/currency controls → Generate → images, totals,
  Buy (real store URL), Save Outfit, Add to Calendar, thumbs feedback;
  confirm loud checks rejected and concept names match garments.
- Production v206 is deployed and source-verified, but there is not yet an
  authenticated live generation response confirming r2 runtime behavior.
  Earlier live screenshots/logs exercised v205, not r2. Run bounded
  Minimalist+Work, Old Money+Work, Semi-Classic+Date Night and explicit
  budget scenarios using an authorized test account. Keep quota metering intact.
- 19 `outfitBudgetWire` Jest failures are pre-existing obsolete-harness
  debt (present at base `737da82`); modernize that harness separately.
- A full iOS Release archive was subsequently reported built successfully at
  `/tmp/AIWardrobe-RC.xcarchive` (208 MB); iOS app store signing/submission and
  direct in-app tap-through still need independent confirmation.

## 6. Stylist 5 r2 (`stylist-5-editorial-v2-rc-20261109-r2`) — DEPLOYED v206, RUNTIME SMOKE PENDING

Seven confirmed v205 product defects were addressed in this code release.
The owner gave new explicit GO for deploying r2. Supabase MCP deployed this
candidate as production v206, replacing v205 only. No changes were made to
other functions, migrations, RLS, authentication configuration, quotas,
retailer/catalogue records or subscriptions. The v205 source backup was
verified before deployment.

What changed (one coordinated candidate, no new ranking engine or provider):

- Ranking contract: a provider response with the right count but one
  repeated ID is safely repaired (valid unique IDs keep provider order,
  unknowns dropped, missing tail filled from deterministic shortlist order)
  with provenance `hybrid`; unrepairable responses stay a full
  `deterministic_fallback`. AI badge renders for `ai` only. Regression
  coverage: `__tests__/backend/stylistR2RankingRepair.test.js` (9/9).
- Fashion coherence: named knit motifs (geometric jacquard, fair isle,
  argyle, intarsia, brocade) are visible patterns while tonal/subtle knits
  stay textures; restrained styles reject competing upper patterns (soft
  penalty for formal dress pairings); formal tailoring over performance
  bottoms and very casual layers on polished date nights are rejected.
  Known-good tailoring stays accepted.
- Naming: evidence-grounded fallback variants (`grounded-knit-layers`,
  `grounded-tailoring`, `grounded-denim-ease`, `grounded-separates`) with
  full en/ru/uz titles; no color claims; deck-distinct localized titles.
- Explanations: relationship sentences name the actual garments plus a
  grounded second sentence (palette or silhouette); localized tips render
  when they add information and are always deduplicated against the
  description.
- Localization: the 26 Stylist keys missing in all locales were added
  (en/ru/uz); style names, occasion names, weather, headers, budget, result
  cards, actions, and error/empty states resolve in the active language;
  switching languages re-localizes stored results without a new request.
  The prompt placeholder no longer contains the old `1 500 000 UZS` budget
  example (budget has its own explicit control; no hidden default limit).
- Client/server contract: response `source` is now
  `ai | hybrid | deterministic_fallback`; per-look `selectionSource`
  matches; generation identifier is the new r2 value below.

Gates on this candidate: `npx tsc --noEmit` clean;
`npm run typecheck:edge` (all Edge functions) clean;
`__tests__/backend/stylistR2RankingRepair.test.js` 9/9,
`__tests__/services/stylistR2QualityFixes.test.ts` 31/31,
fashion neighbors (`stylistFashionRegression`, `stylistPresentation` — the
latter was already red at v205 base and is fixed here,
`stylingQualityRefinement`, `stylingV193`, `outfitCoherence`,
`formalityCoherence`, `nvidiaOutfitRequest`, `generationVersion`,
`generationCandidatePool`, `deckAssembly`, `curatedSelectionQuality`)
all green. The 19 `outfitBudgetWire` failures and 13 other failing
suites are byte-identical pre-existing failures on the v205 base
(verified by stash comparison).

Production deployment performed (2026-11-09):

- GitHub code SHA: `f7b4b54909985d96d01809adb59cab5872201e93`.
- Supabase function: `generate-outfits` in project `fyqpifmrsftsfqibhwhy`.
- Version: **v206 ACTIVE**; fingerprint:
  `9bb8efb4580760691d9df7f95cdf3aa8f203e743604d369e5402eaee62ec650d`.
- Exactly 51 source files provided (48 runtime + 3 type-only).
- All 48 source files returned from the active deployed function match the
  approved r2 source graph without mismatches.
- Deployment source `GENERATION_VERSION` is
  `stylist-5-editorial-v2-rc-20261109-r2`.
- The new NVIDIA helper contains `repairLookRanking`; correct provider
  behavior still requires an authenticated real request to verify.
- The complete previous v205 source is stored with checksum in protected
  backup record `stylist5-v205-backup-before-r2-20261109`.
  To recover, retrieve its JSON path-to-source map using authorized SQL,
  verify SHA-256, call the Supabase Edge deploy method with entrypoint
  `backend/supabase/functions/generate-outfits/index.ts` and
  `verify_jwt=false`, and confirm restoration using runtime results.
  The full source is maintained only for emergency rollback, not a target
  for regular deployments.
- **Do not repeat production deployment without new authorization.**
- **Remaining verification:** authenticated generation scenarios, provider
  provenance and hybrid behavior, budget validation, fresh r2 iOS build and
  manual in-app Generate/Buy/Save/Calendar/feedback tap-through.
