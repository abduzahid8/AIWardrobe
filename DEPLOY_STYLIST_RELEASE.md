# Stylist Integrated Release — Deployment Instructions (code release f1d864e)

Status (2026-11-09): DEPLOYED to Supabase production as Edge Function v205,
verified ACTIVE. New code reports `generationVersion:
stylist-5-editorial-v2-rc-20261109` in the deployed entrypoint.
Authenticated end-to-end generation smoke tests are still PENDING; source
identity and deployment status alone do not establish runtime quality.
Production deployment fingerprint:
`89fd3872ff0f9c29905e9938de69efd4b7458cdd56be8b52b8e2d09ad5cf1fc6`.
No DB migration or RLS/auth/subscription change was applied as part of the
release. The former v204 Edge source is backed up for emergency recovery.

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
- The production v205 source contains the new version identifier and matches
  the approved GitHub release; an authenticated live generation response from
  v205 has not yet been observed. Older v204 budgeted + unbudgeted calls do
  not verify v205 fashion output. Run the bounded cases above with an
  authorized paid test account; keep usage metering intact.
- 19 `outfitBudgetWire` Jest failures are pre-existing obsolete-harness
  debt (present at base `737da82`); modernize that harness separately.
- A full iOS Release archive was subsequently reported built successfully at
  `/tmp/AIWardrobe-RC.xcarchive` (208 MB); iOS app store signing/submission and
  direct in-app tap-through still need independent confirmation.
