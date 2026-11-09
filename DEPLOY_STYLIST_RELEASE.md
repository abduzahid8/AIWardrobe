# Stylist Integrated Release — Deployment Instructions (commit c55a54f)

Status: release candidate VERIFIED locally, NOT deployed. Production Edge
Function `generate-outfits` is still the previous version (`v204`,
`generationVersion: stylist-5-editorial-v2`). No DB migration is required
by this release (no new migrations included).

## 1. Preconditions

- Branch `codex/stylist-integrated-rebuild-20261109` at `c55a54f` (pushed,
  remote SHA verified). No force-push; merge to the release branch via PR.
- Local gates green on that SHA:
  - `npx tsc --noEmit`
  - `deno check --frozen backend/supabase/functions/generate-outfits/index.ts`
  - `npx jest __tests__/services/stylistFashionRegression.test.ts` (9/9)
- iOS Release artifact: arm64-only verification build succeeded, installed
  on simulator `AIWardrobe-RC-Local-20261103` as `com.aiwardrobe` 3.0.3 (66),
  launch + home render screenshotted. Full dual-arch dSYM build is blocked
  by host disk (~1 GB free); run it on a host with headroom before TestFlight.

## 2. Deploy the Edge Function (explicit authorization required)

From the release SHA, with the production Supabase credentials:

```bash
supabase functions deploy generate-outfits --project-ref fyqpifmrsftsfqibhwhy
```

The bundle picks up the fashion fixes automatically — the function imports
`src/domain/garmentDescriptor.ts` and `src/domain/outfitConcepts.ts`
directly, and `src/domain/occasionRules.ts` (+ `outfitCoherence.ts`,
`layerCompatibility.ts`) via the `backend/supabase/functions/_shared`
shims. No other function changes in this release.

## 3. Post-deploy smoke (production, authorized test account only)

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

- Edge: redeploy the previously captured v204 bundle (kept intact — this
  release never overwrote production). Rollback is a single function
  redeploy; no data migration to reverse.
- App: 3.0.3/66 is additive (dedicated budget control, response-integrity
  checks); older builds remain compatible with both edge versions because
  the request contract is unchanged.

## 5. Known remaining blockers (do not present as done)

- In-app tap-through of Generate/Buy/Save/Calendar/feedback was not
  possible in this environment (no scripted touch input); needs a manual
  pass on the installed build.
- 19 `outfitBudgetWire` Jest failures are pre-existing obsolete-harness
  debt (present at base `737da82`); modernize that harness separately.
- Full dual-arch dSYM Release build needs a host with more free disk.
