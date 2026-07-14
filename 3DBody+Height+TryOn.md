# 3D Body + Height + Try-On — Active Plan

> **Note:** The original detailed plan lives in
> `docs/AIWARDROBE_6_MONTH_BODY_FIT_VTON_PLAN.md` (889 lines, 6 months).
> This file is the **live working log** — Month 1 is shipped, Months 2-6 are
> not yet started. Update this as we progress.

---

## Month 1 — Foundation, Data Models, and MVP Body Profile ✅ SHIPPED

**Goal:** Build the body-fit foundation without depending on SAM 3D Body yet.
The app should store body profiles, garment physical parameters, and produce
basic fit assessments from manual measurements.

**Status:** All Month 1 acceptance criteria met. 17/17 fit engine golden tests
pass. TypeScript clean on all new files. Supabase migration ready to apply.

### Acceptance criteria — verified

| Criterion | Status | Evidence |
|---|---|---|
| User can create / edit a body profile | ✅ | `BodyProfileScreen.tsx` |
| User can select a garment size | ✅ | `availableSizes` on `ShopCatalogItem` |
| App can show a basic fit result **before** render | ✅ | `assessFit()` in `src/lib/fit/fitEngine.ts` |
| Existing try-on still works | ✅ | All changes additive; no field removal |
| No SAM dependency yet | ✅ | `mesh` field declared but unused |

### Files added / changed

#### New (TypeScript)
- `src/types/bodyProfile.ts` — `BodyProfile`, `BodyMeasurement`, `BodyTypeId`, `GenderOption`, source enum
- `src/types/garment.ts` — `GarmentPhysicalProfile`, `FitIntent`, `Stretch`, `SEED_GARMENT_PHYSICAL_PROFILES` (7 seeded garments)
- `src/types/fitAssessment.ts` — `FitAssessment`, `FitZoneAssessment`, `OverallFit`, `ZoneStatus`
- `src/lib/fit/fitEngine.ts` — Pure `assessFit()` + `recommendSize()`, category-aware rules (top/pants/jacket/shoes/dress), fit-intent aware, stretch-tolerant, confidence scoring, size recommendation
- `store/bodyProfileStore.ts` — Zustand store with AsyncStorage persistence; CRUD; legacy `avatarStore` shim
- `screens/BodyProfileScreen.tsx` — Full manual entry flow: height, weight, body type, gender, 7 optional measurements, privacy toggles, server sync best-effort
- `__tests__/lib/fit/fitEngine.test.ts` — 17 golden tests covering tops / pants / jacket / shoes / edge cases / size recommendation

#### New (Backend)
- `api/routes/bodyProfiles.js` — POST/GET/PATCH/DELETE `/body-profiles` + `/me` + `/:id/activate` (Express + Supabase)
- `api/routes/fit.js` — POST `/api/fit/assess` + `/api/fit/recommend` (wraps engine for server-side callers)
- `api/services/fitEngine.js` — Server-side JS port of the engine (Express is plain ESM; engine lives in TS on the mobile side — KEEP IN SYNC)
- `api/services/garmentSeed.js` — Server-side mirror of the seed catalog
- `supabase/migrations/022_body_profiles.sql` — `body_profiles` table with one-active-per-user partial unique index, RLS, updated_at trigger, JSONB columns for forward-compat

#### Modified
- `api/index.js` — Registered `bodyProfileRoutes` and `fitRoutes`
- `api/services/mobileVtonClient.js` — Forwards optional `bodyProfile` / `fitAssessment` / `fitAssessments` to Mobile-VTON
- `api/services/strategies/mobileVton.js` — Pulls body context from request params and forwards
- `mobile-vton-service/main.py` — Pydantic models accept `body_profile` / `fit_assessment` / `fit_assessments`; handlers log the context; response includes `body_profile_received` / `fit_assessment_received` booleans
- `src/lib/persistence.ts` — Added `body-profile-storage-v1` to the wipe list; `clearAllPersistedUserData` now resets the body profile store on logout / account deletion
- `features/try-on/types.ts` — `ShopCatalogItem` extended with `physicalProfiles?`, `defaultSize?`, `availableSizes?`
- `navigation/types.ts` + `navigation/RootNavigator.tsx` — `BodyProfile` route registered
- `i18n/locales/{en,ru,uz}.json` — Added `bodyProfile` namespace (43 keys, all three languages)

#### Helper / scripts
- `scripts/i18n_add_body_profile.py` — Idempotent script that adds the `bodyProfile` namespace to all three locales (re-runnable; merges)

### What to apply / run before deploy

```bash
# Apply the new Supabase migration
psql "$SUPABASE_DB_URL" -f supabase/migrations/022_body_profiles.sql

# Re-run fit engine tests in CI
npx jest __tests__/lib/fit/fitEngine.test.ts

# Re-deploy the API + mobile-vton-service
# (no env var changes; new fields are all optional and additive)
```

### What does NOT work yet (deferred to Months 2+)

- ❌ SAM 3D Body photo analysis (Month 3) — `mesh` field exists, no service
- ❌ Multi-garment layering-aware fit (Month 5) — current engine is per-garment
- ❌ Real size-chart ingestion (Month 5) — only 7 seed garments today
- ❌ Body-profile preview screen showing the personalized mannequin (Month 2)
- ❌ Fit panel in `AITryOnScreen` (Month 2)
- ❌ Admin UI to add garment physical profiles (Month 5)

---

## Month 2 — Personalized Mannequin and Fit-Aware Try-On Payloads (next)

Tasks for next session:

1. Connect `bodyProfileStore` to `mannequin3D.ts` — use the active profile's
   height / weight / body type to drive the WebView mannequin. Today the
   AITryOnScreen pulls from `useAvatarStore` (legacy); replace with the new
   store while keeping a compat shim.
2. Add a body-profile preview UI so the user can see the mannequin reflect
   their current profile before going into try-on.
3. Add a Fit Panel in `AITryOnScreen` (overall badge + zone list + size
   recommendation). Render `FitAssessment` from the engine.
4. Update the API contract to ensure `body_profile` / `fit_assessment` flow
   end-to-end in the multi-garment path (the field plumbing is in, but the
   UI doesn't read it back yet).
5. Hook the `BodyProfile` screen into the Profile tab (one row, opens it).
6. Run mobile device test: create profile → pick garment → see fit panel.

---

## Months 3-6 — Reference

See `docs/AIWARDROBE_6_MONTH_BODY_FIT_VTON_PLAN.md` for the full plan.
The Month 3 tech stack is now confirmed:
- **SAM 3D Body**: https://github.com/facebookresearch/sam-3d-body.git
- Hugging Face access / SAM license TBD before service build-out
- Service runs as a separate Python service (`sam-3d-body-service/`) parallel
  to `mobile-vton-service/`
