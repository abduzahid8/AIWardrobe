# AIWardrobe

A smart personal stylist for your phone. Snap each clothing item once,
and AIWardrobe generates weather- and occasion-appropriate outfits
from what you actually own — with a rule-based styling engine,
optional AI try-on, and a shoppable catalog of missing pieces.

React Native · Expo SDK 54 · TypeScript · Supabase · Zustand · React Query

---

## Table of contents

- [Core features](#core-features)
- [Tech stack](#tech-stack)
- [Project layout](#project-layout)
- [Getting started](#getting-started)
- [Configuration & secrets](#configuration--secrets)
- [Scripts](#scripts)
- [Testing](#testing)
- [Releasing](#releasing)
- [Further reading](#further-reading)

---

## Core features

- **Closet management.** Photograph a garment; the app crops the
  background, classifies it, extracts color/material/pattern, and stores
  it in your wardrobe. See `features/closet/`, `src/services/ai/scanService.ts`.
- **Outfit suggestions.** A deterministic, rule-based styling engine
  scores candidate outfits along four axes (formality, novelty, color
  harmony, weather) with a 5-tier formality ladder from black-tie to
  activewear. See `src/services/suggestionEngine.ts`.
- **Outfit calendar.** Plan outfits per day, track wear counts, and
  generate weekly previews.
- **AI try-on.** An optional 3D mannequin visualises the combined
  outfit before you commit (`features/try-on/`).
- **Shop integration.** When the engine suggests an outfit that's
  missing a piece, it surfaces shoppable items from curated catalogs
  (Zara, Uniqlo, Inspo).
- **Subscriptions.** Daily free quota + premium via RevenueCat.
  Entitlement verified against Supabase on every cold start.

## Tech stack

- **Mobile:** React Native 0.81 via Expo SDK 54, new architecture on,
  NativeWind, React Navigation 7, Reanimated 4, Gesture Handler.
- **State:** Zustand 5 (UI/local), React Query 5 (server cache).
- **Backend:** Supabase — Postgres, Auth, Storage, Edge Functions.
- **AI:** All provider calls (NVIDIA, Replicate, Hugging Face, Gemini)
  go through Edge Functions in `backend/supabase/functions/ai-process/`.
  No provider keys ship with the client.
- **Observability:** Sentry (crashes), a home-rolled analytics queue,
  RevenueCat for IAP.
- **Tooling:** Jest, Maestro, ESLint (strict: forbids client-side AI
  SDK imports and `console.*`), Prettier.

## Project layout

```
App.tsx                  Root provider chain (QueryClient, SafeArea, Error)
navigation/              React Navigation stacks + cold-start orchestration
screens/                 Screen-level views (thin; delegate to features)
features/                Self-contained feature modules (outfit-gen, try-on, calendar)
components/              Shared presentational components
hooks/                   Shared hooks
store/                   Zustand stores
src/
  services/              Domain services (AI, suggestion, API clients, IAP, analytics)
  services/ai/           Thin wrappers over the Supabase ai-process Edge Function
  utils/                 Logger, secure storage, validators, image helpers
  types/                 Canonical domain types — single source of truth
  lib/persistence.ts     Registry of every AsyncStorage key (used on logout)
scripts/                 Node scripts (shop sync, i18n/ maintenance) — not bundled into the app
docs/                    Engineering docs, reports/, admin/, product/, research/
__tests__/               Jest tests (incl. suggestionEngine golden cases)
ios/, android/           Native projects managed by expo prebuild

backend/                 Services independent of the RN app (own deploy lifecycle)
  api/                   Legacy Express API — deprecated, see docs/ARCHITECTURE.md ADR-001
  supabase/
    functions/           Edge Functions (ai-process, try-on, delete-account)
    migrations/          SQL migrations, incl. 007_rls_audit.sql drift guard
  mobile-vton-service/   Modal-deployed FastAPI virtual try-on GPU service
```

## Getting started

```bash
# 1. Install deps
npm install

# 2. Copy env template and fill in Supabase values
cp .env.example .env
$EDITOR .env  # only the EXPO_PUBLIC_SUPABASE_* values are required

# 3. Start Metro + dev client
npx expo start
```

The app will bail out gracefully with a `MissingConfigScreen` if
required env vars are absent. No silent black-screen crashes.

### Prerequisites

- Node 22 (matches `eas.json`).
- Xcode 15+ for iOS builds, Android Studio for Android.
- A Supabase project with the migrations in `backend/supabase/migrations/`
  applied. The latest migration (`007_rls_audit.sql`) will fail to
  apply if any public table lacks RLS — by design.

## Configuration & secrets

**Provider API keys never go in `.env`.** They live in Supabase
`app_config` and are read by Edge Functions. The only values you put
in `.env` are the ones with `EXPO_PUBLIC_` — those are deliberately
bundled into the IPA and are considered public.

See [`docs/SECURITY.md`](docs/SECURITY.md) for the full model, the
per-key ownership matrix, and the key rotation checklist.

## Scripts

```bash
npm run lint        # ESLint; no-console and no-explicit-any ratchets
npm run typecheck   # tsc --noEmit
npm test            # Jest
npm run format      # Prettier

# Content sync (Node, uses Apify — needs scripts/.env.local)
npm run shop:sync-men

# EAS
npm run build:ios
npm run build:android
```

## Testing

- **Jest** for unit tests. The golden-case suite for the styling
  engine is at `__tests__/services/suggestionEngine.test.ts`. Keep
  these green: the engine is product-critical.
- **Maestro** flows live in `.maestro/` and exercise the main closet
  path end-to-end on a real device / simulator.
- See [`docs/NATIVE_TESTING.md`](docs/NATIVE_TESTING.md) for native
  testing setup.

## Releasing

The `eas.json` file defines four channels:

- `development` — internal distribution, dev client.
- `preview` — internal distribution, release build.
- `apk`  — Android APK for side-loading.
- `production` — App Store / Play Store store-bound builds.

Before submitting to the App Store:

1. Verify `expo-apple-authentication` is installed and that Sign in
   with Apple appears in the login screen (Apple Guideline 4.8).
2. Check `ios/wardrobeapp/PrivacyInfo.xcprivacy` still reflects the
   SDKs you actually ship.
3. Run `npm test && npm run typecheck && npm run lint`.
4. `npx eas build --platform ios --profile production`.

## Further reading

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module boundaries,
  data flow, Supabase schema.
- [`docs/SECURITY.md`](docs/SECURITY.md) — key management + RLS.
- [`docs/SERVERLESS.md`](docs/SERVERLESS.md) — Edge Function contracts.
- [`docs/INTEGRATION_GUIDE.md`](docs/INTEGRATION_GUIDE.md) — wiring up
  external providers.
- [`docs/NATIVE_TESTING.md`](docs/NATIVE_TESTING.md) — Maestro + device
  testing.
