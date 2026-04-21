# Security Model

## Golden rule

Provider API keys (NVIDIA, Replicate, Hugging Face, Gemini, OpenAI,
FASHN, Apify, etc.) **never live in the mobile app**. They live in
the Supabase `app_config` table and are used only by Edge Functions
running on the server side.

Anything prefixed `EXPO_PUBLIC_` in `.env` is bundled into the IPA/APK
and is visible to anyone who downloads the app. Treat those values as
public, always.

## What belongs where

| Value                                    | Where              | Notes |
|------------------------------------------|--------------------|-------|
| `EXPO_PUBLIC_SUPABASE_URL`               | Mobile app `.env`  | Public by design. |
| `EXPO_PUBLIC_SUPABASE_ANON_KEY`          | Mobile app `.env`  | Public. Security comes from RLS. |
| `EXPO_PUBLIC_SENTRY_DSN`                 | Mobile app `.env`  | DSNs are public. |
| `EXPO_PUBLIC_REVENUECAT_API_KEY`         | Mobile app `.env`  | Public SDK key. |
| `EXPO_PUBLIC_WEATHER_API_KEY`            | Mobile app `.env`  | Public if used on-device. Prefer Edge Function. |
| `EXPO_PUBLIC_AI_VISION_PROVIDER`         | Mobile app `.env`  | Just a toggle string. |
| `nvidia_token`, `replicate_token`, etc.  | Supabase `app_config` | **Never in the app.** |
| `SUPABASE_SERVICE_KEY`                   | Edge Functions env | **Never in the app.** |
| `APIFY_API_TOKEN`                        | `scripts/.env.local` | Node scripts only, never bundled. |

## Adding a new provider

1. Add the token to the Supabase `app_config` table:
   ```sql
   INSERT INTO app_config (key, value) VALUES ('new_provider_token', '...')
     ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
   ```
2. Read it inside an Edge Function (`supabase/functions/*`).
3. In the mobile app, call the Edge Function via
   `supabase.functions.invoke('ai-process', ...)`.
4. **Never** add a new `EXPO_PUBLIC_*_TOKEN` for a provider.

ESLint enforces rule 3: `@google/generative-ai`, `@huggingface/inference`,
and `replicate` are forbidden imports in the mobile app.

## Key rotation checklist

1. Rotate the key at the provider dashboard.
2. Update `app_config` with the new value.
3. If the key was ever in `EXPO_PUBLIC_*`, rotate ALL historical builds
   by releasing a new version and treating all prior TestFlight / App
   Store builds as compromised.
4. `git log -p -- .env` is useless when the file is gitignored, but
   `git log -p -- .env.example` may surface leaked patterns. Scan
   history with `git log -p` | `grep -iE 'sk-|nvapi-|r8_|hf_|apify_'`.

## Row-Level Security

Every table in `public` has RLS enabled. The audit is automated: see
`supabase/migrations/007_rls_audit.sql` — it will refuse to apply if any
public table lacks RLS or if a per-user table has RLS enabled but no
policies.

## App-level protections

- Auth session persisted via `expo-secure-store` (Keychain / Keystore),
  not `AsyncStorage`.
- `ErrorBoundary` wraps the root; crashes are reported via Sentry when
  `EXPO_PUBLIC_SENTRY_DSN` is set.
- Production builds strip `console.*` via `babel-plugin-transform-remove-console`.
- Logger at `src/utils/logger.ts` redacts bearer tokens, JWT-looking
  strings, and known provider-token prefixes before printing.
- `clearAllPersistedUserData()` wipes every known store key on logout
  and account deletion.

## App Store Connect

- **Sign in with Apple** is wired via `expo-apple-authentication`.
  Required by Apple Guideline 4.8 whenever Google Sign-In is offered.
- `ITSAppUsesNonExemptEncryption=false` set in `app.json`.
- `PrivacyInfo.xcprivacy` shipped (`ios/wardrobeapp/PrivacyInfo.xcprivacy`).
- Usage descriptions are specific, per Apple review guidelines.
- `NSUserTrackingUsageDescription` declared — only triggered if a future
  SDK starts tracking.
