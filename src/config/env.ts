/**
 * Centralized environment configuration.
 *
 * All env vars must be prefixed with EXPO_PUBLIC_ to be available in the
 * production JS bundle.
 *
 * CRITICAL: You MUST reference them with a static dotted access
 * (`process.env.EXPO_PUBLIC_FOO`) so Expo/Metro can replace them with the
 * literal value at build time. Dynamic lookups like `process.env[key]` are
 * NOT inlined — they return `undefined` in release builds and break the
 * app on TestFlight / App Store with a "Configuration Error" screen.
 *
 * Never hardcode secrets — add them to .env (gitignored) and .env.example.
 *
 * IMPORTANT: This module must NEVER throw at import time. Throwing during
 * module evaluation crashes the app before React renders, bypassing
 * ErrorBoundary and producing an unrecoverable black screen.
 */

const str = (v: string | undefined, fallback = ''): string =>
  typeof v === 'string' && v.length > 0 ? v : fallback;

export const Config = {
  supabase: {
    url: str(process.env.EXPO_PUBLIC_SUPABASE_URL),
    anonKey: str(process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY),
  },
  api: {
    url: str(process.env.EXPO_PUBLIC_API_URL),
    alicevisionUrl: str(process.env.EXPO_PUBLIC_ALICEVISION_URL),
  },
  ai: {
    provider: (str(process.env.EXPO_PUBLIC_AI_VISION_PROVIDER, 'gemini') as
      | 'gemini'
      | 'nvidia'),
  },
  sentry: {
    dsn: str(process.env.EXPO_PUBLIC_SENTRY_DSN),
  },
  revenueCat: {
    apiKey: str(process.env.EXPO_PUBLIC_REVENUECAT_API_KEY),
  },
  weather: {
    apiKey: str(process.env.EXPO_PUBLIC_WEATHER_API_KEY),
    baseUrl: 'https://api.openweathermap.org/data/2.5',
  },
  admin: {
    email: str(process.env.EXPO_PUBLIC_ADMIN_EMAIL, 'info@aiwardrobe.club'),
  },
} as const;

/**
 * Validate that critical env vars are present.
 * Call inside a React component or effect so errors are catchable.
 * Returns an array of missing variable names (empty = all good).
 */
export function validateConfig(): string[] {
  const missing: string[] = [];
  if (!Config.supabase.url) missing.push('EXPO_PUBLIC_SUPABASE_URL');
  if (!Config.supabase.anonKey) missing.push('EXPO_PUBLIC_SUPABASE_ANON_KEY');
  if (!Config.revenueCat.apiKey || Config.revenueCat.apiKey === 'your-revenuecat-api-key') {
    missing.push('EXPO_PUBLIC_REVENUECAT_API_KEY');
  }
  return missing;
}

export default Config;
