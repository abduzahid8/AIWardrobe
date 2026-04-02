/**
 * Centralized environment configuration
 * All env vars must be prefixed with EXPO_PUBLIC_ to be available in the bundle
 * Never hardcode secrets — add them to .env (gitignored) and .env.example (committed)
 */

const requireEnv = (key: string, fallback?: string): string => {
  const value = process.env[key] || fallback;
  if (!value) {
    throw new Error(
      `[Config] Missing required environment variable: ${key}\n` +
      `Copy .env.example to .env and fill in your values.`
    );
  }
  return value;
};

const getDefaultApiUrl = (): string | undefined => {
  if (__DEV__) {
    return 'http://localhost:3000';
  }
  return undefined;
};

export const Config = {
  supabase: {
    url: requireEnv('EXPO_PUBLIC_SUPABASE_URL'),
    anonKey: requireEnv('EXPO_PUBLIC_SUPABASE_ANON_KEY'),
  },
  api: {
    // No server needed - using external AI APIs directly
    url: process.env.EXPO_PUBLIC_API_URL || '', // Optional: for any legacy endpoints
    alicevisionUrl: process.env.EXPO_PUBLIC_ALICEVISION_URL || '',
  },
  ai: {
    provider: (process.env.EXPO_PUBLIC_AI_VISION_PROVIDER as 'gemini' | 'nvidia') || 'gemini',
    geminiApiKey: process.env.EXPO_PUBLIC_GEMINI_API_KEY || '',
    nvidiaApiKey: process.env.EXPO_PUBLIC_NVIDIA_API_KEY || '',
    huggingfaceToken: process.env.EXPO_PUBLIC_HF_TOKEN || '',
    replicateToken: process.env.EXPO_PUBLIC_REPLICATE_TOKEN || '',
  },
  sentry: {
    dsn: process.env.EXPO_PUBLIC_SENTRY_DSN || '',
  },
  revenueCat: {
    apiKey: process.env.EXPO_PUBLIC_REVENUECAT_API_KEY || '',
  },
  weather: {
    apiKey: process.env.EXPO_PUBLIC_WEATHER_API_KEY || '',
    baseUrl: 'https://api.openweathermap.org/data/2.5',
  },
} as const;

export default Config;
