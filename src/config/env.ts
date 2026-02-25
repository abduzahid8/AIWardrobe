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

export const Config = {
  supabase: {
    url: requireEnv('EXPO_PUBLIC_SUPABASE_URL'),
    anonKey: requireEnv('EXPO_PUBLIC_SUPABASE_ANON_KEY'),
  },
  api: {
    url: process.env.EXPO_PUBLIC_API_URL || (__DEV__ ? 'http://localhost:3000' : 'https://api.aiwardrobe.com'),
    alicevisionUrl: process.env.EXPO_PUBLIC_ALICEVISION_URL || (__DEV__ ? 'http://localhost:5050' : 'https://alicevision.aiwardrobe.com'),
  },
  weather: {
    apiKey: process.env.EXPO_PUBLIC_WEATHER_API_KEY || '',
    baseUrl: 'https://api.openweathermap.org/data/2.5',
  },
} as const;

export default Config;
