-- ============================================================
-- AIWardrobe — Migration 002: app_config table
-- Required by: supabase/functions/ai-process/index.ts
-- Run this in: Supabase Dashboard → SQL Editor
-- ============================================================

CREATE TABLE IF NOT EXISTS public.app_config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Only the service role (Edge Functions) can read/write this table.
-- No public or authenticated access.
ALTER TABLE public.app_config ENABLE ROW LEVEL SECURITY;

-- No RLS policies → zero rows visible to the anon/authenticated roles.
-- The Edge Function uses the service role key (SB_SERVICE_KEY) which bypasses RLS.

-- ── Insert your API keys below ──────────────────────────────
-- Run these separately after creating the table:
--
-- INSERT INTO public.app_config (key, value) VALUES ('nvidia_token',   'nvapi-YOUR_KEY_HERE')
--   ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
--
-- INSERT INTO public.app_config (key, value) VALUES ('replicate_token', 'r8_YOUR_KEY_HERE')
--   ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
--
-- INSERT INTO public.app_config (key, value) VALUES ('hf_token',        'hf_YOUR_KEY_HERE')
--   ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
