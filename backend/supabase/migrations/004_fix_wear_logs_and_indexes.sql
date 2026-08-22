-- ============================================================
-- AIWardrobe — Migration 004: Fix wear_logs schema + add indexes
-- Run this in: Supabase Dashboard → SQL Editor
-- ============================================================

-- ─────────────────────────────────────────────────────────────
-- 1. Fix wear_logs: support multi-item logging
--    App logs multiple item IDs per wear event, but the schema
--    only had a single clothing_item_id FK. Add item_ids array
--    and make the single FK nullable for backward compat.
-- ─────────────────────────────────────────────────────────────

ALTER TABLE public.wear_logs
    ADD COLUMN IF NOT EXISTS item_ids UUID[] DEFAULT '{}';

-- Backfill: copy existing single FK into the array
UPDATE public.wear_logs
SET item_ids = ARRAY[clothing_item_id]
WHERE clothing_item_id IS NOT NULL
  AND (item_ids IS NULL OR item_ids = '{}');

-- Make the single FK nullable (new inserts use item_ids instead)
ALTER TABLE public.wear_logs
    ALTER COLUMN clothing_item_id DROP NOT NULL;

-- Add weather columns used by the app
ALTER TABLE public.wear_logs
    ADD COLUMN IF NOT EXISTS weather_temp NUMERIC,
    ADD COLUMN IF NOT EXISTS weather_condition TEXT DEFAULT '';

-- ─────────────────────────────────────────────────────────────
-- 2. Performance indexes
-- ─────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_clothing_items_user_id
    ON public.clothing_items(user_id);

CREATE INDEX IF NOT EXISTS idx_clothing_items_user_category
    ON public.clothing_items(user_id, category);

CREATE INDEX IF NOT EXISTS idx_wear_logs_user_date
    ON public.wear_logs(user_id, date DESC);

CREATE INDEX IF NOT EXISTS idx_saved_outfits_user_id
    ON public.saved_outfits(user_id);

CREATE INDEX IF NOT EXISTS idx_subscriptions_user_active
    ON public.subscriptions(user_id, status, end_date);

CREATE INDEX IF NOT EXISTS idx_payments_user_id
    ON public.payments(user_id);

-- ─────────────────────────────────────────────────────────────
-- 3. Add missing profiles INSERT policy
--    The trigger uses SECURITY DEFINER, but any direct client
--    insert would fail without this policy.
-- ─────────────────────────────────────────────────────────────

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'profiles'
          AND policyname = 'Users can insert own profile'
    ) THEN
        CREATE POLICY "Users can insert own profile"
            ON public.profiles FOR INSERT
            WITH CHECK (auth.uid() = id);
    END IF;
END $$;
