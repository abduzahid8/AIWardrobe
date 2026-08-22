-- ============================================================
-- Fix: Add missing trial_end_date column to subscriptions
-- Required for promo code redemption system
-- ============================================================

-- Add the missing column
ALTER TABLE public.subscriptions 
ADD COLUMN IF NOT EXISTS trial_end_date TIMESTAMPTZ;

-- Verify the column was added
SELECT 
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns 
WHERE table_name = 'subscriptions' 
AND table_schema = 'public'
AND column_name = 'trial_end_date';
