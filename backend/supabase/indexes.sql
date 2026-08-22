-- ============================================
-- PERFORMANCE INDEXES FOR AIWARDROBE
-- Run after schema.sql has been applied
-- ============================================

-- Clothing items: most queries filter by user_id
CREATE INDEX IF NOT EXISTS idx_clothing_items_user_id
    ON public.clothing_items (user_id);

-- Clothing items: filter by user + category (common on MyCloset screen)
CREATE INDEX IF NOT EXISTS idx_clothing_items_user_category
    ON public.clothing_items (user_id, category);

-- Clothing items: favorites filter
CREATE INDEX IF NOT EXISTS idx_clothing_items_user_favorite
    ON public.clothing_items (user_id) WHERE is_favorite = TRUE;

-- Wear logs: filter by user + date range (analytics, streaks)
CREATE INDEX IF NOT EXISTS idx_wear_logs_user_date
    ON public.wear_logs (user_id, date DESC);

-- Wear logs: join on clothing_item_id (cost-per-wear calculations)
CREATE INDEX IF NOT EXISTS idx_wear_logs_clothing_item
    ON public.wear_logs (clothing_item_id);

-- Saved outfits: filter by user
CREATE INDEX IF NOT EXISTS idx_saved_outfits_user_id
    ON public.saved_outfits (user_id);

-- Saved outfits: filter by date (outfit calendar)
CREATE INDEX IF NOT EXISTS idx_saved_outfits_user_date
    ON public.saved_outfits (user_id, date);

-- Subscriptions: active subscription lookup (most critical query)
CREATE INDEX IF NOT EXISTS idx_subscriptions_user_status
    ON public.subscriptions (user_id, status) WHERE status IN ('active', 'trial');

-- Subscriptions: expiry check
CREATE INDEX IF NOT EXISTS idx_subscriptions_end_date
    ON public.subscriptions (end_date) WHERE status = 'active';

-- Payments: user payment history
CREATE INDEX IF NOT EXISTS idx_payments_user_id
    ON public.payments (user_id);

-- Payments: filter by status (for reconciliation)
CREATE INDEX IF NOT EXISTS idx_payments_status
    ON public.payments (status) WHERE status IN ('pending', 'disputed');

-- Profiles: email lookup (login)
CREATE INDEX IF NOT EXISTS idx_profiles_email
    ON public.profiles (email);
