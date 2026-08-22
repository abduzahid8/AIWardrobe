-- 016_perf_indexes.sql
-- Hot-path indexes for the read patterns the app actually issues at scale.
-- All indexes are IF NOT EXISTS so re-running is safe.
-- NOTE: CREATE INDEX CONCURRENTLY cannot run inside a transaction. Supabase's
-- migration runner wraps statements in a transaction by default, so we use
-- plain CREATE INDEX. This will briefly hold a lock on small tables; for
-- larger tables in the future, run CONCURRENTLY manually via the SQL editor.

-- Wardrobe items: list/filter by owner
CREATE INDEX IF NOT EXISTS wardrobe_items_user_idx
    ON public.wardrobe_items (user_id);

-- Outfits: list by owner, latest first
CREATE INDEX IF NOT EXISTS outfits_user_created_idx
    ON public.outfits (user_id, created_at DESC);

-- Wear logs: "what did I wear lately" queries
CREATE INDEX IF NOT EXISTS wear_logs_user_worn_idx
    ON public.wear_logs (user_id, worn_at DESC);

-- Subscriptions: only active/trial rows are ever queried for gating
CREATE INDEX IF NOT EXISTS subscriptions_user_active_idx
    ON public.subscriptions (user_id)
    WHERE status IN ('active', 'trial');

-- Subscriptions: webhook lookups by product
CREATE INDEX IF NOT EXISTS subscriptions_user_product_idx
    ON public.subscriptions (user_id, product_id);

-- Payments: per-user history, newest first
CREATE INDEX IF NOT EXISTS payments_user_created_idx
    ON public.payments (user_id, created_at DESC);

-- Shop catalog: browse by gender + category, only visible items
CREATE INDEX IF NOT EXISTS shop_catalog_browse_idx
    ON public.shop_catalog (gender, category)
    WHERE active = true;

-- Promo redemptions: idempotency lookup is already UNIQUE on user_id,
-- but add an index on promo_code_id for usage analytics.
CREATE INDEX IF NOT EXISTS promo_redemptions_code_idx
    ON public.promo_redemptions (promo_code_id);

-- Profiles: admin lookups + email search
CREATE INDEX IF NOT EXISTS profiles_email_idx
    ON public.profiles (lower(email));
