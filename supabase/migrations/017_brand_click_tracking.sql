-- ============================================================
-- AIWardrobe — Brand Click Tracking
-- Tracks user clicks on brand products for analytics
-- Run in: Supabase Dashboard → SQL Editor
-- ============================================================

-- ── 1. TABLE ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.brand_clicks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    item_id       TEXT NOT NULL,                    -- shop_catalog item id
    brand         TEXT NOT NULL,                    -- brand name
    product_name  TEXT NOT NULL DEFAULT '',         -- product name
    price         NUMERIC NOT NULL DEFAULT 0,       -- price at time of click
    currency      TEXT NOT NULL DEFAULT 'USD',      -- currency
    clicked_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    source        TEXT NOT NULL DEFAULT 'app',      -- app | web
    device_type   TEXT                              -- ios | android
);

-- ── 2. INDEXES ────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS brand_clicks_user_id_idx ON public.brand_clicks (user_id);
CREATE INDEX IF NOT EXISTS brand_clicks_brand_idx ON public.brand_clicks (brand);
CREATE INDEX IF NOT EXISTS brand_clicks_item_id_idx ON public.brand_clicks (item_id);
CREATE INDEX IF NOT EXISTS brand_clicks_clicked_at_idx ON public.brand_clicks (clicked_at DESC);
CREATE INDEX IF NOT EXISTS brand_clicks_brand_date_idx ON public.brand_clicks (brand, clicked_at DESC);

-- ── 3. ROW-LEVEL SECURITY ─────────────────────────────────────
ALTER TABLE public.brand_clicks ENABLE ROW LEVEL SECURITY;

-- Users can only see their own clicks
CREATE POLICY "brand_clicks_user_read_own"
    ON public.brand_clicks
    FOR SELECT
    USING (auth.uid() = user_id);

-- Users can insert their own clicks
CREATE POLICY "brand_clicks_user_insert"
    ON public.brand_clicks
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Service role can do everything (for analytics queries)
CREATE POLICY "brand_clicks_service_all"
    ON public.brand_clicks
    FOR ALL
    USING (auth.role() = 'service_role');

-- ── 4. ANALYTICS VIEW ─────────────────────────────────────────
CREATE OR REPLACE VIEW public.brand_click_stats AS
SELECT 
    brand,
    COUNT(*) as total_clicks,
    COUNT(DISTINCT user_id) as unique_users,
    DATE_TRUNC('day', clicked_at) as click_date,
    AVG(price) as avg_price_clicked
FROM public.brand_clicks
GROUP BY brand, DATE_TRUNC('day', clicked_at)
ORDER BY total_clicks DESC;

-- ── 5. FUNCTION TO RECORD CLICK ───────────────────────────────
CREATE OR REPLACE FUNCTION public.record_brand_click(
    p_item_id TEXT,
    p_brand TEXT,
    p_product_name TEXT DEFAULT '',
    p_price NUMERIC DEFAULT 0,
    p_currency TEXT DEFAULT 'USD',
    p_source TEXT DEFAULT 'app',
    p_device_type TEXT DEFAULT NULL
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_click_id UUID;
BEGIN
    INSERT INTO public.brand_clicks (
        user_id,
        item_id,
        brand,
        product_name,
        price,
        currency,
        source,
        device_type
    ) VALUES (
        auth.uid(),
        p_item_id,
        p_brand,
        p_product_name,
        p_price,
        p_currency,
        p_source,
        p_device_type
    )
    RETURNING id INTO v_click_id;
    
    RETURN v_click_id;
END;
$$;
