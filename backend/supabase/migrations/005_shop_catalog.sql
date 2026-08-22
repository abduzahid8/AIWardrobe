-- ============================================================
-- AIWardrobe — Shop Catalog
-- Run in: Supabase Dashboard → SQL Editor
-- ============================================================

-- ── 1. TABLE ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.shop_catalog (
    id            TEXT PRIMARY KEY,              -- e.g. 'shop-inspo-1'
    brand         TEXT        NOT NULL DEFAULT '',
    name          TEXT        NOT NULL,
    price         NUMERIC     NOT NULL DEFAULT 0,
    currency      TEXT        NOT NULL DEFAULT 'USD',
    category      TEXT        NOT NULL DEFAULT 'tops',  -- tops | bottoms | shoes | dresses | outerwear
    garment_type  TEXT        NOT NULL DEFAULT 'upper_body', -- upper_body | lower_body | dresses | shoes | outfit
    description   TEXT        NOT NULL DEFAULT '',
    image_url     TEXT        NOT NULL,          -- public URL (Supabase Storage or CDN)
    is_active     BOOLEAN     NOT NULL DEFAULT TRUE,
    sort_order    INTEGER     NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── 2. INDEXES ────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS shop_catalog_category_idx     ON public.shop_catalog (category);
CREATE INDEX IF NOT EXISTS shop_catalog_garment_type_idx ON public.shop_catalog (garment_type);
CREATE INDEX IF NOT EXISTS shop_catalog_sort_order_idx   ON public.shop_catalog (sort_order);
CREATE INDEX IF NOT EXISTS shop_catalog_is_active_idx    ON public.shop_catalog (is_active);

-- ── 3. ROW-LEVEL SECURITY ─────────────────────────────────────
ALTER TABLE public.shop_catalog ENABLE ROW LEVEL SECURITY;

-- Everyone (including unauthenticated visitors) can read active items
CREATE POLICY "shop_catalog_public_read"
    ON public.shop_catalog
    FOR SELECT
    USING (is_active = TRUE);

-- Only service-role (admin scripts / edge functions) can write
CREATE POLICY "shop_catalog_service_write"
    ON public.shop_catalog
    FOR ALL
    USING (auth.role() = 'service_role');

-- ── 4. STORAGE BUCKET ────────────────────────────────────────
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'shop-catalog',
    'shop-catalog',
    TRUE,                          -- publicly readable (no signed URLs needed)
    5242880,                       -- 5 MB per file
    ARRAY['image/png', 'image/jpeg', 'image/webp']
)
ON CONFLICT (id) DO NOTHING;

-- Anyone can read files in this bucket
CREATE POLICY "shop_catalog_storage_public_read"
    ON storage.objects
    FOR SELECT
    USING (bucket_id = 'shop-catalog');

-- Only service-role can upload / delete
CREATE POLICY "shop_catalog_storage_service_write"
    ON storage.objects
    FOR ALL
    USING (bucket_id = 'shop-catalog' AND auth.role() = 'service_role');
