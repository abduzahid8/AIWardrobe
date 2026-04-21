-- ============================================================
-- AIWardrobe — Shop catalog: affiliate / feed sync metadata
-- Run after 005_shop_catalog.sql
-- ============================================================

ALTER TABLE public.shop_catalog
    ADD COLUMN IF NOT EXISTS source         TEXT NOT NULL DEFAULT 'manual',
    ADD COLUMN IF NOT EXISTS source_url     TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS external_id    TEXT,
    ADD COLUMN IF NOT EXISTS last_seen_at   TIMESTAMPTZ;

COMMENT ON COLUMN public.shop_catalog.source       IS 'Origin: manual, awin, cj, impact, zara_feed, etc.';
COMMENT ON COLUMN public.shop_catalog.source_url   IS 'Product page URL (affiliate or canonical).';
COMMENT ON COLUMN public.shop_catalog.external_id  IS 'Merchant / feed product id for idempotent upserts.';
COMMENT ON COLUMN public.shop_catalog.last_seen_at IS 'Last time this row appeared in an import (for stale cleanup).';

CREATE INDEX IF NOT EXISTS shop_catalog_source_idx        ON public.shop_catalog (source);
CREATE INDEX IF NOT EXISTS shop_catalog_last_seen_at_idx  ON public.shop_catalog (last_seen_at);

-- One row per (source, external_id) when external_id is set (feeds)
CREATE UNIQUE INDEX IF NOT EXISTS shop_catalog_source_external_unique
    ON public.shop_catalog (source, external_id)
    WHERE external_id IS NOT NULL AND btrim(external_id) <> '';
