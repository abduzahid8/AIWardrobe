-- AIWardrobe Stylist 6: LICENSE-ATTESTED and exact-current-product-bound
-- merchant photo cache. This does NOT copy any third-party photographs,
-- infer ownership, alter live catalog images, or change Stylist 5.
-- Server-only table: no anon/authenticated privileges or RLS policies.
CREATE TABLE IF NOT EXISTS public.catalog_photo_cache_v1 (
  product_id text PRIMARY KEY REFERENCES public.shop_catalog(id) ON DELETE CASCADE,
  source_image_url text NOT NULL CHECK (source_image_url ~ '^https://'),
  source_hash text NOT NULL CHECK (source_hash ~ '^[a-f0-9]{64}$'),
  storage_bucket text NOT NULL DEFAULT 'shop-catalog'
    CHECK (storage_bucket = 'shop-catalog'),
  storage_path text NOT NULL UNIQUE CHECK
    (storage_path ~ '^licensed-stylist6/[a-f0-9]{2}/[a-f0-9]{64}[.]jpg$'),
  image_sha256 text NOT NULL CHECK (image_sha256 ~ '^[a-f0-9]{64}$'),
  media_type text NOT NULL DEFAULT 'image/jpeg' CHECK (media_type='image/jpeg'),
  status text NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending','ready','revoked')),
  rights_basis text NOT NULL
    CHECK (rights_basis IN ('owner_upload','licensed_provider','merchant_contract')),
  rights_reference text NOT NULL CHECK (length(rights_reference) BETWEEN 8 AND 500),
  cache_hosting_granted boolean NOT NULL DEFAULT false,
  external_ai_inference_granted boolean NOT NULL DEFAULT false,
  rights_granted_at timestamptz NOT NULL,
  rights_expires_at timestamptz,
  last_verified_at timestamptz NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  CHECK (rights_expires_at IS NULL OR rights_expires_at > rights_granted_at)
);
ALTER TABLE public.catalog_photo_cache_v1 ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.catalog_photo_cache_v1 FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.catalog_photo_cache_v1 TO service_role;
COMMENT ON TABLE public.catalog_photo_cache_v1 IS
  'Privileged licensed catalog photo evidence: source SKU and source URL must match original product. Never substitute stale photographs.';
