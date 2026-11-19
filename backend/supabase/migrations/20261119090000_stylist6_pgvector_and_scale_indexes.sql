-- Stylist 6: Strictly additive pgvector ANN retrieval + scale indexes.
-- Preserves Stylist 5 untouched: legacy `catalog_image_embeddings.embedding`
-- (real[]) and all existing `shop_catalog` columns/indexes remain intact.
-- Idempotent, zero-downtime, and pinned to explicit search paths.

BEGIN;
SET LOCAL lock_timeout = '5s';

-- 1. Install extensions in Supabase's dedicated `extensions` schema.
CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;
CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA extensions;

-- 2. Add pgvector column to catalog_image_embeddings (keeping legacy real[] intact).
ALTER TABLE public.catalog_image_embeddings
  ADD COLUMN IF NOT EXISTS embedding_vec extensions.vector(768);

-- Automatically mirror legacy real[] writes (e.g. Stylist 5 visual worker)
-- into embedding_vec with zero changes required to existing ingestion RPCs.
CREATE OR REPLACE FUNCTION public.sync_catalog_embedding_vec()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = ''
AS $$
BEGIN
  IF NEW.embedding_vec IS NULL
     AND NEW.embedding IS NOT NULL
     AND array_length(NEW.embedding, 1) = 768 THEN
    NEW.embedding_vec := NEW.embedding::extensions.vector(768);
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_sync_catalog_embedding_vec ON public.catalog_image_embeddings;
CREATE TRIGGER trg_sync_catalog_embedding_vec
  BEFORE INSERT OR UPDATE OF embedding
  ON public.catalog_image_embeddings
  FOR EACH ROW
  EXECUTE FUNCTION public.sync_catalog_embedding_vec();

REVOKE ALL ON FUNCTION public.sync_catalog_embedding_vec() FROM PUBLIC, anon, authenticated;

-- Partial HNSW cosine index for fast ANN retrieval (only indexed once populated).
CREATE INDEX IF NOT EXISTS catalog_image_embeddings_vec_hnsw_idx
  ON public.catalog_image_embeddings
  USING hnsw (embedding_vec extensions.vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE embedding_vec IS NOT NULL;

-- Covering join index so ANN results resolve to product_id in O(log N).
CREATE INDEX IF NOT EXISTS catalog_visual_fashion_sha_rev_idx
  ON public.catalog_visual_fashion (image_sha256, model_revision, product_id)
  WHERE version = 1;

-- Short-transaction batched backfill helper using SKIP LOCKED (no long locks).
CREATE OR REPLACE FUNCTION public.backfill_catalog_embedding_vecs(
  p_batch_size integer DEFAULT 500
)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = ''
AS $$
DECLARE
  v_updated integer := 0;
  v_remaining bigint := 0;
BEGIN
  IF p_batch_size IS NULL OR p_batch_size < 1 OR p_batch_size > 5000 THEN
    RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid_backfill_batch_size';
  END IF;

  WITH batch AS (
    SELECT e.image_sha256, e.model_revision
    FROM public.catalog_image_embeddings e
    WHERE e.embedding_vec IS NULL
      AND e.embedding IS NOT NULL
      AND array_length(e.embedding, 1) = 768
    LIMIT p_batch_size
    FOR UPDATE SKIP LOCKED
  )
  UPDATE public.catalog_image_embeddings target
  SET embedding_vec = target.embedding::extensions.vector(768)
  FROM batch
  WHERE target.image_sha256 = batch.image_sha256
    AND target.model_revision = batch.model_revision;

  GET DIAGNOSTICS v_updated = ROW_COUNT;

  SELECT count(*)
  INTO v_remaining
  FROM public.catalog_image_embeddings
  WHERE embedding_vec IS NULL
    AND embedding IS NOT NULL
    AND array_length(embedding, 1) = 768;

  RETURN jsonb_build_object(
    'updated', v_updated,
    'remaining', v_remaining
  );
END;
$$;

REVOKE ALL ON FUNCTION public.backfill_catalog_embedding_vecs(integer) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.backfill_catalog_embedding_vecs(integer) TO service_role;

-- 3. Add gender support and scale retrieval indexes on shop_catalog.
-- Instant metadata-only ADD COLUMN with constant DEFAULT in Postgres 11+.
ALTER TABLE public.shop_catalog
  ADD COLUMN IF NOT EXISTS gender text NOT NULL DEFAULT 'menswear';

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'shop_catalog_gender_check'
      AND conrelid = 'public.shop_catalog'::regclass
  ) THEN
    ALTER TABLE public.shop_catalog
      ADD CONSTRAINT shop_catalog_gender_check
      CHECK (gender IN ('menswear', 'womenswear', 'unisex'));
  END IF;
END $$;

-- Composite active retrieval index (equality columns before range column `price`).
CREATE INDEX IF NOT EXISTS shop_catalog_active_gender_cat_price_idx
  ON public.shop_catalog (is_active, gender, category, price)
  WHERE is_active = TRUE;

-- Trigram search index for active product title filtering and hygiene lookups.
CREATE INDEX IF NOT EXISTS shop_catalog_active_name_trgm_idx
  ON public.shop_catalog
  USING gin (name extensions.gin_trgm_ops)
  WHERE is_active = TRUE;

-- 4. Indexed ANN retrieval RPC for visual taste / similarity discovery.
CREATE OR REPLACE FUNCTION public.match_catalog_embeddings(
  p_query_embedding extensions.vector(768),
  p_match_count integer DEFAULT 40,
  p_min_similarity double precision DEFAULT 0.15,
  p_gender text DEFAULT NULL,
  p_slot text DEFAULT NULL,
  p_currency text DEFAULT NULL,
  p_max_price numeric DEFAULT NULL,
  p_model_revision text DEFAULT NULL
)
RETURNS TABLE (
  product_id text,
  image_sha256 text,
  similarity double precision,
  slot text,
  gender text,
  category text,
  price numeric,
  currency text,
  brand text,
  name text
)
LANGUAGE plpgsql
STABLE
SECURITY INVOKER
SET search_path = ''
AS $$
DECLARE
  v_overfetch integer;
BEGIN
  IF p_query_embedding IS NULL THEN
    RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'query_embedding_required';
  END IF;
  IF extensions.vector_dims(p_query_embedding) <> 768 THEN
    RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid_embedding_dimensions';
  END IF;
  IF p_match_count IS NULL OR p_match_count < 1 OR p_match_count > 200 THEN
    RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid_match_count';
  END IF;
  IF p_min_similarity IS NULL OR p_min_similarity < -1.0 OR p_min_similarity > 1.0 THEN
    RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid_min_similarity';
  END IF;
  IF p_gender IS NOT NULL AND p_gender NOT IN ('menswear', 'womenswear', 'unisex') THEN
    RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid_gender_filter';
  END IF;
  IF p_slot IS NOT NULL AND p_slot NOT IN ('top', 'outerwear', 'bottom', 'shoes', 'dress', 'accessory') THEN
    RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid_slot_filter';
  END IF;
  IF p_currency IS NOT NULL AND p_currency !~ '^[A-Z]{3}$' THEN
    RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid_currency_filter';
  END IF;
  IF p_max_price IS NOT NULL AND (p_max_price < 0 OR p_max_price >= 'Infinity'::numeric) THEN
    RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid_max_price_filter';
  END IF;

  v_overfetch := LEAST(800, p_match_count * 8);

  RETURN QUERY
  WITH ann_candidates AS (
    SELECT
      e.image_sha256,
      e.model_revision,
      (1.0 - (e.embedding_vec OPERATOR(extensions.<=>) p_query_embedding))::double precision AS sim
    FROM public.catalog_image_embeddings e
    WHERE e.embedding_vec IS NOT NULL
      AND (p_model_revision IS NULL OR e.model_revision = p_model_revision)
    ORDER BY e.embedding_vec OPERATOR(extensions.<=>) p_query_embedding ASC
    LIMIT v_overfetch
  )
  SELECT
    s.id AS product_id,
    a.image_sha256,
    a.sim AS similarity,
    COALESCE(i.slot, v.profile #>> '{attributes,slot,value}', s.category) AS slot,
    s.gender,
    s.category,
    s.price,
    s.currency,
    s.brand,
    s.name
  FROM ann_candidates a
  JOIN public.catalog_visual_fashion v
    ON v.image_sha256 = a.image_sha256
   AND v.model_revision = a.model_revision
   AND v.version = 1
  JOIN public.shop_catalog s
    ON s.id = v.product_id
   AND s.is_active = TRUE
   AND s.image_url = v.image_url
  LEFT JOIN public.catalog_garment_intelligence i
    ON i.product_id = s.id
   AND i.version = 1
  WHERE a.sim >= p_min_similarity
    AND (p_gender IS NULL OR s.gender = p_gender OR s.gender = 'unisex')
    AND (
      p_slot IS NULL
      OR COALESCE(i.slot, v.profile #>> '{attributes,slot,value}') = p_slot
    )
    AND (p_currency IS NULL OR s.currency = p_currency)
    AND (p_max_price IS NULL OR s.price <= p_max_price)
  ORDER BY a.sim DESC, s.id ASC
  LIMIT p_match_count;
END;
$$;

REVOKE ALL ON FUNCTION public.match_catalog_embeddings(
  extensions.vector, integer, double precision, text, text, text, numeric, text
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.match_catalog_embeddings(
  extensions.vector, integer, double precision, text, text, text, numeric, text
) TO service_role;

NOTIFY pgrst, 'reload schema';
COMMIT;
