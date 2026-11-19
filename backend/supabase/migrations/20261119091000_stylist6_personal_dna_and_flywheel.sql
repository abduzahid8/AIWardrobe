-- Stylist 6: Personal Style DNA, Slot-Attributed Feedback, Catalog Visual
-- Quality Flywheel, and Stateless L3 Cache Table.
-- Strictly additive, idempotent, RLS-protected, and zero-downtime backwards
-- compatible with Stylist 5.

BEGIN;
SET LOCAL lock_timeout = '5s';

-- Shared updated_at trigger helper with fixed empty search_path.
CREATE OR REPLACE FUNCTION public.stylist6_touch_updated_at()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = ''
AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION public.stylist6_touch_updated_at() FROM PUBLIC, anon, authenticated;

-- ---------------------------------------------------------------------------
-- 1. public.user_style_dna (Opt-in Personal Style DNA & Learned Taste Snapshot)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.user_style_dna (
  user_id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  presentation text NOT NULL DEFAULT 'menswear'
    CHECK (presentation IN ('menswear', 'womenswear', 'unisex')),
  appearance_consent boolean NOT NULL DEFAULT false,
  appearance_analysis jsonb NOT NULL DEFAULT '{}'::jsonb
    CHECK (
      jsonb_typeof(appearance_analysis) = 'object'
      AND octet_length(appearance_analysis::text) <= 16384
    ),
  color_season text
    CHECK (color_season IS NULL OR length(color_season) BETWEEN 1 AND 64),
  undertone text
    CHECK (undertone IS NULL OR undertone IN ('warm', 'cool', 'neutral', 'olive', 'unknown')),
  contrast text
    CHECK (contrast IS NULL OR contrast IN ('low', 'medium', 'high', 'unknown')),
  body_directives jsonb NOT NULL DEFAULT '{}'::jsonb
    CHECK (
      jsonb_typeof(body_directives) = 'object'
      AND octet_length(body_directives::text) <= 16384
    ),
  style_preferences jsonb NOT NULL DEFAULT '{}'::jsonb
    CHECK (
      jsonb_typeof(style_preferences) = 'object'
      AND octet_length(style_preferences::text) <= 16384
    ),
  learned_taste_snapshot jsonb NOT NULL DEFAULT '{}'::jsonb
    CHECK (
      jsonb_typeof(learned_taste_snapshot) = 'object'
      AND octet_length(learned_taste_snapshot::text) <= 32768
    ),
  photo_sha256 text
    CHECK (photo_sha256 IS NULL OR photo_sha256 ~ '^[0-9a-f]{64}$'),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

DROP TRIGGER IF EXISTS trg_user_style_dna_updated_at ON public.user_style_dna;
CREATE TRIGGER trg_user_style_dna_updated_at
  BEFORE UPDATE ON public.user_style_dna
  FOR EACH ROW
  EXECUTE FUNCTION public.stylist6_touch_updated_at();

ALTER TABLE public.user_style_dna ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS user_style_dna_own_select ON public.user_style_dna;
CREATE POLICY user_style_dna_own_select
  ON public.user_style_dna
  FOR SELECT
  TO authenticated
  USING ((SELECT auth.uid()) = user_id);

DROP POLICY IF EXISTS user_style_dna_own_insert ON public.user_style_dna;
CREATE POLICY user_style_dna_own_insert
  ON public.user_style_dna
  FOR INSERT
  TO authenticated
  WITH CHECK ((SELECT auth.uid()) = user_id);

DROP POLICY IF EXISTS user_style_dna_own_update ON public.user_style_dna;
CREATE POLICY user_style_dna_own_update
  ON public.user_style_dna
  FOR UPDATE
  TO authenticated
  USING ((SELECT auth.uid()) = user_id)
  WITH CHECK ((SELECT auth.uid()) = user_id);

DROP POLICY IF EXISTS user_style_dna_own_delete ON public.user_style_dna;
CREATE POLICY user_style_dna_own_delete
  ON public.user_style_dna
  FOR DELETE
  TO authenticated
  USING ((SELECT auth.uid()) = user_id);

REVOKE ALL ON TABLE public.user_style_dna FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.user_style_dna TO authenticated;
GRANT ALL ON TABLE public.user_style_dna TO service_role;

COMMENT ON TABLE public.user_style_dna IS
  'Owner-scoped opt-in Personal Style DNA (appearance analysis, color season, body proportion directives, style preferences, and learned taste snapshot).';

-- ---------------------------------------------------------------------------
-- 2. public.stylist_look_feedback (Additive Slot-Level Attribution Columns)
-- ---------------------------------------------------------------------------
ALTER TABLE public.stylist_look_feedback
  ADD COLUMN IF NOT EXISTS target_slot text;

ALTER TABLE public.stylist_look_feedback
  ADD COLUMN IF NOT EXISTS feedback_tags text[] NOT NULL DEFAULT '{}';

ALTER TABLE public.stylist_look_feedback
  ADD COLUMN IF NOT EXISTS swapped_from_sku text;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'stylist_feedback_target_slot_check'
      AND conrelid = 'public.stylist_look_feedback'::regclass
  ) THEN
    ALTER TABLE public.stylist_look_feedback
      ADD CONSTRAINT stylist_feedback_target_slot_check
      CHECK (
        target_slot IS NULL
        OR target_slot IN ('top', 'outerwear', 'bottom', 'shoes', 'accessory', 'dress')
      );
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'stylist_feedback_tags_bounded_check'
      AND conrelid = 'public.stylist_look_feedback'::regclass
  ) THEN
    ALTER TABLE public.stylist_look_feedback
      ADD CONSTRAINT stylist_feedback_tags_bounded_check
      CHECK (cardinality(feedback_tags) <= 12);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'stylist_feedback_swapped_sku_len_check'
      AND conrelid = 'public.stylist_look_feedback'::regclass
  ) THEN
    ALTER TABLE public.stylist_look_feedback
      ADD CONSTRAINT stylist_feedback_swapped_sku_len_check
      CHECK (swapped_from_sku IS NULL OR length(swapped_from_sku) BETWEEN 1 AND 128);
  END IF;
END $$;

-- ---------------------------------------------------------------------------
-- 3. public.catalog_visual_quality (Bayesian/Wilson Quality Flywheel Table)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.catalog_visual_quality (
  product_id text PRIMARY KEY REFERENCES public.shop_catalog(id) ON DELETE CASCADE,
  approvals integer NOT NULL DEFAULT 0 CHECK (approvals >= 0),
  rejections integer NOT NULL DEFAULT 0 CHECK (rejections >= 0),
  photo_defects integer NOT NULL DEFAULT 0 CHECK (photo_defects >= 0),
  category_mismatches integer NOT NULL DEFAULT 0 CHECK (category_mismatches >= 0),
  editorial_score_sum numeric(12, 4) NOT NULL DEFAULT 0 CHECK (editorial_score_sum >= 0),
  editorial_score_count integer NOT NULL DEFAULT 0 CHECK (editorial_score_count >= 0),
  quality_score numeric(5, 4) NOT NULL DEFAULT 0.6500
    CHECK (quality_score >= 0 AND quality_score <= 1),
  quarantined boolean NOT NULL DEFAULT false,
  quarantine_reason text
    CHECK (quarantine_reason IS NULL OR length(quarantine_reason) BETWEEN 1 AND 128),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- Fast partial index for active non-quarantined SKU ranking.
CREATE INDEX IF NOT EXISTS catalog_visual_quality_healthy_idx
  ON public.catalog_visual_quality (quality_score DESC, product_id)
  WHERE quarantined = false;

-- Compact partial index for quarantined SKU monitoring and remediation.
CREATE INDEX IF NOT EXISTS catalog_visual_quality_quarantined_idx
  ON public.catalog_visual_quality (updated_at DESC, product_id)
  WHERE quarantined = true;

ALTER TABLE public.catalog_visual_quality ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS catalog_visual_quality_read_all ON public.catalog_visual_quality;
CREATE POLICY catalog_visual_quality_read_all
  ON public.catalog_visual_quality
  FOR SELECT
  TO anon, authenticated
  USING (true);

REVOKE ALL ON TABLE public.catalog_visual_quality FROM PUBLIC, anon, authenticated;
GRANT SELECT ON TABLE public.catalog_visual_quality TO anon, authenticated;
GRANT ALL ON TABLE public.catalog_visual_quality TO service_role;

COMMENT ON TABLE public.catalog_visual_quality IS
  'Per-SKU Visual Critic approval/defect flywheel with Bayesian/Wilson quality score and automatic quarantine state. Service-role write only.';

-- ---------------------------------------------------------------------------
-- 4. public.stylist6_cache_entries (Service-Role Stateless L3 Cache Fallback)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.stylist6_cache_entries (
  cache_key text PRIMARY KEY CHECK (length(cache_key) BETWEEN 1 AND 256),
  namespace text NOT NULL DEFAULT 'stylist6' CHECK (length(namespace) BETWEEN 1 AND 64),
  payload jsonb NOT NULL,
  expires_at timestamptz NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS stylist6_cache_entries_expires_at_idx
  ON public.stylist6_cache_entries (expires_at);

CREATE INDEX IF NOT EXISTS stylist6_cache_entries_ns_expires_idx
  ON public.stylist6_cache_entries (namespace, expires_at);

ALTER TABLE public.stylist6_cache_entries ENABLE ROW LEVEL SECURITY;

-- Service-role ONLY: zero anon/authenticated policies or grants.
REVOKE ALL ON TABLE public.stylist6_cache_entries FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.stylist6_cache_entries TO service_role;

COMMENT ON TABLE public.stylist6_cache_entries IS
  'Server-only stateless L3 distributed cache fallback for Stylist 6 plans, visual verdicts, and appearance analyses.';

-- Short-transaction batched TTL cleanup helper.
CREATE OR REPLACE FUNCTION public.cleanup_expired_stylist6_cache(
  p_batch_size integer DEFAULT 5000
)
RETURNS integer
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = ''
AS $$
DECLARE
  v_deleted integer := 0;
BEGIN
  IF p_batch_size IS NULL OR p_batch_size < 1 OR p_batch_size > 50000 THEN
    RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid_cleanup_batch_size';
  END IF;

  WITH expired AS (
    SELECT c.cache_key
    FROM public.stylist6_cache_entries c
    WHERE c.expires_at < now()
    ORDER BY c.expires_at ASC
    LIMIT p_batch_size
    FOR UPDATE SKIP LOCKED
  )
  DELETE FROM public.stylist6_cache_entries target
  USING expired
  WHERE target.cache_key = expired.cache_key;

  GET DIAGNOSTICS v_deleted = ROW_COUNT;
  RETURN v_deleted;
END;
$$;

REVOKE ALL ON FUNCTION public.cleanup_expired_stylist6_cache(integer) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.cleanup_expired_stylist6_cache(integer) TO service_role;

-- Idempotent pg_cron registration (runs safely even when pg_cron is absent).
DO $$
DECLARE
  v_job_id bigint;
BEGIN
  IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
    SELECT jobid INTO v_job_id
    FROM cron.job
    WHERE jobname = 'aiw-s6-expired-cache-entries'
    LIMIT 1;

    IF v_job_id IS NOT NULL THEN
      PERFORM cron.unschedule(v_job_id);
    END IF;

    PERFORM cron.schedule(
      'aiw-s6-expired-cache-entries',
      '*/15 * * * *',
      'SELECT public.cleanup_expired_stylist6_cache(5000);'
    );
  END IF;
END $$;

NOTIFY pgrst, 'reload schema';
COMMIT;
