-- Stylist 6: separate PHOTO-critic reward evidence; NOT owner/human taste.
-- Stores only keyed SHA-256 digests of public catalog appearance features and
-- real independently judged numeric whole-outfit photo quality. No user ID,
-- request ID, SKU, brand, store URL, image bytes, free-text prompts or owner
-- measurements. This table must never be exposed to client Data API roles.
BEGIN;
SET LOCAL lock_timeout='2s';
CREATE TABLE IF NOT EXISTS public.stylist6_photo_reward_examples (
  fingerprint text PRIMARY KEY CHECK (fingerprint ~ '^[0-9a-f]{64}$'),
  context_hash text NOT NULL CHECK (context_hash ~ '^[0-9a-f]{64}$'),
  feature_hashes text[] NOT NULL
    CHECK (cardinality(feature_hashes) BETWEEN 12 AND 180
      AND pg_column_size(feature_hashes) <= 13000),
  photo_quality numeric(5,4) NOT NULL
    CHECK (photo_quality BETWEEN 0 AND 1),
  photo_outcome smallint NOT NULL CHECK (photo_outcome IN (-1,1)),
  photo_reason text NOT NULL CHECK (photo_reason IN (
    'approved','model_veto','below_quality_floor',
    'style_mismatch','occasion_mismatch','formality_mismatch')),
  evidence_version text NOT NULL DEFAULT 's6-photo-reward-journal-v1'
    CHECK (evidence_version = 's6-photo-reward-journal-v1'),
  created_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz NOT NULL DEFAULT (now() + interval '30 days')
    CHECK (expires_at > created_at
      AND expires_at <= created_at + interval '31 days')
);
CREATE INDEX IF NOT EXISTS stylist6_photo_reward_context_recent
  ON public.stylist6_photo_reward_examples(context_hash,created_at DESC);
ALTER TABLE public.stylist6_photo_reward_examples ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.stylist6_photo_reward_examples
  FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, DELETE ON TABLE public.stylist6_photo_reward_examples
  TO service_role;
COMMENT ON TABLE public.stylist6_photo_reward_examples IS
  'Staging-only, privacy-pseudonymized photo-critic training judgments; never human preferences. Private service role only, 30-day TTL, no SKU/person/photo identifiers.';
SELECT cron.schedule(
  'aiw-s6-photo-reward-examples-ttl',
  '0 */6 * * *',
  $$DELETE FROM public.stylist6_photo_reward_examples WHERE expires_at < now()$$
);
NOTIFY pgrst,'reload schema';
COMMIT;
