-- Stylist 6: short-lived, owner-scoped recovery of a completed paid deck.
-- Read-only to its owner, writable only by service_role. No new inference,
-- quota debit, client mutation, private wardrobe-photo upload or S5 path.
CREATE TABLE IF NOT EXISTS public.stylist6_completed_results (
  user_id uuid NOT NULL,
  request_id text NOT NULL
    CHECK (length(request_id) BETWEEN 8 AND 128),
  response_payload jsonb NOT NULL
    CHECK (
      jsonb_typeof(response_payload) = 'object'
      AND response_payload->>'success' = 'true'
      AND response_payload->>'generationRequestId' = request_id
      AND left(response_payload->>'generationVersion', 10) = 'stylist-6-'
      AND pg_column_size(response_payload) <= 65536
    ),
  created_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz NOT NULL DEFAULT (now() + interval '45 minutes'),
  PRIMARY KEY (user_id, request_id),
  FOREIGN KEY (user_id, request_id)
    REFERENCES public.stylist6_request_claims(user_id, request_id)
    ON DELETE CASCADE,
  CONSTRAINT s6_result_short_ttl CHECK (
    expires_at > created_at AND
    expires_at <= created_at + interval '60 minutes'
  )
);
ALTER TABLE public.stylist6_completed_results ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.stylist6_completed_results FROM PUBLIC, anon, authenticated;
GRANT SELECT ON public.stylist6_completed_results TO authenticated;
GRANT SELECT, INSERT, DELETE ON public.stylist6_completed_results TO service_role;
DROP POLICY IF EXISTS stylist6_only_owner_recent_complete_results
  ON public.stylist6_completed_results;
CREATE POLICY stylist6_only_owner_recent_complete_results
  ON public.stylist6_completed_results
  FOR SELECT TO authenticated
  USING (user_id=(SELECT auth.uid()) AND expires_at > now());
COMMENT ON TABLE public.stylist6_completed_results IS
  'Ephemeral owner-only already-charged stylist6 result. No client writes, no re-planning on a lost HTTP response. Delete when expired.';
