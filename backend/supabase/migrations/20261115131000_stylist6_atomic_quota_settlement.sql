-- Stylist 6: transactional quota charge and exactly-once refund for failed
-- logical generations. Additive to the existing production schema; Stylist 5
-- continues to use consume_edge_quota unchanged.
-- Only service_role can invoke the new RPCs. All operations validate an
-- existing durable request claim and lock it before updating the counter.
BEGIN;
SET LOCAL lock_timeout = '3s';
ALTER TABLE public.stylist6_request_claims
  ADD COLUMN IF NOT EXISTS quota_debited boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS quota_refunded boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS quota_day date;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='stylist6_refund_requires_debit' AND conrelid='public.stylist6_request_claims'::regclass) THEN
    ALTER TABLE public.stylist6_request_claims ADD CONSTRAINT stylist6_refund_requires_debit
      CHECK (NOT quota_refunded OR (quota_debited AND quota_day IS NOT NULL));
  END IF;
END $$;

CREATE OR REPLACE FUNCTION public.charge_stylist6_request(p_user uuid,p_request_id text)
RETURNS boolean LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$
DECLARE v_claim public.stylist6_request_claims%ROWTYPE; v_allowed boolean;
BEGIN
  IF p_user IS NULL OR p_request_id IS NULL OR p_request_id = '' OR length(p_request_id)>128 THEN
    RAISE EXCEPTION 'invalid_stylist6_claim' USING errcode='22023';
  END IF;
  SELECT * INTO v_claim FROM public.stylist6_request_claims
    WHERE user_id=p_user AND request_id=p_request_id FOR UPDATE;
  IF NOT FOUND OR v_claim.status_code<>0 OR v_claim.quota_debited THEN
    RAISE EXCEPTION 'stylist6_claim_not_chargeable' USING errcode='23505';
  END IF;
  v_allowed := public.consume_edge_quota(p_user,'generate-outfits');
  IF v_allowed IS DISTINCT FROM true THEN RETURN false; END IF;
  UPDATE public.stylist6_request_claims SET quota_debited=true,
    quota_day=(now() at time zone 'utc')::date,updated_at=now()
    WHERE user_id=p_user AND request_id=p_request_id;
  RETURN true;
END;
$$;

CREATE OR REPLACE FUNCTION public.settle_stylist6_request(p_user uuid,p_request_id text,p_status integer,p_code text)
RETURNS boolean LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$
DECLARE v_claim public.stylist6_request_claims%ROWTYPE; v_refunded boolean := false; v_count integer;
BEGIN
  IF p_user IS NULL OR p_request_id IS NULL OR p_request_id='' OR length(p_request_id)>128 OR
     p_status IS NULL OR p_status < 200 OR p_status > 599 OR length(coalesce(p_code,''))>160 THEN
    RAISE EXCEPTION 'invalid_stylist6_settlement' USING errcode='22023';
  END IF;
  SELECT * INTO v_claim FROM public.stylist6_request_claims
    WHERE user_id=p_user AND request_id=p_request_id FOR UPDATE;
  IF NOT FOUND THEN RAISE EXCEPTION 'stylist6_claim_missing' USING errcode='22023'; END IF;
  IF v_claim.status_code<>0 THEN RETURN v_claim.quota_refunded; END IF;
  IF p_status >= 400 AND v_claim.quota_debited AND NOT v_claim.quota_refunded THEN
    UPDATE public.usage_counters SET used=used-1,updated_at=now()
    WHERE user_id=p_user AND feature='edge:aiOutfits' AND day=v_claim.quota_day AND used>0;
    GET DIAGNOSTICS v_count=ROW_COUNT;
    IF v_count<>1 THEN RAISE EXCEPTION 'stylist6_quota_refund_counter_missing' USING errcode='22023'; END IF;
    v_refunded:=true;
  END IF;
  UPDATE public.stylist6_request_claims SET status_code=p_status,
    outcome_code=coalesce(p_code,''),quota_refunded=v_refunded,updated_at=now()
  WHERE user_id=p_user AND request_id=p_request_id;
  RETURN v_refunded;
END;
$$;

-- Manual/operational recovery: a Vercel SIGKILL cannot settle a claim itself.
-- Only claims older than 180 seconds are considered; the Node runtime has
-- a much shorter execution deadline. This procedure is repeat-safe.
CREATE OR REPLACE FUNCTION public.recover_interrupted_stylist6_requests(p_limit integer DEFAULT 20)
RETURNS integer LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$
DECLARE v_claim public.stylist6_request_claims%ROWTYPE; v_count integer; v_total integer:=0;
BEGIN
  IF p_limit IS NULL OR p_limit<1 OR p_limit>100 THEN
    RAISE EXCEPTION 'invalid_stylist6_recovery_limit' USING errcode='22023';
  END IF;
  FOR v_claim IN SELECT * FROM public.stylist6_request_claims
    WHERE status_code=0 AND quota_debited AND NOT quota_refunded
      AND updated_at<now()-interval '180 seconds'
    ORDER BY updated_at ASC LIMIT p_limit FOR UPDATE SKIP LOCKED
  LOOP
    UPDATE public.usage_counters SET used=used-1,updated_at=now()
      WHERE user_id=v_claim.user_id AND feature='edge:aiOutfits'
        AND day=v_claim.quota_day AND used>0;
    GET DIAGNOSTICS v_count=ROW_COUNT;
    IF v_count<>1 THEN RAISE EXCEPTION 'stylist6_stale_refund_counter_missing' USING errcode='22023'; END IF;
    UPDATE public.stylist6_request_claims
      SET status_code=504,outcome_code='stylist6_interrupted_refunded',
        quota_refunded=true,updated_at=now()
      WHERE user_id=v_claim.user_id AND request_id=v_claim.request_id;
    v_total:=v_total+1;
  END LOOP;
  RETURN v_total;
END;
$$;

REVOKE ALL ON FUNCTION public.charge_stylist6_request(uuid,text) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.settle_stylist6_request(uuid,text,integer,text) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.recover_interrupted_stylist6_requests(integer) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.charge_stylist6_request(uuid,text) TO service_role;
GRANT EXECUTE ON FUNCTION public.settle_stylist6_request(uuid,text,integer,text) TO service_role;
GRANT EXECUTE ON FUNCTION public.recover_interrupted_stylist6_requests(integer) TO service_role;
NOTIFY pgrst,'reload schema';
COMMIT;
