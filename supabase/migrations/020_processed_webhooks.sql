-- 017_processed_webhooks.sql
-- Idempotency ledger for inbound webhooks (RevenueCat, etc.).
-- A webhook is processed at most once per (source, event_id).

CREATE TABLE IF NOT EXISTS public.processed_webhooks (
    source TEXT NOT NULL,
    event_id TEXT NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source, event_id)
);

ALTER TABLE public.processed_webhooks ENABLE ROW LEVEL SECURITY;

-- Service role only. No client read or write.
DROP POLICY IF EXISTS processed_webhooks_service_only ON public.processed_webhooks;
CREATE POLICY processed_webhooks_service_only
    ON public.processed_webhooks
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Auto-prune events older than 30 days. Run by a cron job or manually.
-- Kept as a function so an external scheduler can call it.
CREATE OR REPLACE FUNCTION public.prune_processed_webhooks()
RETURNS void
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    DELETE FROM public.processed_webhooks
    WHERE processed_at < now() - INTERVAL '30 days';
$$;
