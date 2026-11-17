-- The recoverable, owner-scoped results contain short-lived personal outfit
-- data. Remove expired payloads every fifteen minutes, not merely hide them
-- behind the RLS expiry filter.
-- Supabase pg_cron 1.6.x is already installed on this project.
SELECT cron.schedule(
  'aiw-s6-expired-delivery-results',
  '*/15 * * * *',
  $$DELETE FROM public.stylist6_completed_results WHERE expires_at < now()$$
);
