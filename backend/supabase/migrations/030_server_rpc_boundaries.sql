-- Hosted ACLs can contain explicit role grants in addition to PUBLIC grants.
begin;
set local lock_timeout = '5s';
set local statement_timeout = '30s';

revoke execute on function public.increment_rate_bucket(uuid, text) from public, anon, authenticated;
grant execute on function public.increment_rate_bucket(uuid, text) to service_role;
revoke execute on function public.prune_processed_webhooks() from public, anon, authenticated;
grant execute on function public.prune_processed_webhooks() to service_role;

-- These handlers accept arbitrary account identifiers and belong behind a
-- trusted auth service; the mobile app signs in through Supabase Auth directly.
-- These legacy handlers exist on the hosted project but were not created by
-- the historical migrations. Harden them when present without breaking a
-- clean installation that relies exclusively on Supabase Auth.
do $$
begin
  if to_regprocedure('public.handle_failed_login(text)') is not null then
    revoke execute on function public.handle_failed_login(text) from public, anon, authenticated;
    grant execute on function public.handle_failed_login(text) to service_role;
    alter function public.handle_failed_login(text) set search_path = public;
  end if;
  if to_regprocedure('public.handle_successful_login(text,text)') is not null then
    revoke execute on function public.handle_successful_login(text, text) from public, anon, authenticated;
    grant execute on function public.handle_successful_login(text, text) to service_role;
    alter function public.handle_successful_login(text, text) set search_path = public;
  end if;
end;
$$;

-- Owner-scoped usage reads/writes remain available after sign-in.
revoke execute on function public.check_and_increment_usage(text, integer, integer) from public, anon;
grant execute on function public.check_and_increment_usage(text, integer, integer) to authenticated, service_role;
revoke execute on function public.get_usage_today(text) from public, anon;
grant execute on function public.get_usage_today(text) to authenticated, service_role;

-- Analytics queries must honor the caller's underlying brand_clicks policies.
alter view public.brand_click_stats set (security_invoker = true);
commit;
