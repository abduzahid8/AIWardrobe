-- COALESCE must wrap the scalar subquery: a missing row never evaluates
-- COALESCE(used,0) inside the original SELECT.
set lock_timeout='3s';
set statement_timeout='15s';
create or replace function public.get_usage_today(p_feature text)
returns integer language sql stable security definer set search_path='' as $$
 select coalesce((
  select c.used from public.usage_counters c
  where c.user_id=auth.uid() and c.feature=p_feature
   and c.day=(now() at time zone 'utc')::date
 ),0);
$$;
