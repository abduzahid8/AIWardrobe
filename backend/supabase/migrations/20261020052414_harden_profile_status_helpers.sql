-- Legacy status helpers must respect the caller's profile RLS.
-- Missing rows must never be interpreted as paid access.
set local lock_timeout='3s';
set local statement_timeout='30s';
create or replace function public.has_premium_access(user_id uuid)
returns boolean language sql stable security invoker set search_path=''
as $$
 select exists (
  select 1 from public.profiles p where p.id=$1
   and p.subscription_tier in ('premium','vip')
   and (p.subscription_expires_at is null or p.subscription_expires_at>=pg_catalog.now())
 );
$$;
create or replace function public.is_account_locked(user_id uuid)
returns boolean language sql stable security invoker set search_path=''
as $$
 select exists (
  select 1 from public.profiles p where p.id=$1 and p.locked_until>pg_catalog.now()
 );
$$;
revoke execute on function public.has_premium_access(uuid),public.is_account_locked(uuid) from public,anon;
grant execute on function public.has_premium_access(uuid),public.is_account_locked(uuid) to authenticated,service_role;
-- This function is only an auth.users trigger, never a public RPC.
-- Trigger execution remains verified by the rollback fixture test.
alter function public.handle_new_user() set search_path='';
revoke execute on function public.handle_new_user() from public,anon,authenticated;
