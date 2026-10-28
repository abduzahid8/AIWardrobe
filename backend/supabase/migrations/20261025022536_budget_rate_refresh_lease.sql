-- Server-only, fenced refresh coordination. No transaction spans an HTTP call.
set lock_timeout='3s';
set statement_timeout='15s';
create table public.budget_rate_refresh (
 singleton boolean primary key default true check(singleton),
 token uuid,
 lease_until timestamptz not null default '-infinity',
 failed_until timestamptz not null default '-infinity'
);
alter table public.budget_rate_refresh enable row level security;
revoke all on public.budget_rate_refresh from public,anon,authenticated;
grant select,insert,update on public.budget_rate_refresh to service_role;

create function public.claim_budget_rate_refresh(p_token uuid) returns text
language plpgsql security invoker set search_path='' as $$
declare state public.budget_rate_refresh%rowtype;
begin
 if p_token is null then raise exception 'refresh token required' using errcode='22023'; end if;
 insert into public.budget_rate_refresh(singleton) values(true) on conflict do nothing;
 select * into state from public.budget_rate_refresh where singleton for update;
 if state.failed_until>clock_timestamp() then return 'cooldown'; end if;
 if state.lease_until>clock_timestamp() then return 'busy'; end if;
 update public.budget_rate_refresh set token=p_token,
  lease_until=clock_timestamp()+interval '15 seconds',failed_until='-infinity' where singleton;
 return 'acquired';
end $$;

create function public.finish_budget_rate_refresh(p_token uuid,p_snapshot jsonb default null) returns boolean
language plpgsql security invoker set search_path='' as $$
declare state public.budget_rate_refresh%rowtype;
begin
 select * into state from public.budget_rate_refresh where singleton for update;
 if not found or p_token is null or state.token is distinct from p_token
  or state.lease_until<=clock_timestamp() then return false; end if;
 if p_snapshot is not null then
  insert into public.budget_exchange_rates(id,as_of,fetched_at,snapshot)
   values(p_snapshot->>'id',(p_snapshot->>'asOf')::date,(p_snapshot->>'fetchedAt')::timestamptz,p_snapshot)
   on conflict(id) do update set as_of=excluded.as_of,fetched_at=excluded.fetched_at,snapshot=excluded.snapshot
   where excluded.fetched_at>=public.budget_exchange_rates.fetched_at;
 end if;
 update public.budget_rate_refresh set token=null,lease_until='-infinity',
  failed_until=case when p_snapshot is null then clock_timestamp()+interval '10 seconds' else '-infinity'::timestamptz end
  where singleton;
 return true;
end $$;

revoke all on function public.claim_budget_rate_refresh(uuid) from public,anon,authenticated;
revoke all on function public.finish_budget_rate_refresh(uuid,jsonb) from public,anon,authenticated;
grant execute on function public.claim_budget_rate_refresh(uuid) to service_role;
grant execute on function public.finish_budget_rate_refresh(uuid,jsonb) to service_role;
