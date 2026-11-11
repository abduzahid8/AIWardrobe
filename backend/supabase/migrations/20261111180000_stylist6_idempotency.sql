-- Stylist 6 staging request idempotency: durable, atomic uniqueness per
-- (user, logical request) so redelivered generations cannot consume another
-- quota unit. Concurrent duplicates serialize on the primary key: exactly
-- one claim wins, the loser replays the recorded outcome via 409.
--
-- APPLY (owner action; never auto-applied):
--   supabase db push --project-ref fyqpifmrsftsfqibhwhy
-- Until applied, the Stylist 6 cloud handler falls back to a process-local
-- claim table and reports replayProtection:'process-local' on every
-- admitted response. Production billing behavior is unchanged.
begin;
create table if not exists public.stylist6_request_claims (
  user_id uuid not null,
  request_id text not null,
  status_code integer not null default 0,
  outcome_code text not null default '',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (user_id, request_id)
);
create or replace function public.claim_stylist6_request(p_user uuid, p_request_id text)
returns table (duplicate boolean, status_code integer, outcome_code text)
language plpgsql security definer set search_path = '' as $$
declare
  v_status integer;
  v_code text;
begin
  if p_user is null or p_request_id is null or p_request_id = '' then
    raise exception 'invalid claim identity' using errcode = '22023';
  end if;
  insert into public.stylist6_request_claims as claims(user_id, request_id)
  values (p_user, p_request_id)
  on conflict (user_id, request_id) do nothing
  returning 0 into v_status;
  if found then
    return query select false, 0, '';
    return;
  end if;
  select claims.status_code, claims.outcome_code into v_status, v_code
  from public.stylist6_request_claims as claims
  where claims.user_id = p_user and claims.request_id = p_request_id;
  return query select true, coalesce(v_status, 0), coalesce(v_code, '');
end;
$$;
create or replace function public.finish_stylist6_request(
  p_user uuid, p_request_id text, p_status integer, p_code text)
returns void language plpgsql security definer set search_path = '' as $$
begin
  update public.stylist6_request_claims as claims
  set status_code = p_status, outcome_code = coalesce(p_code, ''), updated_at = now()
  where claims.user_id = p_user and claims.request_id = p_request_id;
end;
$$;
revoke execute on function public.claim_stylist6_request(uuid, text) from public, anon, authenticated;
revoke execute on function public.finish_stylist6_request(uuid, text, integer, text) from public, anon, authenticated;
grant execute on function public.claim_stylist6_request(uuid, text) to service_role;
grant execute on function public.finish_stylist6_request(uuid, text, integer, text) to service_role;
commit;
