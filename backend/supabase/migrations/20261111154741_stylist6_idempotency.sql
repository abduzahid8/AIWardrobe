-- Stylist 6 staging request idempotency: durable, atomic uniqueness per
-- (user, logical request) so redelivered generations cannot consume another
-- quota unit. Concurrent duplicates serialize on the primary key: exactly
-- one claim wins, the loser replays the recorded outcome via 409.
--
-- Apply only this reviewed migration with the Supabase migration API.
-- Claims and quota debit are separate transactions: a crash can strand a claim
-- without a debit, or preserve a debit without a recorded outcome. Never reclaim
-- or automatically retry an ambiguous claim. This is at-most-once admission,
-- not atomic exactly-once billing or response delivery.
begin;
create table if not exists public.stylist6_request_claims (
  user_id uuid not null,
  request_id text not null,
  request_fingerprint text not null check (request_fingerprint ~ '^[0-9a-f]{64}$'),
  status_code integer not null default 0,
  outcome_code text not null default '',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (user_id, request_id)
);
-- Defense in depth: RLS on, no direct table access for anyone except the
-- server role. All reads/writes go through the SECURITY DEFINER RPCs below,
-- which are executable by service_role only. No policies for anon or
-- regular authenticated users: without an explicit policy every direct
-- row access is denied.
alter table public.stylist6_request_claims enable row level security;
revoke all on public.stylist6_request_claims from public, anon, authenticated;
grant all on public.stylist6_request_claims to service_role;
create or replace function public.claim_stylist6_request(p_user uuid, p_request_id text, p_fingerprint text)
returns table (duplicate boolean, status_code integer, outcome_code text)
language plpgsql security definer set search_path = '' as $$
declare
  v_status integer;
  v_code text;
  v_fingerprint text;
begin
  if p_user is null or p_request_id is null or p_request_id = '' or length(p_request_id) > 128 or
     p_fingerprint is null or p_fingerprint !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid claim identity' using errcode = '22023';
  end if;
  insert into public.stylist6_request_claims as claims(user_id, request_id, request_fingerprint)
  values (p_user, p_request_id, p_fingerprint)
  on conflict (user_id, request_id) do nothing
  returning 0 into v_status;
  if found then
    return query select false, 0, '';
    return;
  end if;
  select claims.status_code, claims.outcome_code, claims.request_fingerprint into v_status, v_code, v_fingerprint
  from public.stylist6_request_claims as claims
  where claims.user_id = p_user and claims.request_id = p_request_id;
  if v_fingerprint <> p_fingerprint then
    return query select true, 409, 'request_content_conflict';
    return;
  end if;
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
revoke execute on function public.claim_stylist6_request(uuid, text, text) from public, anon, authenticated;
revoke execute on function public.finish_stylist6_request(uuid, text, integer, text) from public, anon, authenticated;
grant execute on function public.claim_stylist6_request(uuid, text, text) to service_role;
grant execute on function public.finish_stylist6_request(uuid, text, integer, text) to service_role;
commit;
