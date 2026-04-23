-- 010_usage_counters.sql
-- Server-authoritative daily quota counters.
--
-- Problem: trial/quota state lives in AsyncStorage; users can reset it by
-- clearing app storage. We move the accounting server-side and treat the
-- client as a read-through cache.
--
-- A `check_and_increment_usage(feature, limit)` RPC is provided so callers
-- cannot increment past the limit. The RPC is SECURITY DEFINER so ordinary
-- users cannot modify `usage_counters` directly.

create extension if not exists "pgcrypto";

create table if not exists public.usage_counters (
    user_id     uuid        not null references auth.users(id) on delete cascade,
    feature     text        not null,
    day         date        not null default (now() at time zone 'utc')::date,
    used        integer     not null default 0,
    updated_at  timestamptz not null default now(),
    primary key (user_id, feature, day)
);

create index if not exists idx_usage_counters_user_day
    on public.usage_counters (user_id, day desc);

alter table public.usage_counters enable row level security;

-- Users can read their own counters; writes must go through the RPC.
drop policy if exists "usage_counters_select_self" on public.usage_counters;
create policy "usage_counters_select_self"
    on public.usage_counters for select
    using (auth.uid() = user_id);

-- No INSERT/UPDATE/DELETE policies for end-users; service role bypasses RLS.

-- Atomic "check and increment" RPC.
-- Returns { allowed: bool, used: int, remaining: int }
-- daily_limit: pass -1 for unlimited; 0 to always deny.
create or replace function public.check_and_increment_usage(
    p_feature   text,
    p_limit     integer,
    p_amount    integer default 1
) returns table (
    allowed   boolean,
    used      integer,
    remaining integer
)
language plpgsql
security definer
set search_path = public
as $$
declare
    v_user uuid := auth.uid();
    v_day  date := (now() at time zone 'utc')::date;
    v_new  integer;
begin
    if v_user is null then
        raise exception 'auth required' using errcode = '42501';
    end if;

    -- Unlimited tier short-circuit — record the usage but always allow.
    if p_limit = -1 then
        insert into public.usage_counters (user_id, feature, day, used)
        values (v_user, p_feature, v_day, p_amount)
        on conflict (user_id, feature, day)
        do update set used = public.usage_counters.used + p_amount,
                      updated_at = now()
        returning used into v_new;

        return query select true, v_new, -1;
        return;
    end if;

    -- Read current, then attempt atomic insert-or-increment guarded by the limit.
    insert into public.usage_counters (user_id, feature, day, used)
    values (v_user, p_feature, v_day, p_amount)
    on conflict (user_id, feature, day)
    do update set used = public.usage_counters.used + p_amount,
                  updated_at = now()
        where public.usage_counters.used + p_amount <= p_limit
    returning used into v_new;

    if v_new is null then
        -- Conflict update blocked by WHERE clause → over limit.
        select used into v_new
          from public.usage_counters
         where user_id = v_user and feature = p_feature and day = v_day;
        return query select false, coalesce(v_new, 0), greatest(0, p_limit - coalesce(v_new, 0));
        return;
    end if;

    return query select true, v_new, greatest(0, p_limit - v_new);
end;
$$;

grant execute on function public.check_and_increment_usage(text, integer, integer) to authenticated;

-- Read-only helper for UI ("you have N left today").
create or replace function public.get_usage_today(p_feature text)
returns integer
language sql
security definer
set search_path = public
stable
as $$
    select coalesce(used, 0)
      from public.usage_counters
     where user_id = auth.uid()
       and feature = p_feature
       and day = (now() at time zone 'utc')::date;
$$;

grant execute on function public.get_usage_today(text) to authenticated;

-- ─────────────────────────────────────────────────────────────
-- Rate-limit bucket counter (used by Edge Functions).
-- Buckets are rolling minute-windows encoded into `feature`.
-- ─────────────────────────────────────────────────────────────
create or replace function public.increment_rate_bucket(
    p_user    uuid,
    p_feature text
) returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
    v_used integer;
    v_day  date := (now() at time zone 'utc')::date;
begin
    insert into public.usage_counters (user_id, feature, day, used)
    values (p_user, p_feature, v_day, 1)
    on conflict (user_id, feature, day)
    do update set used = public.usage_counters.used + 1,
                  updated_at = now()
    returning used into v_used;
    return v_used;
end;
$$;

-- Only the service role (Edge Functions) can call this.
revoke execute on function public.increment_rate_bucket(uuid, text) from public;
revoke execute on function public.increment_rate_bucket(uuid, text) from authenticated;
grant execute on function public.increment_rate_bucket(uuid, text) to service_role;
