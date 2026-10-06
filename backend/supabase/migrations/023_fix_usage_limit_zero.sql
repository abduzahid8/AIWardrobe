-- 023_fix_usage_limit_zero.sql
-- Harden check_and_increment_usage against limit = 0.
--
-- Problem: with p_limit = 0 (e.g. free-tier aiOutfits post-trial) the first
-- call of the day performed a bare INSERT (no conflicting row yet), so the
-- ON CONFLICT ... WHERE guard never fired and the call wrongly returned
-- allowed = true with used = 1. The client gate (checkFeatureAccess) already
-- short-circuits limit-0 features before reaching the RPC, but the server
-- must enforce it too so direct callers cannot sneak one free use.
--
-- Fix: deny immediately when p_limit = 0 (reporting current usage).

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
    v_cur  integer;
begin
    if v_user is null then
        raise exception 'auth required' using errcode = '42501';
    end if;

    -- Zero (or negative, non-unlimited) limit → always deny, no write.
    if p_limit = 0 or (p_limit < 0 and p_limit <> -1) then
        select used into v_cur
          from public.usage_counters
         where user_id = v_user and feature = p_feature and day = v_day;
        return query select false, coalesce(v_cur, 0), 0;
        return;
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

    -- First insert of the day bypassed the WHERE guard — re-check the limit
    -- so a zero/over-limit first write can never report allowed = true.
    if v_new > p_limit then
        return query select false, v_new, 0;
    end if;

    return query select true, v_new, greatest(0, p_limit - v_new);
end;
$$;
