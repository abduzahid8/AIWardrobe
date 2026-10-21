CREATE OR REPLACE FUNCTION public.check_and_increment_usage(p_feature text, p_limit integer, p_amount integer DEFAULT 1)
 RETURNS TABLE(allowed boolean, used integer, remaining integer)
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public'
AS $function$
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
$function$
