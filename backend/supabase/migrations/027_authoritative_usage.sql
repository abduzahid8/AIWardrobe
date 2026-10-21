-- Daily limits must come from server entitlements, never client-supplied p_limit.
begin;
create or replace function public.effective_subscription_tier(p_user uuid)
returns text language sql stable security definer set search_path = '' as $$
    select coalesce(
        (select s.tier from public.subscriptions s
         where s.user_id = p_user and s.tier in ('lite', 'premium', 'vip')
           and s.status in ('active', 'trial', 'cancelled')
           and (s.status <> 'cancelled' or s.end_date > now())
           and (s.end_date is null or s.end_date > now())
           and (s.status <> 'trial' or s.trial_end_date is null or s.trial_end_date > now())
         order by case s.tier when 'vip' then 3 when 'premium' then 2 else 1 end desc limit 1),
        (select 'premium' from public.profiles p where p.id = p_user
           and p.trial_started_at <= now() and p.trial_started_at > now() - interval '7 days'),
        'free');
$$;
revoke execute on function public.effective_subscription_tier(uuid) from public, anon, authenticated;
grant execute on function public.effective_subscription_tier(uuid) to service_role;

create or replace function public.check_and_increment_usage(
    p_feature text, p_limit integer, p_amount integer default 1
) returns table (allowed boolean, used integer, remaining integer)
language plpgsql security definer set search_path = ''
as $$
declare
    v_user uuid := auth.uid();
    v_day date := (now() at time zone 'utc')::date;
    v_tier text := 'free';
    v_limit integer;
    v_new integer;
begin
    if v_user is null then raise exception 'auth required' using errcode = '42501'; end if;
    if p_amount is null or p_amount < 1 or p_amount > 100 then
        raise exception 'invalid usage amount' using errcode = '22023';
    end if;
    if p_feature is null or p_feature not in ('aiOutfits', 'tryOns', 'wardrobeScans', 'homeSwipes') then
        raise exception 'invalid quota feature' using errcode = '22023';
    end if;
    v_tier := public.effective_subscription_tier(v_user);
    v_limit := case
      when v_tier in ('premium', 'vip') then -1
      when v_tier = 'lite' then case p_feature when 'tryOns' then 0 when 'wardrobeScans' then 30 else -1 end
      else case p_feature when 'homeSwipes' then 3 when 'wardrobeScans' then 5 else 0 end
    end;
    -- Keep the legacy parameter for client compatibility; it grants no authority.
    if v_limit = 0 or (v_limit <> -1 and p_amount > v_limit) then
        select c.used into v_new from public.usage_counters c
        where c.user_id = v_user and c.feature = p_feature and c.day = v_day;
        return query select false, coalesce(v_new, 0), greatest(0, v_limit - coalesce(v_new, 0));
        return;
    end if;
    insert into public.usage_counters as counters(user_id, feature, day, used)
    values (v_user, p_feature, v_day, p_amount)
    on conflict (user_id, feature, day) do update
      set used = counters.used + p_amount, updated_at = now()
      where v_limit = -1 or counters.used + p_amount <= v_limit
    returning counters.used into v_new;
    if v_new is null then
        select c.used into v_new from public.usage_counters c
        where c.user_id = v_user and c.feature = p_feature and c.day = v_day;
        return query select false, coalesce(v_new, 0), greatest(0, v_limit - coalesce(v_new, 0));
        return;
    end if;
    return query select true, v_new, case when v_limit = -1 then -1 else greatest(0, v_limit - v_new) end;
end;
$$;
revoke execute on function public.check_and_increment_usage(text, integer, integer) from public, anon;
grant execute on function public.check_and_increment_usage(text, integer, integer) to authenticated;

-- Independently meter actual paid endpoint work. Client UI reservations remain
-- separate, so a normal client is not double-charged, and skipping the UI RPC
-- cannot bypass the endpoint's own cap.
create or replace function public.consume_edge_quota(p_user uuid, p_endpoint text)
returns boolean language plpgsql security definer set search_path = '' as $$
declare
    v_tier text := public.effective_subscription_tier(p_user);
    v_feature text;
    v_limit integer;
    v_used integer;
begin
    if p_user is null then return false; end if;
    if p_endpoint in ('ai-chat', 'analyze-outfit') then return v_tier <> 'free'; end if;
    if p_endpoint = 'create-trip-plan' then return v_tier in ('premium', 'vip'); end if;
    v_feature := case p_endpoint when 'ai-process' then 'wardrobeScans'
      when 'generate-outfits' then 'aiOutfits' when 'try-on' then 'tryOns' end;
    if v_feature is null then return false; end if;
    v_limit := case when v_tier in ('premium', 'vip') then -1
      when v_tier = 'lite' then case v_feature when 'tryOns' then 0 when 'wardrobeScans' then 30 else -1 end
      else case v_feature when 'wardrobeScans' then 5 else 0 end end;
    if v_limit = 0 then return false; end if;
    insert into public.usage_counters as counters(user_id, feature, day, used)
    values(p_user, 'edge:' || v_feature, (now() at time zone 'utc')::date, 1)
    on conflict (user_id, feature, day) do update
      set used = counters.used + 1, updated_at = now()
      where v_limit = -1 or counters.used < v_limit
    returning counters.used into v_used;
    return v_used is not null;
end;
$$;
revoke execute on function public.consume_edge_quota(uuid, text) from public, anon, authenticated;
grant execute on function public.consume_edge_quota(uuid, text) to service_role;

-- Rate-limit buckets otherwise accumulate indefinitely, one row per active minute.
create or replace function public.prune_usage_counters()
returns void language sql security definer set search_path = '' as $$
    delete from public.usage_counters where day < (now() at time zone 'utc')::date - 2;
$$;
revoke execute on function public.prune_usage_counters() from public, anon, authenticated;
grant execute on function public.prune_usage_counters() to service_role;
revoke execute on function public.prune_processed_webhooks() from public, anon, authenticated;
grant execute on function public.prune_processed_webhooks() to service_role;
commit;
