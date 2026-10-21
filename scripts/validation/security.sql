-- Assertions run as an ordinary authenticated user, inside a rolled-back transaction.
begin;
set local role authenticated;
select set_config('request.jwt.claim.sub', md5('load-user-1')::uuid::text, true);
select set_config('request.jwt.claim.role', 'authenticated', true);
do $$
declare n integer;
begin
    select count(*) into n from public.profiles;
    if n <> 1 then raise exception 'profile RLS failed: %', n; end if;
    select count(*) into n from public.clothing_items where user_id = md5('load-user-2')::uuid;
    if n <> 0 then raise exception 'wardrobe RLS failed'; end if;
    begin
        update public.profiles set is_admin = true where id = auth.uid();
        raise exception 'admin escalation was allowed';
    exception when insufficient_privilege then null; end;
    begin
        update public.profiles set subscription_tier = 'vip' where id = auth.uid();
        raise exception 'tier escalation was allowed';
    exception when insufficient_privilege then null; end;
    begin
        update public.profiles set email = 'info@aiwardrobe.club' where id = auth.uid();
        raise exception 'admin-email spoof was allowed';
    exception when insufficient_privilege then null; end;
    begin
        insert into public.subscriptions(user_id, tier) values(auth.uid(), 'vip');
        raise exception 'subscription fabrication was allowed';
    exception when insufficient_privilege then null; end;
    begin
        perform public.increment_rate_bucket(auth.uid(), 'rl:attack');
        raise exception 'service-only RPC was allowed';
    exception when insufficient_privilege then null; end;
    if (select allowed from public.check_and_increment_usage('aiOutfits', -1, 1)) then
        raise exception 'caller-selected unlimited quota was accepted';
    end if;
    if (select allowed from public.check_and_increment_usage('homeSwipes', 999999, 4)) then
        raise exception 'first-write over-limit amount was accepted';
    end if;
    if not (select allowed from public.check_and_increment_usage('homeSwipes', -1, 3)) then
        raise exception 'legal free quota was rejected';
    end if;
    if (select allowed from public.check_and_increment_usage('homeSwipes', -1, 1)) then
        raise exception 'atomic daily cap was exceeded';
    end if;
    begin
        perform public.check_and_increment_usage('homeSwipes', -1, -10);
        raise exception 'negative usage was accepted';
    exception when invalid_parameter_value then null; end;
    begin
        perform public.consume_edge_quota(auth.uid(), 'try-on');
        raise exception 'service-only admission RPC was allowed';
    exception when insufficient_privilege then null; end;
    update public.profiles set trial_started_at = now() + interval '10 years' where id = auth.uid();
    if (select trial_started_at > now() from public.profiles where id = auth.uid()) then
        raise exception 'future trial start was accepted';
    end if;
    begin
        update public.profiles set trial_started_at = null where id = auth.uid();
        raise exception 'trial reset was accepted';
    exception when insufficient_privilege then null; end;
    update public.profiles set username = 'normal-profile-edit' where id = auth.uid();
end;
$$;
set local role service_role;
select set_config('request.jwt.claim.role', 'service_role', true);
-- Use a different synthetic free account, whose trial has not started.
do $$
declare v_user uuid := md5('load-user-2')::uuid;
begin
    insert into public.subscriptions(user_id, tier, status, end_date)
    values(md5('load-user-5')::uuid, 'premium', 'cancelled', now() + interval '1 day');
    if public.effective_subscription_tier(md5('load-user-5')::uuid) <> 'premium' then
        raise exception 'cancellation revoked a paid, unexpired period';
    end if;
    update public.subscriptions set end_date = now() - interval '1 second' where user_id = md5('load-user-5')::uuid;
    if public.effective_subscription_tier(md5('load-user-5')::uuid) <> 'free' then
        raise exception 'expired cancelled period retained access';
    end if;
    update public.subscriptions set end_date = null where user_id = md5('load-user-5')::uuid;
    if public.effective_subscription_tier(md5('load-user-5')::uuid) <> 'free' then
        raise exception 'cancelled period without expiry granted indefinite access';
    end if;
    if public.consume_edge_quota(v_user, 'generate-outfits') then
        raise exception 'free user bypassed generation entitlement';
    end if;
    for i in 1..5 loop
        if not public.consume_edge_quota(v_user, 'ai-process') then raise exception 'legal scan denied'; end if;
    end loop;
    if public.consume_edge_quota(v_user, 'ai-process') then raise exception 'scan daily cap exceeded'; end if;
end;
$$;
set local role service_role;
insert into public.subscriptions(user_id, tier, status, end_date) values
  (md5('load-user-10')::uuid, 'vip', 'active', now() + interval '1 day'),
  (md5('load-user-10')::uuid, 'lite', 'active', now() + interval '1 year'),
  (md5('load-user-12')::uuid, 'premium', 'active', null);
set local role authenticated;
select set_config('request.jwt.claim.role', 'authenticated', true);
select set_config('request.jwt.claim.sub', md5('load-user-10')::uuid::text, true);
do $$ begin
  if (select tier from public.get_subscription_status()) <> 'vip' then
    raise exception 'status ranked a longer lower tier above Max';
  end if;
end; $$;
select set_config('request.jwt.claim.sub', md5('load-user-11')::uuid::text, true);
do $$ begin
  if (select tier from public.get_subscription_status()) <> 'free' then
    raise exception 'subscription status leaked another owner';
  end if;
end; $$;
select set_config('request.jwt.claim.sub', md5('load-user-12')::uuid::text, true);
do $$ declare status record; begin
  select * into status from public.get_subscription_status();
  if status.tier <> 'premium' or status.expiry_date is not null then
    raise exception 'lifetime entitlement status was fabricated';
  end if;
end; $$;
rollback;
