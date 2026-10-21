-- Reconcile verified Customer Info and acknowledge the webhook in one transaction.
begin;
alter table public.subscriptions add column if not exists revenuecat_managed boolean not null default false;
update public.subscriptions set revenuecat_managed = true
where platform in ('apple', 'google', 'ios', 'android', 'stripe')
  and product_id in ('2.99', 'com.aiwardrobe.lite.monthly', 'com.aiwardrobe.premium.monthly', 'com.aiwardrobe.premium.yearly');
create table if not exists public.revenuecat_sync_state (
  user_id uuid primary key references public.profiles(id) on delete cascade,
  snapshot_ms bigint not null
);
alter table public.revenuecat_sync_state enable row level security;
revoke all on public.revenuecat_sync_state from anon, authenticated;

create or replace function public.apply_revenuecat_snapshots(p_event jsonb, p_accounts jsonb)
returns jsonb language plpgsql security definer set search_path = '' as $$
declare
  account jsonb; entitlement jsonb; owner uuid; snapshot_ms bigint; previous_ms bigint;
  subscription_id uuid; resolved_tier text; resolved_expiry timestamptz;
  event_id text := p_event->>'id'; txn text := p_event->>'transaction_id';
  event_owner uuid; payment_currency text; payment_amount numeric;
  payment_platform text := case upper(p_event->>'store') when 'APP_STORE' then 'apple' when 'PLAY_STORE' then 'google' when 'STRIPE' then 'stripe' else 'revenuecat' end;
begin
  if event_id is null or event_id = '' or jsonb_typeof(p_accounts) <> 'array' or jsonb_array_length(p_accounts) = 0 then
    raise exception 'invalid RevenueCat sync payload' using errcode = '22023';
  end if;
  -- Stable lock order serializes per-owner syncs and prevents transfer deadlocks.
  for owner in select distinct (value->>'user_id')::uuid from jsonb_array_elements(p_accounts) order by 1 loop
    perform 1 from public.profiles where id = owner for update;
    if not found then raise exception 'RevenueCat owner not found' using errcode = '23503'; end if;
  end loop;
  insert into public.processed_webhooks(source, event_id) values ('revenuecat', event_id) on conflict do nothing;
  if not found then return jsonb_build_object('duplicate', true); end if;

  for account in select value from jsonb_array_elements(p_accounts) loop
    owner := (account->>'user_id')::uuid;
    snapshot_ms := (account->>'request_date_ms')::bigint;
    if snapshot_ms is null or snapshot_ms <= 0 or jsonb_typeof(account->'entitlements') <> 'array' then
      raise exception 'invalid RevenueCat snapshot' using errcode = '22023';
    end if;
    select s.snapshot_ms into previous_ms from public.revenuecat_sync_state s where s.user_id = owner;
    if previous_ms is not null and snapshot_ms <= previous_ms then continue; end if;
    update public.subscriptions set status = 'expired', auto_renew = false where user_id = owner and revenuecat_managed;
    for entitlement in select value from jsonb_array_elements(account->'entitlements') loop
      if entitlement->>'tier' not in ('lite', 'premium', 'vip') or coalesce(entitlement->>'product_id', '') = ''
         or entitlement->>'status' not in ('active', 'trial', 'cancelled') then
        raise exception 'invalid verified entitlement' using errcode = '22023';
      end if;
      subscription_id := null;
      select id into subscription_id from public.subscriptions
      where user_id = owner and product_id = entitlement->>'product_id' and revenuecat_managed
      order by created_at desc limit 1;
      if subscription_id is null then
        insert into public.subscriptions(user_id, tier, status, start_date, end_date, trial_end_date, auto_renew, product_id, platform, revenuecat_managed)
        values(owner, entitlement->>'tier', entitlement->>'status', (entitlement->>'start_date')::timestamptz,
          (entitlement->>'end_date')::timestamptz,
          case when entitlement->>'status' = 'trial' then (entitlement->>'end_date')::timestamptz else null end,
          (entitlement->>'auto_renew')::boolean, entitlement->>'product_id', entitlement->>'platform', true);
      else
        update public.subscriptions set tier = entitlement->>'tier', status = entitlement->>'status',
          start_date = (entitlement->>'start_date')::timestamptz, end_date = (entitlement->>'end_date')::timestamptz,
          trial_end_date = case when entitlement->>'status' = 'trial' then (entitlement->>'end_date')::timestamptz else null end,
          auto_renew = (entitlement->>'auto_renew')::boolean, platform = entitlement->>'platform'
        where id = subscription_id;
      end if;
    end loop;
    resolved_tier := null; resolved_expiry := null;
    select s.tier, s.end_date into resolved_tier, resolved_expiry from public.subscriptions s
    where s.user_id = owner and s.tier <> 'free' and s.status in ('active', 'trial', 'cancelled')
      and (s.status <> 'cancelled' or s.end_date > now())
      and (s.end_date is null or s.end_date > now())
      and (s.status <> 'trial' or s.trial_end_date is null or s.trial_end_date > now())
    order by case s.tier when 'vip' then 3 when 'premium' then 2 else 1 end desc, s.end_date desc nulls first limit 1;
    update public.profiles set subscription_tier = coalesce(resolved_tier, 'free'), subscription_expires_at = resolved_expiry where id = owner;
    insert into public.revenuecat_sync_state(user_id, snapshot_ms) values(owner, snapshot_ms)
    on conflict(user_id) do update set snapshot_ms = excluded.snapshot_ms;
  end loop;

  -- An event can be redelivered under a different event ID. Serialize payment
  -- insertion under the owner lock and recognize an existing transaction.
  if p_event->>'type' in ('INITIAL_PURCHASE', 'RENEWAL', 'NON_RENEWING_PURCHASE') and txn is not null then
    event_owner := (p_event->>'resolved_user_id')::uuid;
    if not exists(select 1 from jsonb_array_elements(p_accounts) a where (a->>'user_id')::uuid = event_owner) then
      raise exception 'payment owner not in verified accounts' using errcode = '22023';
    end if;
    select id into subscription_id from public.subscriptions where user_id = event_owner
      and product_id = p_event->>'product_id' and revenuecat_managed order by created_at desc limit 1;
    payment_currency := case when p_event->>'price_in_purchased_currency' is not null then coalesce(p_event->>'currency', 'USD') else 'USD' end;
    payment_amount := coalesce((p_event->>'price_in_purchased_currency')::numeric, (p_event->>'price')::numeric);
    perform pg_advisory_xact_lock(hashtextextended(payment_platform || ':' || txn, 0));
    insert into public.payments(user_id, subscription_id, amount, currency, status, product_id, transaction_id, platform)
    select event_owner, subscription_id, payment_amount, payment_currency, 'completed', p_event->>'product_id', txn, payment_platform
    where not exists(select 1 from public.payments where transaction_id = txn and platform = payment_platform);
  end if;
  if (p_event->>'type' = 'REFUND' or (p_event->>'type' = 'CANCELLATION' and p_event->>'cancel_reason' = 'CUSTOMER_SUPPORT')) and txn is not null then
    perform pg_advisory_xact_lock(hashtextextended(payment_platform || ':' || txn, 0));
    -- A transferred entitlement can be refunded after its original payer changed.
    update public.payments set status = 'refunded' where transaction_id = txn and platform = payment_platform;
  end if;
  return jsonb_build_object('duplicate', false);
end;
$$;
revoke execute on function public.apply_revenuecat_snapshots(jsonb, jsonb) from public, anon, authenticated;
grant execute on function public.apply_revenuecat_snapshots(jsonb, jsonb) to service_role;
commit;
