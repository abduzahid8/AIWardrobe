-- Atomicity, retry, stale delivery, payment amount and transfer assertions.
begin;
set local role service_role;
select set_config('request.jwt.claim.role', 'service_role', true);
do $$
declare
  owner uuid := md5('load-user-6')::uuid;
  destination uuid := md5('load-user-7')::uuid;
  stamp bigint := (extract(epoch from now()) * 1000)::bigint;
  account jsonb; event jsonb; result jsonb; n integer;
begin
  account := jsonb_build_object('user_id', owner, 'request_date_ms', stamp, 'entitlements', jsonb_build_array(
    jsonb_build_object('tier', 'premium', 'product_id', 'com.aiwardrobe.premium.monthly', 'status', 'active',
      'start_date', now(), 'end_date', now() + interval '30 days', 'auto_renew', true, 'platform', 'apple')));
  event := jsonb_build_object('id', 'local-rc-initial', 'type', 'INITIAL_PURCHASE', 'resolved_user_id', owner,
    'product_id', 'com.aiwardrobe.premium.monthly', 'transaction_id', 'local-txn', 'price', 9.99,
    'price_in_purchased_currency', 129000, 'currency', 'UZS', 'store', 'APP_STORE');
  result := public.apply_revenuecat_snapshots(event, jsonb_build_array(account));
  if (result->>'duplicate')::boolean then raise exception 'first event was a duplicate'; end if;
  if public.effective_subscription_tier(owner) <> 'premium' then raise exception 'purchase did not grant verified tier'; end if;
  if not exists(select 1 from public.payments where user_id = owner and transaction_id = 'local-txn' and amount = 129000 and currency = 'UZS') then
    raise exception 'local-currency payment was recorded inaccurately';
  end if;
  result := public.apply_revenuecat_snapshots(event, jsonb_build_array(account));
  if not (result->>'duplicate')::boolean then raise exception 'duplicate event was applied twice'; end if;
  event := event || jsonb_build_object('id', 'local-rc-redelivery-new-id');
  perform public.apply_revenuecat_snapshots(event, jsonb_build_array(account));
  select count(*) into n from public.payments where user_id = owner and transaction_id = 'local-txn';
  if n <> 1 then raise exception 'same payment transaction recorded twice'; end if;

  -- A stale empty snapshot cannot revoke a newer verified purchase.
  perform public.apply_revenuecat_snapshots(jsonb_build_object('id', 'local-rc-stale', 'type', 'EXPIRATION'),
    jsonb_build_array(jsonb_build_object('user_id', owner, 'request_date_ms', stamp - 1, 'entitlements', '[]'::jsonb)));
  if public.effective_subscription_tier(owner) <> 'premium' then raise exception 'stale delivery revoked current access'; end if;

  -- Failure after the ledger insert must roll back both the ledger and writes.
  begin
    perform public.apply_revenuecat_snapshots(jsonb_build_object('id', 'local-rc-retry', 'type', 'RENEWAL'),
      jsonb_build_array(jsonb_build_object('user_id', owner, 'request_date_ms', stamp + 1, 'entitlements',
        jsonb_build_array(jsonb_build_object('tier', 'invalid', 'product_id', 'bad', 'status', 'active')))));
    raise exception 'invalid snapshot was accepted';
  exception when invalid_parameter_value then null; end;
  if exists(select 1 from public.processed_webhooks where source = 'revenuecat' and event_id = 'local-rc-retry') then
    raise exception 'failed delivery poisoned the retry ledger';
  end if;
  if public.effective_subscription_tier(owner) <> 'premium' then raise exception 'failed delivery partially revoked access'; end if;
  account := account || jsonb_build_object('request_date_ms', stamp + 2);
  result := public.apply_revenuecat_snapshots(jsonb_build_object('id', 'local-rc-retry', 'type', 'RENEWAL'), jsonb_build_array(account));
  if (result->>'duplicate')::boolean then raise exception 'corrected retry was ignored'; end if;

  -- Both sides of a transfer commit together from independently fetched snapshots.
  perform public.apply_revenuecat_snapshots(jsonb_build_object('id', 'local-rc-transfer', 'type', 'TRANSFER'),
    jsonb_build_array(
      jsonb_build_object('user_id', owner, 'request_date_ms', stamp + 3, 'entitlements', '[]'::jsonb),
      jsonb_build_object('user_id', destination, 'request_date_ms', stamp + 3, 'entitlements', account->'entitlements')));
  if public.effective_subscription_tier(owner) <> 'free' or public.effective_subscription_tier(destination) <> 'premium' then
    raise exception 'transfer ownership did not reconcile';
  end if;
  perform public.apply_revenuecat_snapshots(jsonb_build_object('id', 'local-rc-transfer-refund', 'type', 'CANCELLATION',
    'cancel_reason', 'CUSTOMER_SUPPORT', 'transaction_id', 'local-txn', 'store', 'APP_STORE'),
    jsonb_build_array(jsonb_build_object('user_id', destination, 'request_date_ms', stamp + 4, 'entitlements', '[]'::jsonb)));
  if not exists(select 1 from public.payments where user_id = owner and transaction_id = 'local-txn' and status = 'refunded') then
    raise exception 'refund after transfer did not update the original payment';
  end if;
end;
$$;
set local role authenticated;
select set_config('request.jwt.claim.role', 'authenticated', true);
do $$ begin
  begin
    perform public.apply_revenuecat_snapshots('{}', '[]');
    raise exception 'client could call subscription sync RPC';
  exception when insufficient_privilege then null; end;
end; $$;
rollback;
