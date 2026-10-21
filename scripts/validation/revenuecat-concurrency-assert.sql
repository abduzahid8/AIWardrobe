do $$
declare n integer;
begin
  select count(*) into n from public.processed_webhooks where source = 'revenuecat' and event_id = 'local-rc-concurrent';
  if n <> 1 then raise exception 'event count is %', n; end if;
  select count(*) into n from public.payments where user_id = md5('load-user-8')::uuid and transaction_id = 'local-rc-concurrent-txn';
  if n <> 1 then raise exception 'payment count is %', n; end if;
  select count(*) into n from public.subscriptions where user_id = md5('load-user-8')::uuid and revenuecat_managed;
  if n <> 1 then raise exception 'subscription count is %', n; end if;
end;
$$;
