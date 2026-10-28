-- Run after the migration in the same transaction; caller must ROLLBACK.
do $test$
declare
  owner_a uuid := pg_catalog.gen_random_uuid();
  owner_b uuid := pg_catalog.gen_random_uuid();
  today date := (now() at time zone 'utc')::date;
  result record;
  count_used integer;
begin
  insert into auth.users(id,email,raw_user_meta_data,raw_app_meta_data)
  values(owner_a,'receipt-'||owner_a||'@aiwardrobe.club','{}','{}'),
        (owner_b,'receipt-'||owner_b||'@aiwardrobe.club','{}','{}');
  perform set_config('request.jwt.claim.sub', owner_a::text, true);
  for i in 1..3 loop
    select * into strict result from public.authorize_home_swipe('card-'||i,today);
    assert result.allowed and result.used=i and result.remaining=3-i, 'three free cards';
  end loop;
  select * into strict result from public.authorize_home_swipe('card-4',today);
  assert not result.allowed and result.used=3, 'fourth card denied';
  for i in 1..3 loop
    select * into strict result from public.authorize_home_swipe('card-'||i,today);
    assert result.allowed and result.used=3 and result.remaining=0, 'replay after exhaustion';
  end loop;
  assert (select count(*)=3 from public.home_swipe_receipts where user_id=owner_a), 'only approvals retained';
  -- Failed outer transaction must roll back both receipt and debit.
  perform set_config('request.jwt.claim.sub', owner_b::text, true);
  begin
    perform public.authorize_home_swipe('rolled-back',today);
    raise exception 'rollback probe' using errcode='P0002';
  exception when no_data_found then null;
  end;
  assert not exists(select 1 from public.home_swipe_receipts where user_id=owner_b), 'receipt atomic rollback';
  assert not exists(select 1 from public.usage_counters where user_id=owner_b), 'debit atomic rollback';
  select * into strict result from public.authorize_home_swipe('card-1',today);
  assert result.allowed and result.used=1, 'same key isolated by owner';
  begin
    perform public.authorize_home_swipe('card-1',today-1);
    raise exception 'old day accepted';
  exception when invalid_parameter_value then null;
  end;
  begin
    perform public.authorize_home_swipe('',today);
    raise exception 'empty key accepted';
  exception when invalid_parameter_value then null;
  end;
  begin
    perform public.authorize_home_swipe(repeat('x',513),today);
    raise exception 'oversized key accepted';
  exception when invalid_parameter_value then null;
  end;
  perform set_config('request.jwt.claim.sub','',true);
  begin
    perform public.authorize_home_swipe('card-1',today);
    raise exception 'anonymous accepted';
  exception when insufficient_privilege then null;
  end;
  assert not has_function_privilege('anon','public.authorize_home_swipe(text,date)','execute'), 'anonymous ACL';
  assert has_function_privilege('authenticated','public.authorize_home_swipe(text,date)','execute'), 'authenticated ACL';
  assert not has_table_privilege('authenticated','public.home_swipe_receipts','SELECT,INSERT,UPDATE,DELETE'), 'private receipts';
  assert not has_table_privilege('anon','public.home_swipe_receipts','SELECT,INSERT,UPDATE,DELETE'), 'anonymous receipts';
  assert (select relrowsecurity from pg_class where oid='public.home_swipe_receipts'::regclass), 'RLS enabled';
  insert into public.home_swipe_receipts values(owner_a,today-4,'expired',1);
  perform public.prune_usage_counters();
  assert not exists(select 1 from public.home_swipe_receipts where user_id=owner_a and day=today-4), 'expired receipt retention';
  assert (select count(*)=3 from public.home_swipe_receipts where user_id=owner_a and day=today), 'today retained';
end;
$test$;
select 'home receipt assertions passed; caller rolls back all fixtures and schema' as result;
