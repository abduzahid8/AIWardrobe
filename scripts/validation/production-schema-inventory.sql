-- One result set: the Management API returns only the final SELECT's rows.
select jsonb_build_object(
  'columns', (select jsonb_agg(to_jsonb(c)) from (
    select table_name, column_name, data_type from information_schema.columns
    where table_schema = 'public' and table_name in ('profiles','subscriptions','payments','usage_counters','processed_webhooks')
    order by table_name, ordinal_position
  ) c),
  'policies', (select jsonb_agg(to_jsonb(p)) from (
    select tablename, policyname, cmd, qual, with_check from pg_policies
    where schemaname = 'public' and tablename in ('profiles','subscriptions') order by tablename, policyname
  ) p),
  'functions', (select jsonb_agg(to_jsonb(f)) from (
    select p.proname, pg_get_function_identity_arguments(p.oid) as arguments
    from pg_proc p join pg_namespace n on n.oid = p.pronamespace
    where n.nspname = 'public' and p.proname in ('is_admin','check_and_increment_usage','check_rate_limit','get_subscription_status','apply_revenuecat_snapshots')
  ) f),
  'profile_triggers', (select jsonb_agg(to_jsonb(t)) from (
    select tgname, pg_get_triggerdef(oid) as definition from pg_trigger
    where tgrelid = 'public.profiles'::regclass and not tgisinternal
  ) t),
  'constraints', (select jsonb_agg(to_jsonb(c)) from (
    select conrelid::regclass::text as table_name, conname, pg_get_constraintdef(oid) as definition
    from pg_constraint where conrelid in ('public.profiles'::regclass, 'public.subscriptions'::regclass, 'public.payments'::regclass)
  ) c),
  'profile_tiers', (select jsonb_agg(to_jsonb(p)) from (select subscription_tier as tier, count(*) from public.profiles group by subscription_tier) p),
  'subscription_statuses', (select jsonb_agg(to_jsonb(s)) from (select status, count(*) from public.subscriptions group by status) s)
) as inventory;
