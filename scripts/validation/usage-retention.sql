-- Transactional fixtures only; preserve all seeded data and counters.
begin;
delete from public.usage_counters;
insert into public.usage_counters(user_id, feature, day, used)
select (select id from auth.users limit 1), '__retention_audit:' || n,
       (now() at time zone 'utc')::date - 3, 1
from generate_series(1, 10005) n;
insert into public.usage_counters(user_id, feature, day, used)
select (select id from auth.users limit 1), '__retention_kept:' || n,
       (now() at time zone 'utc')::date - n, 3
from generate_series(0, 2) n;
select public.prune_usage_counters();
do $$ begin
 if (select count(*) from public.usage_counters where feature like '__retention_audit:%') <> 5 then
  raise exception 'retention did not honor 10000-row batch';
 end if;
 if (select count(*) from public.usage_counters where feature like '__retention_kept:%' and used=3) <> 3 then
  raise exception 'retention changed current/boundary counters';
 end if;
 if has_function_privilege('authenticated','public.prune_usage_counters()','EXECUTE')
 or has_function_privilege('anon','public.prune_usage_counters()','EXECUTE') then
  raise exception 'clients can prune counters';
 end if;
end $$;
select public.prune_usage_counters();
do $$ begin
 if (select count(*) from public.usage_counters) <> 3 then
  raise exception 'second retention batch failed';
 end if;
end $$;
rollback;
