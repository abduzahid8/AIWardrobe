-- Keep admission counters bounded without one large delete transaction.
set lock_timeout = '3s';
create index if not exists idx_usage_counters_retention_day
    on public.usage_counters(day);

create or replace function public.prune_usage_counters()
returns void language sql security definer set search_path = '' as $$
    with expired as (
        select ctid from public.usage_counters
        where day < (now() at time zone 'utc')::date - 2
        order by day
        limit 10000
        for update skip locked
    )
    delete from public.usage_counters counters using expired
    where counters.ctid = expired.ctid;
$$;
revoke execute on function public.prune_usage_counters() from public, anon, authenticated;
grant execute on function public.prune_usage_counters() to service_role;

-- Hosted Supabase supplies pg_cron. Plain Postgres validation installations
-- lacking that module still validate the table/function; scheduled execution
-- must be verified separately on the hosted project, never inferred locally.
do $install$
begin
    if exists (select 1 from pg_available_extensions where name = 'pg_cron') then
        create extension if not exists pg_cron with schema pg_catalog;
        grant usage on schema cron to postgres;
        grant all privileges on all tables in schema cron to postgres;
        perform cron.schedule(
            'aiwardrobe-usage-retention', '* * * * *',
            $job$
                set statement_timeout = '10s';
                select public.prune_usage_counters();
                delete from cron.job_run_details
                where jobid = (select jobid from cron.job where jobname = 'aiwardrobe-usage-retention')
                  and end_time < now() - interval '7 days';
            $job$
        );
    else
        raise notice 'pg_cron is unavailable here: hosted scheduler verification is required';
    end if;
end;
$install$;
reset lock_timeout;
