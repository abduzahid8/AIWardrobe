-- Public reference data only. Clients cannot set rates used for budget decisions.
set lock_timeout='3s';
set statement_timeout='15s';
create table public.budget_exchange_rates (
 id text primary key,
 as_of date not null,
 fetched_at timestamptz not null,
 snapshot jsonb not null,
 constraint budget_exchange_rates_shape check (coalesce(
   jsonb_typeof(snapshot)='object' and snapshot->>'source'='CBU' and snapshot->>'id'=id
   and snapshot->>'asOf'=as_of::text
   and jsonb_typeof(snapshot->'uzsPerUnit')='object'
   and (snapshot->>'fetchedAt')::timestamptz=fetched_at
   and jsonb_typeof(snapshot#>'{uzsPerUnit,USD}')='number'
   and jsonb_typeof(snapshot#>'{uzsPerUnit,EUR}')='number'
   and jsonb_typeof(snapshot#>'{uzsPerUnit,GBP}')='number'
   and (snapshot#>>'{uzsPerUnit,USD}')::numeric > 0
   and (snapshot#>>'{uzsPerUnit,EUR}')::numeric > 0
   and (snapshot#>>'{uzsPerUnit,GBP}')::numeric > 0
   and snapshot#>'{uzsPerUnit,UZS}'='1'::jsonb
 ,false))
);
alter table public.budget_exchange_rates enable row level security;
revoke all on public.budget_exchange_rates from public,anon,authenticated;
grant select on public.budget_exchange_rates to authenticated;
grant select,insert,update,delete on public.budget_exchange_rates to service_role;
create policy budget_rates_read on public.budget_exchange_rates for select to authenticated using(true);
create index budget_exchange_rates_latest_idx on public.budget_exchange_rates(fetched_at desc);
