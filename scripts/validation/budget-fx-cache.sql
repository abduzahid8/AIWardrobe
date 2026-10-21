-- Public reference data is readable, never writable, by app users.
begin;
set local lock_timeout='3s';set local statement_timeout='15s';
select set_config('readiness.fx_id','fx_'||gen_random_uuid()::text,true);
insert into public.budget_exchange_rates(id,as_of,fetched_at,snapshot)
values(current_setting('readiness.fx_id'),'2000-01-01','2000-01-01T00:00:00Z',jsonb_build_object('id',current_setting('readiness.fx_id'),'source','CBU','asOf','2000-01-01','fetchedAt','2000-01-01T00:00:00Z','uzsPerUnit',jsonb_build_object('USD',100,'EUR',200,'GBP',300,'UZS',1)));
set local role authenticated;
do $$begin
 if (select count(*) from public.budget_exchange_rates where id=current_setting('readiness.fx_id'))<>1 then raise exception 'Authenticated cache read failed';end if;
 begin update public.budget_exchange_rates set snapshot='{}' where id=current_setting('readiness.fx_id');raise exception 'App user edited rates';exception when insufficient_privilege then null;end;
 begin delete from public.budget_exchange_rates where id=current_setting('readiness.fx_id');raise exception 'App user deleted rates';exception when insufficient_privilege then null;end;
 begin insert into public.budget_exchange_rates(id,as_of,fetched_at,snapshot) values('forged','2000-01-01',now(),'{}');raise exception 'App user inserted rates';exception when insufficient_privilege then null;end;
end $$;
reset role;
set local role anon;
do $$begin
 begin perform 1 from public.budget_exchange_rates;raise exception 'Anonymous table access allowed';exception when insufficient_privilege then null;end;
end $$;
reset role;
set local role service_role;
do $$begin
 update public.budget_exchange_rates set fetched_at='2000-01-02T00:00:00Z',snapshot=jsonb_set(snapshot,'{fetchedAt}','"2000-01-02T00:00:00Z"') where id=current_setting('readiness.fx_id');
 if not found then raise exception 'Service cache refresh failed';end if;
 begin insert into public.budget_exchange_rates(id,as_of,fetched_at,snapshot) values('invalid','2000-01-01',now(),'{}');raise exception 'Malformed rate row accepted';exception when check_violation then null;end;
end $$;
reset role;
select jsonb_build_object('passed',true,'scope','Authenticated read, client/anonymous write restrictions, service refresh and malformed-row constraint; rolled back') as budget_fx_cache_checks;
rollback;
