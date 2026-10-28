-- Run as the database administrator. All fixtures and shadow objects roll back.
begin;
set local lock_timeout='2s';
set local statement_timeout='10s';
create schema readiness_path_audit_20261025;
create function readiness_path_audit_20261025.now() returns timestamptz language sql as $$select '2000-01-01'::timestamptz$$;
create temp table readiness_timestamp_probe(updated_at timestamptz);
insert into readiness_timestamp_probe values('2000-01-01');
set local search_path=readiness_path_audit_20261025,pg_catalog,public;
do $audit$
declare f text;
begin
 if now()<>'2000-01-01'::timestamptz then raise exception 'shadow-path fixture inactive'; end if;
 foreach f in array array['featured_capsules_set_updated_at','update_updated_at_column','set_updated_at','update_guide_page_updated_at','update_home_occasions_updated_at'] loop
  execute format('create trigger probe before update on pg_temp.readiness_timestamp_probe for each row execute function public.%I()',f);
  update pg_temp.readiness_timestamp_probe set updated_at='2000-01-01';
  if (select updated_at from pg_temp.readiness_timestamp_probe) is distinct from pg_catalog.transaction_timestamp() then
   raise exception 'timestamp trigger failed: %',f;
  end if;
  drop trigger probe on pg_temp.readiness_timestamp_probe;
 end loop;
end $audit$;
set local role anon;
do $audit$
declare result uuid;
begin
 result=public.record_brand_click('readiness-search-path-4b7c19a3','READINESS_AUDIT','Scoped SQL fixture',1,'USD','audit','audit');
 if result is null then raise exception 'click UUID missing'; end if;
end $audit$;
reset role;
do $audit$
begin
 if (select count(*) from public.brand_clicks where item_id='readiness-search-path-4b7c19a3' and brand='READINESS_AUDIT' and product_name='Scoped SQL fixture' and price=1 and currency='USD' and source='audit' and device_type='audit')<>1 then raise exception 'click row mismatch'; end if;
end $audit$;
rollback;
select not exists(select 1 from public.brand_clicks where item_id='readiness-search-path-4b7c19a3') as click_fixture_absent,
 not exists(select 1 from pg_namespace where nspname='readiness_path_audit_20261025') as shadow_schema_absent;
