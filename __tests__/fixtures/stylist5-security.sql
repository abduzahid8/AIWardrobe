\set ON_ERROR_STOP on
begin;
set local role authenticated;
do $$ begin
 if has_table_privilege(current_user,'public.catalog_stylist_capsule','SELECT')
    or has_table_privilege(current_user,'public.catalog_stylist_capsule','INSERT')
    or has_table_privilege(current_user,'public.catalog_image_embeddings','SELECT')
    or has_function_privilege(current_user,'public.catalog_stylist_candidates(text[],text,jsonb,integer,integer,text,numeric)','EXECUTE')
 then raise exception 'private evidence privilege escalation';end if;
end $$;
reset role;
set local role service_role;
do $$ declare n integer; begin
 select count(*) into n from public.catalog_stylist_candidates(array['top','outerwear','bottom','shoes'],'security-v5','{"engine":"stylist5","concepts":[{"roles":{"top":["shirt"],"outerwear":["cardigan"],"bottom":["chino"],"shoes":["loafer"]}}]}');
 if n>400 or n=0 then raise exception 'unbounded or empty fixed inventory retrieval';end if;
 if exists(select 1 from public.catalog_stylist_candidates(array['shoes'],'security-v5','{"budget":0,"currency":"USD","rates":{"UZS":1,"USD":12000,"EUR":13000,"GBP":15000}}')) then raise exception 'zero budget bypass';end if;
 if exists(select 1 from public.catalog_stylist_candidates(array['shoes'],'security-v5') r where r->'visual_embedding'<>'null'::jsonb) then raise exception 'embedding leaked to legacy protocol';end if;
end $$;
reset role;
do $$ begin
 if not (select relrowsecurity from pg_class where oid='public.catalog_stylist_capsule'::regclass) then raise exception 'capsule RLS disabled';end if;
 if (select prosecdef from pg_proc where oid='public.catalog_stylist_candidates(text[],text,jsonb,integer,integer,text,numeric)'::regprocedure) then raise exception 'definer retrieval';end if;
end $$;
select 'PASS: private capsule/embeddings, bounded invoker roles, explicit FX/budget and legacy wire isolation' result;
rollback;
