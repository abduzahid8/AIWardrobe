-- Runs only inside the runner's disposable local cluster.
do $$ begin
  if (select prosecdef from pg_proc where oid='public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric)'::regprocedure) then raise exception 'RPC must be invoker'; end if;
  if exists(select 1 from pg_proc p, lateral aclexplode(p.proacl) a where p.oid='public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric)'::regprocedure and a.grantee=0) then raise exception 'PUBLIC execute must be revoked'; end if;
end $$;
set role anon;
do $$
declare n integer; roles integer; brands integer; archetypes integer;
begin
 select count(*),count(distinct case when r->>'category'='outerwear' then 'outerwear' else r->>'garment_type' end)
 into n,roles from public.catalog_discovery_candidates(p_seed=>'coverage')r;
 if n<>100 or roles<>4 then raise exception 'bounded balanced slots failed: % rows % roles',n,roles; end if;
 if exists(select 1 from public.catalog_discovery_candidates(p_seed=>'coverage')r where r->>'id' in ('inactive','rls-secret','person','unknown-provenance')) then raise exception 'RLS/provenance violated'; end if;
 select count(distinct r->>'brand'),count(distinct r->>'name') into brands,archetypes from public.catalog_discovery_candidates(p_slot=>'shoes',p_seed=>'shoe-proof',p_limit=>12)r;
 if brands<4 or archetypes<>3 then raise exception 'brand/shoe coverage failed: % %',brands,archetypes; end if;
 if exists(select 1 from public.catalog_discovery_candidates(p_slot=>'shoes',p_currency=>'USD',p_max_price=>50,p_limit=>100)r where r->>'garment_type'<>'shoes' or (r->>'price')::numeric>50 or r->>'currency'<>'USD') then raise exception 'hard retrieval filters violated'; end if;
 if (select count(*) from public.catalog_discovery_candidates(p_search=>'no-such-product'))<>0 then raise exception 'search filter violated'; end if;
 if not exists(select 1 from public.catalog_discovery_candidates(p_slot=>'top',p_seed=>'interior')r where (substring(r->>'id' from '^top-([0-9]+)$'))::integer>1000) then raise exception 'later rows unreachable'; end if;
 if (select array_agg(r->>'id') from public.catalog_discovery_candidates(p_seed=>'a')r)=(select array_agg(r->>'id') from public.catalog_discovery_candidates(p_seed=>'b')r) then raise exception 'seed must rotate candidate discovery'; end if;
 if exists(select r->>'id' from public.catalog_discovery_candidates(p_seed=>'pages',p_offset=>0)r intersect select r->>'id' from public.catalog_discovery_candidates(p_seed=>'pages',p_offset=>100)r) then raise exception 'pagination overlaps'; end if;
 if not exists(select 1 from public.catalog_discovery_candidates(p_slot=>'shoes',p_seed=>'floors',p_limit=>12)r where (r->>'price')::numeric=(select min(price) from public.shop_catalog where category='shoes' and is_active)) then raise exception 'native price floor dropped'; end if;
 begin perform public.catalog_discovery_candidates(p_limit=>101);raise exception 'invalid limit accepted';exception when sqlstate '22023' then null;end;
 begin perform public.catalog_discovery_candidates(p_offset=>-1);raise exception 'invalid offset accepted';exception when sqlstate '22023' then null;end;
 begin perform public.catalog_discovery_candidates(p_max_price=>'NaN'::numeric);raise exception 'NaN accepted';exception when sqlstate '22023' then null;end;
end $$;
reset role;
set role authenticated;
do $$ begin
 if exists(select 1 from public.catalog_discovery_candidates()r where r->>'id'='rls-secret') then raise exception 'authenticated RLS bypass';end if;
end $$;
reset role;
explain (analyze,buffers) select * from public.catalog_discovery_candidates(p_seed=>'performance');
