-- Additive derived metadata. Original catalog, prices, ownership and RLS remain authoritative.
begin;
create table public.catalog_garment_intelligence (
  product_id text primary key references public.shop_catalog(id) on delete cascade,
  version integer not null check(version>0),
  source_hash text not null check(source_hash ~ '^[0-9a-f]{32}$'),
  slot text not null check(slot in ('top','outerwear','bottom','shoes','dresses','accessory','outfits','other')),
  subtype text not null,
  formality text not null,
  search_document tsvector not null,
  intelligence jsonb not null check(jsonb_typeof(intelligence)='object' and octet_length(intelligence::text)<12000),
  updated_at timestamptz not null default now()
);
alter table public.catalog_garment_intelligence enable row level security;
create policy catalog_intelligence_active_read on public.catalog_garment_intelligence for select to anon,authenticated
  using (exists(select 1 from public.shop_catalog s where s.id=product_id and s.is_active));
revoke all on public.catalog_garment_intelligence from public,anon,authenticated;
grant select on public.catalog_garment_intelligence to anon,authenticated;
grant all on public.catalog_garment_intelligence to service_role;
create index catalog_intelligence_slot_context on public.catalog_garment_intelligence(slot,formality,subtype,product_id);
create index catalog_intelligence_search on public.catalog_garment_intelligence using gin(search_document);
create index if not exists catalog_active_currency_price on public.shop_catalog(currency,price,id) where is_active;

create function public.invalidate_catalog_intelligence() returns trigger
language plpgsql security definer set search_path='' as $$
begin
  if row(old.name,old.description,old.brand,old.category,old.garment_type,old.image_url)
     is distinct from row(new.name,new.description,new.brand,new.category,new.garment_type,new.image_url) then
    delete from public.catalog_garment_intelligence where product_id=old.id;
  end if;
  return new;
end; $$;
revoke all on function public.invalidate_catalog_intelligence() from public,anon,authenticated;
create trigger catalog_intelligence_invalidation after update on public.shop_catalog
  for each row execute function public.invalidate_catalog_intelligence();
commit;

-- ACTIVATE RETRIEVAL: release tooling stages the table, derives metadata and
-- validates coverage before this backwards-compatible capability is activated.
begin;
create function public.catalog_discovery_candidates(
  p_slot text default 'all',p_source text default 'all',p_search text default '',
  p_seed text default '0',p_limit integer default 100,p_offset integer default 0,
  p_currency text default null,p_max_price numeric default null,p_context jsonb default '{}'::jsonb
) returns setof jsonb language plpgsql stable security invoker set search_path='' as $$
declare
  rates jsonb:=p_context->'rates'; target_currency text:=p_context->>'currency';
  cap numeric; search_query tsquery; c text;
begin
  if p_slot is null or p_slot not in ('all','top','outerwear','bottom','shoes','dresses','accessory','outfits')
    or p_source is null or length(p_source)>256 or p_search is null or length(p_search)>256
    or p_seed is null or length(p_seed)>128 or p_limit is null or p_limit<1 or p_limit>100
    or p_offset is null or p_offset<0 or p_offset>500000
    or (p_currency is not null and p_currency not in ('UZS','USD','EUR','GBP'))
    or (p_max_price is not null and (p_currency is null or p_max_price<0 or p_max_price>='Infinity'::numeric))
    or p_context is null or jsonb_typeof(p_context)<>'object' or octet_length(p_context::text)>4096 then
      raise exception using errcode='22023',message='invalid_catalog_discovery';
  end if;
  if p_context?'budget' then
    if jsonb_typeof(p_context->'budget')<>'number' or target_currency is null or target_currency not in ('UZS','USD','EUR','GBP') then
      raise exception using errcode='22023',message='invalid_catalog_budget';
    end if;
    cap:=(p_context->>'budget')::numeric;
    if cap<0 or cap>='Infinity'::numeric then raise exception using errcode='22023',message='invalid_catalog_budget';end if;
  end if;
  if rates is not null then
    -- FX evidence is supplied only by the server's request-scoped reference.
    if current_user<>'service_role' then raise exception using errcode='42501',message='catalog_rates_require_service';end if;
    foreach c in array array['UZS','USD','EUR','GBP'] loop
      if jsonb_typeof(rates->c) is distinct from 'number' or (rates->>c)::numeric<=0 or (rates->>c)::numeric>1000000000000 then
        raise exception using errcode='22023',message='invalid_catalog_rates';
      end if;
    end loop;
    if (rates->>'UZS')::numeric<>1 then raise exception using errcode='22023',message='invalid_catalog_rates';end if;
  end if;
  if p_search<>'' then
    select pg_catalog.to_tsquery('simple',string_agg(pg_catalog.quote_literal(token)||':*',' & ')) into search_query
      from unnest(regexp_split_to_array(lower(p_search),'[^[:alnum:]]+')) token where token<>'';
  end if;
  return query
  with normalized as (
    -- Indexed normalized slot/search path for enriched products.
    select s.*,i.slot as fashion_slot,i.subtype,i.formality,i.intelligence,i.search_document
    from public.catalog_garment_intelligence i join public.shop_catalog s on s.id=i.product_id
    where i.version=1 and s.is_active and (p_slot='all' or i.slot=p_slot)
      and (search_query is null or i.search_document@@search_query)
    union all
    -- New/edited products remain discoverable as unknown until offline refresh.
    select s.*,case when s.garment_type='shoes' or s.category='shoes' then 'shoes'
      when s.garment_type='lower_body' or s.category='bottoms' then 'bottom'
      when s.garment_type='outerwear' or s.category='outerwear' or s.name ~* '\m(jacket|coat|blazer|cardigan|sweater|overshirt)\M' then 'outerwear'
      when s.garment_type='upper_body' or s.category='tops' then 'top'
      else coalesce(s.garment_type,'other') end,'unknown','unknown',null::jsonb,
      pg_catalog.to_tsvector('simple',concat_ws(' ',s.name,s.brand,s.description))
    from public.shop_catalog s where s.is_active and not exists(select 1 from public.catalog_garment_intelligence i where i.product_id=s.id and i.version=1)
  ), eligible as (
    select n.*,
      case when cap is null then null
        when rates is null and upper(n.currency)=target_currency then n.price
        when rates is not null and upper(n.currency) in ('UZS','USD','EUR','GBP') then n.price*(rates->>upper(n.currency))::numeric/(rates->>target_currency)::numeric
        else null end as cost,
      case when n.subtype in (select jsonb_array_elements_text(case when jsonb_typeof(p_context->'preferredSubtypes')='array' then p_context->'preferredSubtypes' else '[]'::jsonb end)) then 2 else 0 end
        +case when n.formality=p_context->>'dressCode' then 1 else 0 end as relevance
    from normalized n where (p_slot='all' or n.fashion_slot=p_slot)
      and (search_query is null or n.search_document@@search_query)
      and (p_source='all' or n.source=p_source)
      and (p_currency is null or n.currency=p_currency)
      and (p_max_price is null or (n.price>0 and n.price<=p_max_price))
      and n.price>0 and n.price<'Infinity'::numeric and coalesce(n.image_url,'')<>''
      and (n.source||' '||n.description) ~* '(no[-[:space:]]?model|no[-[:space:]]?mannequin|product[-[:space:]]?only|plain[-[:space:]]?background|white[-[:space:]]?(bg|background)|elegant-menswear-2026)'
      and not (n.name||' '||n.description) ~* '(\m(photo|image)[[:space:]]+(shows?|features?)\M[^.]*\m(model|mannequin)\M|\mgarment[[:space:]]+worn[[:space:]]+by[[:space:]]+(a[[:space:]]+)?(male[[:space:]]+)?model\M|\mmodel[[:space:]]+(styled|wearing|wears)\M)'
      and (p_context->>'dressCode' is distinct from 'formal' or n.formality in ('formal','unknown'))
  ), affordable as (
    select e.* from eligible e where cap is null or (e.cost is not null and e.cost<=cap)
  ), diversified as (
    select a.*,row_number() over(partition by fashion_slot,currency order by price,id) as price_floor,
      row_number() over(partition by fashion_slot,subtype,coalesce(intelligence#>>'{descriptor,color}','unknown'),coalesce(intelligence#>>'{descriptor,volume}','unknown')
        order by relevance desc,md5(id||p_seed),id) as role_rank
    from affordable a
  )
  select jsonb_build_object('id',d.id,'name',d.name,'brand',d.brand,'description',d.description,
    'image_url',d.image_url,'garment_type',d.garment_type,'category',d.category,'price',d.price,'currency',d.currency,
    'source',d.source,'source_url',d.source_url,'external_id',d.external_id,
    'fashion_slot',d.fashion_slot,'fashion_intelligence',d.intelligence,'_budget_cost',d.cost::text)
  from diversified d
  order by (price_floor=1) desc,relevance desc,role_rank,md5(id||p_seed),id
  limit p_limit offset p_offset;
end; $$;
revoke all on function public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric,jsonb) from public,anon,authenticated;
grant execute on function public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric,jsonb) to anon,authenticated,service_role;
notify pgrst,'reload schema';
commit;
