-- STYLIST 5: slot-specific editorial roles, current image facets/embeddings,
-- catalog-wide native price peers and evidence capsule. Existing discovery,
-- product identities, authoritative price/source fields and permissions unchanged.
begin;
set local lock_timeout='2s';
create table if not exists public.catalog_stylist_capsule (
  product_id text primary key references public.shop_catalog(id) on delete cascade,
  source_hash text not null check(source_hash ~ '^[a-f0-9]{32}$'),
  image_sha256 text not null check(image_sha256 ~ '^[a-f0-9]{64}$'),
  archetype text not null check(length(archetype) between 1 and 100),
  evidence jsonb not null check(jsonb_typeof(evidence)='object'),
  created_at timestamptz not null default now()
);
alter table public.catalog_stylist_capsule enable row level security;
revoke all on public.catalog_stylist_capsule from public,anon,authenticated;
grant select,insert,update,delete on public.catalog_stylist_capsule to service_role;
-- Explicit capability survives cloud saves; old clients retain their default.
alter table public.saved_outfits add column if not exists composition_protocol text
  check(composition_protocol is null or composition_protocol='optional-layer-v1');
create or replace function public.catalog_stylist_candidates(
  p_slots text[],p_seed text,p_context jsonb default '{}'::jsonb,
  p_limit integer default 100,p_offset integer default 0,
  p_currency text default null,p_max_price numeric default null
) returns setof jsonb language plpgsql stable security invoker set search_path='' as $$
declare
  rates jsonb:=p_context->'rates'; target_currency text:=p_context->>'currency';
  cap numeric; c text;
begin
  if p_slots is null or cardinality(p_slots)<1 or cardinality(p_slots)>4
    or exists(select 1 from unnest(p_slots) s where s is null or s not in ('top','outerwear','bottom','shoes'))
    or p_seed is null or length(p_seed)>128 or p_limit is null or p_limit<1 or p_limit>100
    or p_offset is null or p_offset not in (0,100)
    or (p_currency is not null and p_currency not in ('UZS','USD','EUR','GBP'))
    or (p_max_price is not null and (p_currency is null or p_max_price<0 or p_max_price>='Infinity'::numeric))
    or p_context is null or jsonb_typeof(p_context)<>'object' or octet_length(p_context::text)>4096
    or (p_context?'concepts' and (jsonb_typeof(p_context->'concepts')<>'array' or jsonb_array_length(p_context->'concepts') not between 1 and 3)) then
    raise exception using errcode='22023',message='invalid_stylist_candidates';
  end if;
  if p_context?'budget' then
    if jsonb_typeof(p_context->'budget')<>'number' or target_currency is null or target_currency not in ('UZS','USD','EUR','GBP') then
      raise exception using errcode='22023',message='invalid_catalog_budget';end if;
    cap:=(p_context->>'budget')::numeric;
    if cap<0 or cap>='Infinity'::numeric then raise exception using errcode='22023',message='invalid_catalog_budget';end if;
  end if;
  if rates is not null then
    if current_user<>'service_role' then raise exception using errcode='42501',message='catalog_rates_require_service';end if;
    foreach c in array array['UZS','USD','EUR','GBP'] loop
      if jsonb_typeof(rates->c) is distinct from 'number' or (rates->>c)::numeric<=0 or (rates->>c)::numeric>1000000000000 then
        raise exception using errcode='22023',message='invalid_catalog_rates';end if;
    end loop;
    if (rates->>'UZS')::numeric<>1 then raise exception using errcode='22023',message='invalid_catalog_rates';end if;
  end if;
  return query
  with concepts as materialized (
    select ordinality as concept_number,value->'roles' roles,value->>'layerPurpose' layer_purpose,array(select jsonb_array_elements_text(
      case when jsonb_typeof(value->'preferredSubtypes')='array' then value->'preferredSubtypes'
           when jsonb_typeof(p_context->'preferredSubtypes')='array' then p_context->'preferredSubtypes' else '[]'::jsonb end)) preferred
    from jsonb_array_elements(coalesce(p_context->'concepts','[{}]'::jsonb)) with ordinality
  ), normalized as materialized (
    -- Only narrow ranking facets survive this scan; descriptions and full
    -- descriptors are hydrated after LIMIT to avoid repeated sort spills.
    select s.id,s.price,s.currency,s.garment_type,
      coalesce(i.slot,case when s.garment_type='shoes' or s.category='shoes' then 'shoes'
        when s.garment_type='lower_body' or s.category='bottoms' then 'bottom'
        when s.garment_type='outerwear' or s.category='outerwear' or s.name ~* '\m(jacket|coat|blazer|cardigan|sweater|overshirt)\M' then 'outerwear'
        when s.garment_type='upper_body' or s.category='tops' then 'top' else coalesce(s.garment_type,'other') end) fashion_slot,
      coalesce(case when
        (case when jsonb_typeof(v.profile#>'{attributes,subtype,confidence}')='number' then (v.profile#>>'{attributes,subtype,confidence}')::numeric else 0 end) between 0.8 and 1
        and (case when jsonb_typeof(v.profile#>'{attributes,subtype,margin}')='number' then (v.profile#>>'{attributes,subtype,margin}')::numeric else 0 end) between 0.1 and 1
        and case i.slot
          when 'top' then v.profile#>>'{attributes,subtype,value}' in ('dress-shirt','shirt','field-shirt','polo','t-shirt','fine-knit','sweater')
          when 'outerwear' then v.profile#>>'{attributes,subtype,value}' in ('blazer','cardigan','overshirt','coat','bomber','jacket','hoodie','pullover')
          when 'bottom' then v.profile#>>'{attributes,subtype,value}' in ('tailored-trouser','chino','trouser','jeans','knit-trouser','athletic-trouser','cargo')
          when 'shoes' then v.profile#>>'{attributes,subtype,value}' in ('oxford','derby','monk','loafer','chelsea','dress-boot','boot','driver','sandal','espadrille','minimal-sneaker','sneaker','sport-sneaker','hiking-boot')
          else false end
        then v.profile#>>'{attributes,subtype,value}' end,i.subtype,'unknown') subtype,coalesce(i.formality,'unknown') formality,
      coalesce(i.intelligence#>>'{descriptor,color}','unknown') color,
      coalesce(i.intelligence#>>'{descriptor,volume}','unknown') volume,
      case when rates is not null and upper(s.currency) in ('UZS','USD','EUR','GBP') then s.price*(rates->>upper(s.currency))::numeric else null end value_cost,
      (i.intelligence#>>'{descriptor,insulated}'='true' or (s.name||' '||coalesce(s.description,'')) ~* '\m(puffer|padded|quilted|insulated|shearling|down[- ]filled)\M' or s.name ~* '\mdown([ -]+[a-z]+){0,3}[ -]+(jacket|coat|parka)\M') insulated,
      coalesce(s.brand,'') brand,v.product_id is not null has_photo, capsule.product_id is not null in_capsule,
      case when cap is null then null
        when rates is null and upper(s.currency)=target_currency then s.price
        when rates is not null and upper(s.currency) in ('UZS','USD','EUR','GBP') then s.price*(rates->>upper(s.currency))::numeric/(rates->>target_currency)::numeric
        else null end cost
    from public.shop_catalog s
    left join public.catalog_garment_intelligence i on i.product_id=s.id and i.version=1
    left join public.catalog_visual_fashion v on v.product_id=s.id and v.version=1 and v.image_url=s.image_url
      and v.source_hash=md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text)
    left join public.catalog_stylist_capsule capsule on capsule.product_id=s.id and capsule.source_hash=v.source_hash and capsule.image_sha256=v.image_sha256
    where s.is_active and s.price>0 and s.price<'Infinity'::numeric and coalesce(s.image_url,'')<>''
      and (p_currency is null or s.currency=p_currency)
      and (p_max_price is null or s.price<=p_max_price)
      and (s.source||' '||s.description) ~* '(no[-[:space:]]?model|no[-[:space:]]?mannequin|product[-[:space:]]?only|plain[-[:space:]]?background|white[-[:space:]]?(bg|background)|elegant-menswear-2026)'
      and not (s.name||' '||s.description) ~* '(\m(photo|image)[[:space:]]+(shows?|features?)\M[^.]*\m(model|mannequin)\M|\mgarment[[:space:]]+worn[[:space:]]+by[[:space:]]+(a[[:space:]]+)?(male[[:space:]]+)?model\M|\mmodel[[:space:]]+(styled|wearing|wears)\M)'
  ), eligible as materialized (
    select n.*,row_number() over(partition by n.fashion_slot,n.currency order by n.price,n.id) price_floor
    from normalized n where n.fashion_slot=any(p_slots)
      and (n.fashion_slot<>'shoes' or n.garment_type='shoes')
      and (p_context->>'dressCode' is distinct from 'formal' or n.formality in ('formal','unknown'))
      and (cap is null or (n.cost is not null and n.cost<=cap))
  ), peer_stats as materialized (
    select fashion_slot,subtype,currency,count(*) peer_count,
      percentile_cont(0.5) within group(order by price) median,
      percentile_cont(0.25) within group(order by price) p25,
      percentile_cont(0.75) within group(order by price) p75
    from eligible group by fashion_slot,subtype,currency
  ), value_peers as materialized (
    select fashion_slot,subtype,count(value_cost) normalized_count,percentile_cont(0.5) within group(order by value_cost) normalized_median from eligible group by fashion_slot,subtype
  ), roles as (
    select e.*,c.concept_number,ps.median,ps.p25,ps.p75,ps.peer_count,vp.normalized_count,vp.normalized_median,
      case when e.subtype=any(case when c.roles?e.fashion_slot then array(select jsonb_array_elements_text(c.roles->e.fashion_slot)) else c.preferred end) then 2 else 0 end
        -case when c.layer_purpose='light-shell' and e.fashion_slot='outerwear' and e.insulated then 2 else 0 end
        +case when e.formality=p_context->>'dressCode' then 1 else 0 end
        +case when e.has_photo then 0.3 else 0 end +case when e.in_capsule then 0.15 else 0 end
        -case when p_context->>'luxury'='true' then 0 when vp.normalized_count>=4 then least(1.5,greatest(0,ln(e.value_cost/vp.normalized_median))) when ps.peer_count>=4 then least(1.5,greatest(0,ln(e.price/ps.median))) else 0 end relevance,
      row_number() over(partition by e.fashion_slot,c.concept_number,e.subtype,e.color,e.volume
        order by md5(e.id||p_seed),e.id) role_rank,
      row_number() over(partition by e.fashion_slot,c.concept_number,e.brand
        order by md5(e.id||p_seed),e.id) brand_rank
    from eligible e cross join concepts c join peer_stats ps on ps.fashion_slot=e.fashion_slot and ps.subtype=e.subtype and ps.currency=e.currency join value_peers vp on vp.fashion_slot=e.fashion_slot and vp.subtype=e.subtype
  ), concept_order as (
    select r.*,row_number() over(partition by fashion_slot,concept_number
      order by (price_floor=1) desc,relevance desc,role_rank,brand_rank,md5(id||p_seed||concept_number::text),id) concept_rank
    from roles r
  ), unique_products as (
    select distinct on(fashion_slot,id) * from concept_order
    order by fashion_slot,id,concept_rank,concept_number
  ), bounded as (
    select u.*,row_number() over(partition by fashion_slot order by (price_floor=1) desc,concept_rank,concept_number,id) slot_number
    from unique_products u
  )
  select jsonb_build_object('id',s.id,'name',s.name,'brand',s.brand,'description',s.description,
    'image_url',s.image_url,'garment_type',s.garment_type,'category',s.category,'price',s.price,'currency',s.currency,
    'source',s.source,'source_url',s.source_url,'external_id',s.external_id,
    'fashion_slot',b.fashion_slot,'fashion_intelligence',i.intelligence,'_budget_cost',b.cost::text,
    'visual_fashion',v.profile,'visual_source_hash',v.source_hash,'visual_embedding',case when p_context->>'engine'='stylist5' then embedding.embedding else null end,
    'price_peers',jsonb_build_object('median',b.median,'p25',b.p25,'p75',b.p75,'count',b.peer_count,'normalizedPrice',b.value_cost,'normalizedMedian',b.normalized_median,'normalizedCount',b.normalized_count))
  from bounded b join public.shop_catalog s on s.id=b.id
  left join public.catalog_garment_intelligence i on i.product_id=s.id and i.version=1
  left join public.catalog_visual_fashion v on v.product_id=s.id and v.version=1 and v.image_url=s.image_url
    and v.source_hash=md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text)
  left join public.catalog_image_embeddings embedding on embedding.image_sha256=v.image_sha256 and embedding.model_revision=v.model_revision
  where b.slot_number>p_offset and b.slot_number<=p_offset+p_limit
  order by b.fashion_slot,b.slot_number;
end; $$;
revoke all on function public.catalog_stylist_candidates(text[],text,jsonb,integer,integer,text,numeric) from public,anon,authenticated;
grant execute on function public.catalog_stylist_candidates(text[],text,jsonb,integer,integer,text,numeric) to service_role;
notify pgrst,'reload schema';
commit;
