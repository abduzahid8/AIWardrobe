-- Two-stage Stylist 5 retrieval. The compact stage scans only identity,
-- price/source and narrow styling facets. The hydration stage is called only
-- for the bounded shortlist and is the only stage allowed to return vectors
-- or full visual/intelligence payloads.
begin;
set local lock_timeout='2s';

create or replace function public.catalog_stylist_candidates_compact(
  p_slots text[],p_seed text,p_context jsonb default '{}'::jsonb,
  p_limit integer default 100,p_offset integer default 0,
  p_currency text default null,p_max_price numeric default null
) returns setof jsonb
language plpgsql stable security invoker set search_path=''
as $$
declare
  rates jsonb:=p_context->'rates'; target_currency text:=p_context->>'currency';
  cap numeric; c text;
begin
  if p_slots is null or cardinality(p_slots)<1 or cardinality(p_slots)>4
    or exists(select 1 from unnest(p_slots) s where s is null or s not in ('top','outerwear','bottom','shoes'))
    or p_seed is null or length(p_seed)>128 or p_limit is null or p_limit<1 or p_limit>100
    or p_offset is null or p_offset not in (0,100)
    or (p_currency is not null and upper(p_currency) not in ('UZS','USD','EUR','GBP'))
    or (p_max_price is not null and (p_currency is null or p_max_price<0 or p_max_price>='Infinity'::numeric))
    or p_context is null or jsonb_typeof(p_context)<>'object' then
    raise exception using errcode='22023',message='invalid_compact_stylist_candidates';
  end if;
  if octet_length(p_context::text)>4096
    or (p_context?'concepts' and
      (jsonb_typeof(p_context->'concepts')<>'array' or jsonb_array_length(p_context->'concepts') not between 1 and 3)) then
    raise exception using errcode='22023',message='invalid_compact_context';
  end if;
  if p_context?'budget' then
    if jsonb_typeof(p_context->'budget')<>'number' or target_currency is null
      or target_currency not in ('UZS','USD','EUR','GBP') then
      raise exception using errcode='22023',message='invalid_compact_budget';
    end if;
    cap:=(p_context->>'budget')::numeric;
    if cap<0 or cap>='Infinity'::numeric then
      raise exception using errcode='22023',message='invalid_compact_budget';
    end if;
  end if;
  if rates is not null then
    if current_user<>'service_role' then
      raise exception using errcode='42501',message='compact_rates_require_service';
    end if;
    foreach c in array array['UZS','USD','EUR','GBP'] loop
      if jsonb_typeof(rates->c) is distinct from 'number'
        or (rates->>c)::numeric<=0 or (rates->>c)::numeric>1000000000000 then
        raise exception using errcode='22023',message='invalid_compact_rates';
      end if;
    end loop;
    if (rates->>'UZS')::numeric<>1 then
      raise exception using errcode='22023',message='invalid_compact_rates';
    end if;
  end if;
  return query
  with concepts as materialized (
    select ordinality::int as concept_number,
      value->'roles' as roles,
      array(select jsonb_array_elements_text(
        case when jsonb_typeof(value->'preferredSubtypes')='array'
          then value->'preferredSubtypes'
          when jsonb_typeof(p_context->'preferredSubtypes')='array'
          then p_context->'preferredSubtypes'
          else '[]'::jsonb end)) preferred
    from jsonb_array_elements(coalesce(p_context->'concepts','[{}]'::jsonb)) with ordinality
  ), base as materialized (
    select s.id,s.name,s.brand,s.description,s.image_url,s.garment_type,s.category,
      s.price,s.currency,s.source,s.source_url,s.external_id,
      coalesce(i.slot,case
        when s.garment_type='shoes' or s.category='shoes' then 'shoes'
        when s.garment_type='lower_body' or s.category='bottoms' then 'bottom'
        when s.garment_type='outerwear' or s.category='outerwear'
          or s.name ~* '\m(jacket|coat|blazer|cardigan|sweater|overshirt)\M' then 'outerwear'
        when s.garment_type='upper_body' or s.category='tops' then 'top'
        else coalesce(s.garment_type,'other') end) as fashion_slot,
      coalesce(case when i.subtype is not null then i.subtype
        when s.garment_type='shoes' then 'shoe' else 'unknown' end,'unknown') as subtype,
      coalesce(i.formality,'unknown') as formality,
      coalesce(i.intelligence#>>'{descriptor,color}','unknown') as color,
      coalesce(i.intelligence#>>'{descriptor,volume}','unknown') as volume,
      case when rates is not null and upper(s.currency) in ('UZS','USD','EUR','GBP')
        then s.price*(rates->>upper(s.currency))::numeric else null end as value_cost,
      case when cap is null then null
        when rates is null and upper(s.currency)=target_currency then s.price
        when rates is not null and upper(s.currency) in ('UZS','USD','EUR','GBP')
          then s.price*(rates->>upper(s.currency))::numeric/(rates->>target_currency)::numeric
        else null end as cost,
      (v.product_id is not null) as has_visual,
      (capsule.product_id is not null) as in_capsule
    from public.shop_catalog s
    left join public.catalog_garment_intelligence i on i.product_id=s.id and i.version=1
    left join public.catalog_visual_fashion v on v.product_id=s.id and v.version=1
      and v.image_url=s.image_url
      and v.source_hash=md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text)
    left join public.catalog_stylist_capsule capsule on capsule.product_id=s.id
      and capsule.source_hash=v.source_hash and capsule.image_sha256=v.image_sha256
    where s.is_active and s.price>0 and s.price<'Infinity'::numeric
      and coalesce(s.image_url,'')<>''
      and coalesce(s.currency,'')=coalesce(p_currency,s.currency)
      and (p_max_price is null or s.price<=p_max_price)
      and (s.source||' '||coalesce(s.description,'')) ~* '(no[-[:space:]]?model|no[-[:space:]]?mannequin|product[-[:space:]]?only|plain[-[:space:]]?background|white[-[:space:]]?(bg|background)|elegant-menswear-2026)'
      and not (s.name||' '||coalesce(s.description,'')) ~* '(\m(photo|image)[[:space:]]+(shows?|features?)\M[^.]*\m(model|mannequin)\M|\mgarment[[:space:]]+worn[[:space:]]+by[[:space:]]+(a[[:space:]]+)?(male[[:space:]]+)?model\M|\mmodel[[:space:]]+(styled|wearing|wears)\M)'
  ), eligible as materialized (
    select b.*,c.concept_number,
      row_number() over(partition by b.fashion_slot,b.currency order by b.price,b.id) as price_floor,
      case when b.subtype=any(case when c.roles ? b.fashion_slot
        then array(select jsonb_array_elements_text(c.roles->b.fashion_slot))
        else c.preferred end) then 2 else 0 end
        +case when b.has_visual then 0.3 else 0 end
        +case when b.in_capsule then 0.15 else 0 end as relevance,
      row_number() over(partition by b.fashion_slot,c.concept_number,b.subtype,b.color,b.volume
        order by md5(b.id||p_seed||c.concept_number::text),b.id) as role_rank
    from base b cross join concepts c
    where b.fashion_slot=any(p_slots)
      and (b.fashion_slot<>'shoes' or b.garment_type='shoes')
      and (p_context->>'dressCode' is distinct from 'formal' or b.formality in ('formal','unknown'))
      and (cap is null or (b.cost is not null and b.cost<=cap))
  ), peer_stats as materialized (
    select fashion_slot,subtype,currency,count(*) peer_count,
      percentile_cont(0.5) within group(order by price) median,
      percentile_cont(0.25) within group(order by price) p25,
      percentile_cont(0.75) within group(order by price) p75
    from eligible group by fashion_slot,subtype,currency
  ), value_peers as materialized (
    select fashion_slot,subtype,count(value_cost) normalized_count,
      percentile_cont(0.5) within group(order by value_cost) normalized_median
    from eligible group by fashion_slot,subtype
  ), scored as (
    select e.*,ps.median,ps.p25,ps.p75,ps.peer_count,vp.normalized_count,vp.normalized_median,
      e.relevance
      -case when p_context->>'luxury'='true' then 0
        when vp.normalized_count>=4 then least(1.5,greatest(0,ln(e.value_cost/vp.normalized_median)))
        when ps.peer_count>=4 then least(1.5,greatest(0,ln(e.price/ps.median)))
        else 0 end as value_relevance
    from eligible e
    join peer_stats ps on ps.fashion_slot=e.fashion_slot and ps.subtype=e.subtype and ps.currency=e.currency
    join value_peers vp on vp.fashion_slot=e.fashion_slot and vp.subtype=e.subtype
  ), covered as materialized (
    select distinct on (fashion_slot,id) *
    from scored
    order by fashion_slot,id,value_relevance desc,concept_number
  ), concept_reserved as materialized (
    select distinct on (fashion_slot,concept_number,subtype,color,volume)
      fashion_slot,id,concept_number,subtype,color,volume,price_floor,value_relevance,
      median,p25,p75,peer_count,normalized_count,normalized_median,value_cost
    from scored
    where role_rank<=2
    order by fashion_slot,concept_number,subtype,color,volume,value_relevance desc,id
  ), ranked as (
    select c.*,row_number() over(partition by fashion_slot order by
      (c.id in (select r.id from concept_reserved r where r.fashion_slot=c.fashion_slot)) desc,
      (value_relevance>=2) desc,value_relevance desc,md5(id||p_seed),id) as slot_number
    from covered c
  )
  select jsonb_build_object(
    'id',id,'name',name,'brand',brand,'description',description,'image_url',image_url,
    'garment_type',garment_type,'category',category,'price',price,'currency',currency,
    'source',source,'source_url',source_url,'external_id',external_id,
    'fashion_slot',fashion_slot,'subtype',subtype,'formality',formality,
    'color',color,'volume',volume,'has_visual',has_visual,'in_capsule',in_capsule,
    'price_peers',jsonb_build_object('median',median,'p25',p25,'p75',p75,'count',peer_count,
      'normalizedPrice',value_cost,'normalizedMedian',normalized_median,'normalizedCount',normalized_count)
  )
  from ranked
  where slot_number>p_offset and slot_number<=p_offset+p_limit
  order by fashion_slot,slot_number;
end; $$;

drop function if exists public.catalog_stylist_candidate_hydration(text[]);
create or replace function public.catalog_stylist_candidate_hydration(
  p_ids text[], p_context jsonb default '{}'::jsonb
) returns setof jsonb
language sql stable security invoker set search_path=''
as $$
  with selected as (
    select s.*,coalesce(i.slot,case
      when s.garment_type='shoes' or s.category='shoes' then 'shoes'
      when s.garment_type='lower_body' or s.category='bottoms' then 'bottom'
      when s.garment_type='outerwear' or s.category='outerwear' then 'outerwear'
      when s.garment_type='upper_body' or s.category='tops' then 'top'
      else coalesce(s.garment_type,'other') end) as fashion_slot,
      coalesce(i.subtype,case when s.garment_type='shoes' then 'shoe' else 'unknown' end) as subtype
    from public.shop_catalog s
    left join public.catalog_garment_intelligence i on i.product_id=s.id and i.version=1
    where s.id=any(p_ids)
  ), peers as (
    select selected.id,
      percentile_cont(0.5) within group(order by peer.price) median,
      percentile_cont(0.25) within group(order by peer.price) p25,
      percentile_cont(0.75) within group(order by peer.price) p75,
      count(*) peer_count,
      count(case when p_context->'rates' ? upper(peer.currency)
        then 1 end) normalized_count,
      percentile_cont(0.5) within group(order by
        case when p_context->'rates' ? upper(peer.currency)
          and p_context->>'currency' is not null
          then peer.price*(p_context->'rates'->>upper(peer.currency))::numeric
            /(p_context->'rates'->>(p_context->>'currency'))::numeric end) normalized_median
    from selected
    join public.shop_catalog peer on peer.is_active and peer.price>0
      and peer.price<'Infinity'::numeric
    left join public.catalog_garment_intelligence pi on pi.product_id=peer.id and pi.version=1
    where coalesce(pi.slot,case
        when peer.garment_type='shoes' or peer.category='shoes' then 'shoes'
        when peer.garment_type='lower_body' or peer.category='bottoms' then 'bottom'
        when peer.garment_type='outerwear' or peer.category='outerwear' then 'outerwear'
        when peer.garment_type='upper_body' or peer.category='tops' then 'top'
        else coalesce(peer.garment_type,'other') end)=selected.fashion_slot
      and coalesce(pi.subtype,case when peer.garment_type='shoes' then 'shoe' else 'unknown' end)=selected.subtype
      and peer.currency=selected.currency
    group by selected.id
  )
  select jsonb_build_object(
    'id',s.id,
    'fashion_intelligence',i.intelligence,
    'visual_fashion',v.profile,
    'visual_source_hash',v.source_hash,
    'visual_embedding',embedding.embedding,
    'price_peers',jsonb_build_object(
      'median',p.median,'p25',p.p25,'p75',p.p75,'count',p.peer_count,
      'normalizedPrice',case when p_context->'rates' ? upper(s.currency)
        and p_context->>'currency' is not null
        then s.price*(p_context->'rates'->>upper(s.currency))::numeric
          /(p_context->'rates'->>(p_context->>'currency'))::numeric end,
      'normalizedMedian',p.normalized_median,'normalizedCount',p.normalized_count
    )
  )
  from selected s
  left join public.catalog_garment_intelligence i on i.product_id=s.id and i.version=1
  left join public.catalog_visual_fashion v on v.product_id=s.id and v.version=1
    and v.image_url=s.image_url
    and v.source_hash=md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text)
  left join public.catalog_image_embeddings embedding on embedding.image_sha256=v.image_sha256
    and embedding.model_revision=v.model_revision
  left join peers p on p.id=s.id;
$$;

revoke all on function public.catalog_stylist_candidates_compact(text[],text,jsonb,integer,integer,text,numeric)
  from public,anon,authenticated;
grant execute on function public.catalog_stylist_candidates_compact(text[],text,jsonb,integer,integer,text,numeric)
  to service_role;
revoke all on function public.catalog_stylist_candidate_hydration(text[],jsonb) from public,anon,authenticated;
grant execute on function public.catalog_stylist_candidate_hydration(text[],jsonb) to service_role;
notify pgrst,'reload schema';
commit;
