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
      case when b.subtype=any(case when c.roles ? b.fashion_slot
        then array(select jsonb_array_elements_text(c.roles->b.fashion_slot))
        else c.preferred end) then 2 else 0 end
        +case when b.has_visual then 0.3 else 0 end
        +case when b.in_capsule then 0.15 else 0 end as relevance
    from base b cross join concepts c
    where b.fashion_slot=any(p_slots)
      and (b.fashion_slot<>'shoes' or b.garment_type='shoes')
      and (p_context->>'dressCode' is distinct from 'formal' or b.formality in ('formal','unknown'))
  ), covered as materialized (
    select distinct on (fashion_slot,id) *
    from eligible
    order by fashion_slot,id,relevance desc,concept_number
  ), ranked as (
    select c.*,row_number() over(partition by fashion_slot order by
      (relevance>=2) desc,relevance desc,md5(id||p_seed),id) as slot_number
    from covered c
  )
  select jsonb_build_object(
    'id',id,'name',name,'brand',brand,'description',description,'image_url',image_url,
    'garment_type',garment_type,'category',category,'price',price,'currency',currency,
    'source',source,'source_url',source_url,'external_id',external_id,
    'fashion_slot',fashion_slot,'subtype',subtype,'formality',formality,
    'color',color,'volume',volume,'has_visual',has_visual,'in_capsule',in_capsule,
    'fashion_intelligence',jsonb_build_object('version',1,'descriptor',jsonb_build_object(
      'slot',fashion_slot,'subtype',subtype,'formality',formality,'color',color,'volume',volume))
  )
  from ranked
  where slot_number>p_offset and slot_number<=p_offset+p_limit
  order by fashion_slot,slot_number;
end; $$;

create or replace function public.catalog_stylist_candidate_hydration(
  p_ids text[]
) returns setof jsonb
language sql stable security invoker set search_path=''
as $$
  select jsonb_build_object(
    'id',s.id,
    'fashion_intelligence',i.intelligence,
    'visual_fashion',v.profile,
    'visual_source_hash',v.source_hash,
    'visual_embedding',embedding.embedding,
    'price_peers',jsonb_build_object(
      'currency',s.currency,'price',s.price
    )
  )
  from public.shop_catalog s
  left join public.catalog_garment_intelligence i on i.product_id=s.id and i.version=1
  left join public.catalog_visual_fashion v on v.product_id=s.id and v.version=1
    and v.image_url=s.image_url
    and v.source_hash=md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text)
  left join public.catalog_image_embeddings embedding on embedding.image_sha256=v.image_sha256
    and embedding.model_revision=v.model_revision
  where s.id=any(p_ids);
$$;

revoke all on function public.catalog_stylist_candidates_compact(text[],text,jsonb,integer,integer,text,numeric)
  from public,anon,authenticated;
grant execute on function public.catalog_stylist_candidates_compact(text[],text,jsonb,integer,integer,text,numeric)
  to service_role;
revoke all on function public.catalog_stylist_candidate_hydration(text[]) from public,anon,authenticated;
grant execute on function public.catalog_stylist_candidate_hydration(text[]) to service_role;
notify pgrst,'reload schema';
commit;
