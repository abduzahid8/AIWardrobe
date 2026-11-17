CREATE OR REPLACE FUNCTION public.catalog_stylist_candidates(p_slots text[], p_seed text, p_context jsonb DEFAULT '{}'::jsonb, p_limit integer DEFAULT 100, p_offset integer DEFAULT 0, p_currency text DEFAULT NULL::text, p_max_price numeric DEFAULT NULL::numeric)
 RETURNS SETOF jsonb
 LANGUAGE plpgsql
 STABLE
 SET search_path TO ''
AS $function$
begin
  if p_slots is null or cardinality(p_slots)<1 or cardinality(p_slots)>4
    or exists(select 1 from unnest(p_slots) s where s is null or s not in ('top','outerwear','bottom','shoes'))
    or p_seed is null or length(p_seed)>128
    or p_limit is null or p_limit<1 or p_limit>100 or p_offset is null or p_offset not in (0,100) then
    raise exception using errcode='22023',message='invalid_stylist_candidates';
  end if;
  if p_context is null or jsonb_typeof(p_context)<>'object' or octet_length(p_context::text)>4096
    or (p_context?'concepts' and (jsonb_typeof(p_context->'concepts')<>'array' or jsonb_array_length(p_context->'concepts') not between 1 and 3)) then
    raise exception using errcode='22023',message='invalid_stylist_context';
  end if;
  return query
  with concepts as (
    select value concept,ordinality concept_number,count(*) over() concept_count
    from jsonb_array_elements(coalesce(p_context->'concepts','[{}]'::jsonb)) with ordinality
  ), discovered as (
    -- Each concept asks the same indexed, source-safe catalog for its own roles.
    -- A single network round trip searches all slots; SQL work stays bounded.
    select slots.slot,c.concept_number,r.ordinality,candidate
    from (select distinct unnest(p_slots) slot) slots cross join concepts c
    cross join lateral public.catalog_discovery_candidates(slots.slot,'all','',
      left(p_seed,80)||md5(c.concept::text),
      least(100,ceil(p_limit::numeric/c.concept_count)::integer),
      floor(p_offset::numeric/c.concept_count)::integer,p_currency,p_max_price,
      (p_context-'concepts') || jsonb_build_object('preferredSubtypes',coalesce(c.concept->'preferredSubtypes',p_context->'preferredSubtypes','[]'::jsonb))) with ordinality r(candidate,ordinality)
  ), unique_products as (
    select distinct on(slot,candidate->>'id') * from discovered
    order by slot,candidate->>'id',ordinality,concept_number
  ), bounded as (
    select u.*,row_number() over(partition by slot order by ordinality,concept_number,candidate->>'id') slot_number
    from unique_products u
  )
  select b.candidate || jsonb_build_object('visual_fashion',v.profile,'visual_source_hash',v.source_hash)
  from bounded b join public.shop_catalog s on s.id=b.candidate->>'id'
  left join public.catalog_visual_fashion v on v.product_id=s.id and v.version=1
    and v.image_url=s.image_url
    and v.source_hash=md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text)
  where b.slot_number<=p_limit;
end; $function$
