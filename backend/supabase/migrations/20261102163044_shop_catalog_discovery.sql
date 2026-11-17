-- Additive bounded discovery. Existing rows, IDs, budget RPC and RLS are unchanged.
-- This function is retrieval evidence only; existing outfit gates remain authoritative.
begin;
create function public.catalog_discovery_candidates(
  p_slot text default 'all', p_source text default 'all', p_search text default '',
  p_seed text default '0', p_limit integer default 100, p_offset integer default 0,
  p_currency text default null, p_max_price numeric default null
) returns setof jsonb
language plpgsql stable security invoker set search_path = ''
as $$
begin
  if p_slot is null or p_slot not in ('all','top','bottom','shoes','outerwear','dresses','accessory','outfits')
    or p_source is null or octet_length(p_source)>256 or p_search is null or length(p_search)>256
    or p_seed is null or octet_length(p_seed)>512 or p_limit is null or p_limit<1 or p_limit>100
    or p_offset is null or p_offset<0 or p_offset>500000
    or (p_currency is not null and p_currency !~ '^[A-Z]{3}$')
    or (p_max_price is not null and (p_max_price<0 or p_max_price>='Infinity'::numeric)) then
    raise exception using errcode='22023',message='invalid_catalog_discovery';
  end if;
  return query
  with eligible as materialized (
    select s.id,s.name,s.brand,s.description,s.image_url,s.garment_type,s.category,
      s.price,s.currency,s.source,s.source_url,s.external_id,
      case
        when s.garment_type='shoes' or s.category='shoes' then 'shoes'
        when s.garment_type='lower_body' or s.category='bottoms' then 'bottom'
        when s.category='outerwear' or s.garment_type='outerwear' or s.name ~* '\m(jacket|coat|blazer|cardigan|sweater|overshirt)\M' then 'outerwear'
        when s.garment_type='upper_body' or s.category='tops' then 'top'
        else s.garment_type end as role,
      case when s.garment_type='shoes' or s.category='shoes' then
        case when s.name ~* '(loafer|penny|лофер)' then 'loafer'
          when s.name ~* '(boot|chelsea|chukka|ботин)' then 'boot'
          when s.name ~* '(sneaker|trainer|кроссов|кеды)' then 'sneaker'
          when s.name ~* '(derby|oxford|brogue|туфл)' then 'dress-shoe'
          when s.name ~* '(sandal|slide|сандал)' then 'sandal' else 'other-shoe' end
        else s.garment_type end as visual_key,
      lower(btrim(s.brand)) as brand_key,
      -- This is a diversity bucket, never a persisted family or physical identity.
      regexp_replace(lower(s.name),'\m(black|white|navy|blue|brown|burgundy|grey|gray|beige|green|red|cream)\M','','g') as model_bucket,
      md5(s.id || ':' || p_seed) as seeded_order
    from public.shop_catalog s
    where s.is_active=true
      and (p_source='all' or s.source=p_source)
      and (p_search='' or position(lower(p_search) in lower(s.name||' '||s.brand||' '||s.description))>0)
      and (p_currency is null or s.currency=p_currency)
      and (p_max_price is null or (s.price>=0 and s.price<=p_max_price))
      and (p_slot='all'
        or (p_slot='top' and (s.garment_type='upper_body' or s.category='tops'))
        or (p_slot='bottom' and (s.garment_type='lower_body' or s.category='bottoms'))
        or (p_slot='shoes' and (s.garment_type='shoes' or s.category='shoes'))
        or (p_slot='outerwear' and (s.category='outerwear' or s.garment_type='outerwear' or s.name ~* '\m(jacket|coat|blazer|cardigan|sweater|overshirt)\M'))
        or (p_slot='outfits' and s.garment_type='outfit')
        or (p_slot in ('dresses','accessory') and s.garment_type=p_slot))
      and (s.source||' '||s.description) ~* '(no[-[:space:]]?model|no[-[:space:]]?mannequin|product[-[:space:]]?only|plain[-[:space:]]?background|white[-[:space:]]?(bg|background)|elegant-menswear-2026)'
      and not (s.name||' '||s.description) ~* '(\m(photo|image)[[:space:]]+(shows?|features?)\M[^.]*\m(model|mannequin)\M|\mgarment[[:space:]]+worn[[:space:]]+by[[:space:]]+(a[[:space:]]+)?(male[[:space:]]+)?model\M|\mmodel[[:space:]]+(styled|wearing|wears)\M)'
  ), models as (
    select e.*,
      row_number() over(partition by role,brand_key,model_bucket order by seeded_order,id) as model_rank,
      row_number() over(partition by role,currency order by case when price>0 and price<'Infinity'::numeric then price end nulls last,id) as floor_rank
    from eligible e
  ), brands as (
    select m.*,row_number() over(partition by role,brand_key order by model_rank,seeded_order,id) as brand_rank from models m
  ), visuals as (
    select b.*,row_number() over(partition by role,visual_key order by brand_rank,seeded_order,id) as visual_rank from brands b
  ), roles as (
    select v.*,row_number() over(partition by role order by visual_rank,seeded_order,id) as role_rank from visuals v
  )
  select jsonb_build_object('id',r.id,'name',r.name,'brand',r.brand,'description',r.description,
    'image_url',r.image_url,'garment_type',r.garment_type,'category',r.category,'price',r.price,
    'currency',r.currency,'source',r.source,'source_url',r.source_url,'external_id',r.external_id)
  from roles r
  order by (r.floor_rank=1 and r.price>0 and r.price<'Infinity'::numeric) desc,
    r.role_rank,r.seeded_order,r.id
  limit p_limit offset p_offset;
end;
$$;
revoke all on function public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric) from public,anon,authenticated,service_role;
grant execute on function public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric) to anon,authenticated,service_role;
comment on function public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric)
  is 'Bounded active product-only catalog discovery with invoker RLS, seeded model/brand/role coverage and native price floors. Does not validate outfits.';
commit;
