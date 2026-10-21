-- Cost-ordered bounded catalog pages for the authenticated Edge handler.
-- FX comes from its pinned server-owned reference snapshot, never client rates.
begin;
create or replace function public.budget_catalog_candidates(
    p_slot text,
    p_rates jsonb,
    p_after_cost text default null,
    p_after_id text default null,
    p_limit integer default 100,
    p_activity boolean default false
) returns setof jsonb
language plpgsql stable security invoker set search_path = ''
as $$
declare
    c text;
begin
    if p_slot is null or p_slot not in ('top','bottom','shoes','outerwear')
       or p_limit is null or p_limit < 1 or p_limit > 100
       or p_activity is null or p_rates is null or jsonb_typeof(p_rates) <> 'object'
       or octet_length(p_rates::text) > 1000
       or (p_after_cost is not null and (p_after_id is null or length(p_after_cost) > 120 or p_after_cost !~ '^[0-9]+([.][0-9]+)?$'))
       or (p_after_id is not null and (length(p_after_id) = 0 or octet_length(p_after_id) > 512)) then
        raise exception using errcode = '22023', message = 'invalid_catalog_page';
    end if;
    foreach c in array array['UZS','USD','EUR','GBP'] loop
        if jsonb_typeof(p_rates -> c) is distinct from 'number'
           or (p_rates ->> c)::numeric <= 0 or (p_rates ->> c)::numeric > 1000000000000 then
            raise exception using errcode = '22023', message = 'invalid_catalog_rates';
        end if;
    end loop;
    if (p_rates ->> 'UZS')::numeric <> 1 then
        raise exception using errcode = '22023', message = 'invalid_catalog_rates';
    end if;
    return query
    with candidates as materialized (
        select s.id, s.brand, s.name, s.price, s.currency, s.image_url,
               s.garment_type, s.category, s.description, s.source, s.source_url,
               case when s.price > 0 and s.price < 'Infinity'::numeric
                    and upper(btrim(s.currency)) in ('UZS','USD','EUR','GBP')
                    then s.price * (p_rates ->> upper(btrim(s.currency)))::numeric
                    else null end as cost
        from public.shop_catalog as s
        where s.is_active = true and (
            (p_slot = 'shoes' and (s.category = 'shoes' or s.garment_type = 'shoes')) or
            (p_slot = 'top' and (s.category = 'tops' or s.garment_type = 'upper_body')) or
            (p_slot = 'bottom' and (s.category = 'bottoms' or s.garment_type = 'lower_body')) or
            (p_slot = 'outerwear' and (s.category = 'outerwear' or s.garment_type = 'outerwear'
                or s.name ilike any(array['%jacket%','%coat%','%blazer%','%cardigan%','%sweater%','%hoodie%','%puffer%','%vest%'])))
        ) and (not p_activity or lower(coalesce(s.name,'') || ' ' || coalesce(s.description,''))
            ~ '(athletic|training|running|workout|performance|sportswear|activewear|gym|спортивн|трениров)')
    )
    select jsonb_build_object(
        'id',v.id,'brand',v.brand,'name',v.name,'price',v.price,'currency',v.currency,
        'image_url',v.image_url,'garment_type',v.garment_type,'category',v.category,
        'description',v.description,'source',v.source,'source_url',v.source_url,
        '_budget_cost',v.cost::text)
    from candidates as v
    where p_after_id is null
       or (p_after_cost is not null and (v.cost > p_after_cost::numeric
           or (v.cost = p_after_cost::numeric and v.id > p_after_id) or v.cost is null))
       or (p_after_cost is null and v.cost is null and v.id > p_after_id)
    order by v.cost asc nulls last, v.id asc
    limit p_limit;
end;
$$;
revoke all on function public.budget_catalog_candidates(text,jsonb,text,text,integer,boolean) from public, anon, authenticated;
grant execute on function public.budget_catalog_candidates(text,jsonb,text,text,integer,boolean) to service_role;
commit;
