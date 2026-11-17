-- Correct derived taxonomy with reversible evidence; never update shop_catalog.
begin;
set local lock_timeout='2s';
create table if not exists public.stylist5_taxonomy_rollback(product_id text primary key,original jsonb not null);
alter table public.stylist5_taxonomy_rollback enable row level security;
revoke all on public.stylist5_taxonomy_rollback from public,anon,authenticated;
grant all on public.stylist5_taxonomy_rollback to service_role;
create or replace function public.stylist5_source_taxonomy(p_name text,p_type text,p_category text)
returns jsonb language sql immutable security invoker set search_path='' as $$
 select case
 when n ~* '\mcardigans?\M|кардиган' then jsonb_build_object('slot','outerwear','subtype','cardigan','productKind','garment')
 when n ~* '\m(blazers?|suit jackets?|tailored jackets?|sports? coats?|dinner jackets?)\M|пиджак' then jsonb_build_object('slot','outerwear','subtype','blazer','productKind','garment')
 when n ~* '\moxfords?\M' and (n ~* '\mshirts?\M' or coalesce(p_type,p_category,'') ~* '(upper_body|tops|shirts)') then jsonb_build_object('slot','top','subtype','dress-shirt','productKind','garment')
 when n ~* '\m(shirts?|trousers?|pants?|skirts?|shorts?|jackets?|waistcoats?|vests?|shoes?|boots?|loafers?)\M' then null
 when n ~* '\msuits?\M|\m(two|three|2|3) piece\M.*\m(set|tailor)|костюм' or coalesce(p_type,p_category,'') ~* '^(suits?|complete.suit|outfits)$' then jsonb_build_object('slot','outfits','subtype','complete-suit','productKind','complete-suit')
 else null end from (select lower(regexp_replace(coalesce(p_name,''),'[_-]+',' ','g')) n) normalized;
$$;
revoke all on function public.stylist5_source_taxonomy(text,text,text) from public,anon,authenticated;
grant execute on function public.stylist5_source_taxonomy(text,text,text) to service_role;
insert into public.stylist5_taxonomy_rollback(product_id,original)
 select i.product_id,to_jsonb(i) from public.catalog_garment_intelligence i join public.shop_catalog s on s.id=i.product_id
 cross join lateral (select public.stylist5_source_taxonomy(s.name,s.garment_type,s.category) value) t
 where t.value is not null and (i.slot is distinct from t.value->>'slot' or i.subtype is distinct from t.value->>'subtype'
   or i.intelligence#>>'{descriptor,productKind}' is distinct from t.value->>'productKind')
 on conflict(product_id) do nothing;
update public.catalog_garment_intelligence i set
 slot=t.value->>'slot',subtype=t.value->>'subtype',
 formality=case when t.value->>'subtype'='cardigan' then 'smart' when t.value->>'subtype'='dress-shirt' then 'formal' else i.formality end,
 intelligence=jsonb_set(i.intelligence,'{descriptor}',coalesce(i.intelligence->'descriptor','{}'::jsonb)||t.value
   ||case when t.value->>'subtype'='cardigan' then '{"formality":"smart"}'::jsonb when t.value->>'subtype'='dress-shirt' then '{"formality":"formal"}'::jsonb else '{}'::jsonb end)
  ||jsonb_build_object('taxonomyRevision','stylist5-source-nouns-1'),updated_at=now()
 from public.shop_catalog s cross join lateral (select public.stylist5_source_taxonomy(s.name,s.garment_type,s.category) value) t
 where i.product_id=s.id and t.value is not null;
notify pgrst,'reload schema';
commit;
