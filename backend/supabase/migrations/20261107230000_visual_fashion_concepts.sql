-- Stylist 4: additive, versioned offline image evidence and private feedback.
-- No source products/prices are rewritten. v197 can run with this migration.
begin;
create table public.catalog_visual_fashion (
  product_id text not null references public.shop_catalog(id) on delete cascade,
  version integer not null check(version=1),
  source_hash text not null check(source_hash ~ '^[0-9a-f]{32}$'),
  image_url text not null check(image_url like 'https://%'),
  image_sha256 text not null check(image_sha256 ~ '^[0-9a-f]{64}$'),
  model text not null check(length(model) between 1 and 128),
  model_revision text not null check(model_revision ~ '^[0-9a-f]{40}$'),
  profile jsonb not null check(jsonb_typeof(profile)='object' and octet_length(profile::text)<16000),
  created_at timestamptz not null default now(),
  primary key(product_id,version)
);
alter table public.catalog_visual_fashion enable row level security;
create policy visual_active_read on public.catalog_visual_fashion for select to anon,authenticated
  using(exists(select 1 from public.shop_catalog s where s.id=product_id and s.is_active));
revoke all on public.catalog_visual_fashion from public,anon,authenticated;
grant select on public.catalog_visual_fashion to anon,authenticated;
grant all on public.catalog_visual_fashion to service_role;
create index visual_category on public.catalog_visual_fashion((profile#>>'{attributes,slot,value}'),product_id);
create table public.catalog_image_embeddings (
  image_sha256 text not null check(image_sha256 ~ '^[0-9a-f]{64}$'),
  model_revision text not null check(model_revision ~ '^[0-9a-f]{40}$'),
  embedding real[] not null check(array_length(embedding,1)=768),
  primary key(image_sha256,model_revision)
);
alter table public.catalog_image_embeddings enable row level security;
revoke all on public.catalog_image_embeddings from public,anon,authenticated;
grant all on public.catalog_image_embeddings to service_role;
-- Embeddings are reusable offline. Only compact evidence travels to Edge.

create function public.catalog_stylist_candidates(
  p_slots text[],p_seed text,p_context jsonb default '{}'::jsonb,
  p_limit integer default 100,p_offset integer default 0,
  p_currency text default null,p_max_price numeric default null
) returns setof jsonb language plpgsql stable security invoker set search_path='' as $$
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
end; $$;
revoke all on function public.catalog_stylist_candidates(text[],text,jsonb,integer,integer,text,numeric) from public,anon,authenticated;
grant execute on function public.catalog_stylist_candidates(text[],text,jsonb,integer,integer,text,numeric) to service_role;

-- Anonymous to other users; auth.uid() is used only for ownership and deletion.
-- No prompt, image, product history or free text is collected.
create table public.stylist_look_feedback (
  user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
  request_id uuid not null,
  look_id text not null check(look_id ~ '^outfit_[1-3]$'),
  concept_id text not null check(concept_id ~ '^[a-z][a-z0-9-]{0,63}$'),
  engine_version text not null check(length(engine_version) between 1 and 80),
  rating smallint not null check(rating in (-1,1)),
  reason text check(reason in ('coherence','occasion','color','fit','footwear','like')),
  created_at timestamptz not null default now(),
  primary key(user_id,request_id,look_id)
);
alter table public.stylist_look_feedback enable row level security;
create policy feedback_own_read on public.stylist_look_feedback for select to authenticated using(user_id=(select auth.uid()));
create policy feedback_own_insert on public.stylist_look_feedback for insert to authenticated with check(user_id=(select auth.uid()));
create policy feedback_own_update on public.stylist_look_feedback for update to authenticated using(user_id=(select auth.uid())) with check(user_id=(select auth.uid()));
create policy feedback_own_delete on public.stylist_look_feedback for delete to authenticated using(user_id=(select auth.uid()));
revoke all on public.stylist_look_feedback from public,anon,authenticated;
grant select,insert,update,delete on public.stylist_look_feedback to authenticated;
grant all on public.stylist_look_feedback to service_role;
create index feedback_owner_recent on public.stylist_look_feedback(user_id,created_at desc);
notify pgrst,'reload schema';
commit;
