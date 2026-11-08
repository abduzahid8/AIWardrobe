-- Keep the existing retrieval RPC available for controlled embedding hydration,
-- but make the normal candidate scan strip embeddings before they cross the
-- Edge/PostgREST boundary.
begin;
set local lock_timeout='2s';

create or replace function public.catalog_stylist_candidates_compact(
  p_slots text[],p_seed text,p_context jsonb default '{}'::jsonb,
  p_limit integer default 100,p_offset integer default 0,
  p_currency text default null,p_max_price numeric default null
) returns setof jsonb
language sql stable security invoker set search_path=''
as $$
  select value - 'visual_embedding'
  from public.catalog_stylist_candidates(
    p_slots,p_seed,p_context,p_limit,p_offset,p_currency,p_max_price
  ) as rows(value);
$$;

revoke all on function public.catalog_stylist_candidates_compact(text[],text,jsonb,integer,integer,text,numeric)
  from public,anon,authenticated;
grant execute on function public.catalog_stylist_candidates_compact(text[],text,jsonb,integer,integer,text,numeric)
  to service_role;
notify pgrst,'reload schema';
commit;
