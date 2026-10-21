-- Apply the same free20 / Lite100 / Pro+unlimited rule to direct API writes.
-- Only a new ID consumes capacity: edits/upsert retries retain existing garments.
begin;
set local lock_timeout = '5s';

create or replace function app_security.enforce_wardrobe_capacity()
returns trigger
language plpgsql volatile security definer
set search_path = ''
as $$
declare
    owner uuid := auth.uid();
    request_role text := auth.role();
    plan text;
    item_limit integer;
    item_count integer;
begin
    -- Trusted jobs remain able to repair/import data. Anonymous API writes
    -- still fail the existing ownership RLS policy (and are rejected below).
    if request_role is null or request_role = 'service_role' then return new; end if;
    if request_role <> 'authenticated' or owner is null or new.user_id is distinct from owner then
        raise exception 'auth owner required' using errcode = '42501';
    end if;

    -- Serialize admission for one owner across requests/devices. NO KEY UPDATE
    -- is compatible with the profile FK's KEY SHARE lock. The subsequent SELECT
    -- in this VOLATILE trigger gets a fresh READ COMMITTED snapshot after waiting.
    perform 1 from public.profiles p where p.id = owner for no key update;
    if not found then raise exception 'profile not found' using errcode = '42501'; end if;

    -- BEFORE INSERT also runs for INSERT ... ON CONFLICT DO UPDATE.
    -- Preserve edits/retries even when a downgrade leaves the wardrobe over cap.
    if exists (select 1 from public.clothing_items c where c.id = new.id and c.user_id = owner) then
        return new;
    end if;
    plan := public.effective_subscription_tier(owner);
    item_limit := case plan when 'premium' then -1 when 'vip' then -1 when 'lite' then 100 else 20 end;
    if item_limit = -1 then return new; end if;

    -- Bound work by the cap and use the existing owner index.
    select count(*) into item_count from
        (select 1 from public.clothing_items c where c.user_id = owner limit item_limit) owned;
    if item_count >= item_limit then
        raise exception 'WARDROBE_LIMIT_REACHED'
            using errcode = 'P0001', detail = pg_catalog.json_build_object('limit', item_limit)::text;
    end if;
    return new;
end;
$$;
revoke all on function app_security.enforce_wardrobe_capacity() from public, anon, authenticated, service_role;

drop trigger if exists enforce_wardrobe_capacity on public.clothing_items;
create trigger enforce_wardrobe_capacity
before insert on public.clothing_items
for each row execute function app_security.enforce_wardrobe_capacity();
commit;
