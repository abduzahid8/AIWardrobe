-- Wear logs are the source of truth for garment usage statistics.
-- Supports both JSONB and legacy UUID[] item_ids through to_jsonb at call sites.
set local lock_timeout = '3s';
create schema if not exists app_security;
revoke all on schema app_security from public;
grant usage on schema app_security to authenticated, service_role;

create or replace function app_security.wear_log_ids(value jsonb)
returns uuid[] language sql immutable security invoker set search_path = '' as $$
  select coalesce(array_agg(distinct item::uuid order by item::uuid), '{}'::uuid[])
  from jsonb_array_elements_text(case when jsonb_typeof(value) = 'array' then value else '[]'::jsonb end) as entries(item)
  where item ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$';
$$;
revoke all on function app_security.wear_log_ids(jsonb) from public, anon;
grant execute on function app_security.wear_log_ids(jsonb) to authenticated, service_role;

create or replace function app_security.sync_wear_log_counters()
returns trigger language plpgsql security invoker set search_path = '' as $$
declare
  old_owner uuid; new_owner uuid;
  old_ids uuid[] := '{}'; new_ids uuid[] := '{}';
  affected record; delta integer; latest timestamptz; recompute_latest boolean;
begin
  if TG_OP <> 'INSERT' then
    old_owner := OLD.user_id;
    old_ids := app_security.wear_log_ids(to_jsonb(OLD.item_ids));
  end if;
  if TG_OP <> 'DELETE' then
    new_owner := NEW.user_id;
    new_ids := app_security.wear_log_ids(to_jsonb(NEW.item_ids));
  end if;
  if TG_OP = 'UPDATE' and old_owner = new_owner and old_ids = new_ids
     and OLD.created_at is not distinct from NEW.created_at then return NEW; end if;

  -- Canonical ID order prevents opposite outfit ordering from inverting row locks.
  for affected in
    select c.id, c.user_id from public.clothing_items c
    where (c.user_id = old_owner and c.id = any(old_ids))
       or (c.user_id = new_owner and c.id = any(new_ids))
    order by c.id for update
  loop
    delta := case when affected.user_id = new_owner and affected.id = any(new_ids) then 1 else 0 end
           - case when affected.user_id = old_owner and affected.id = any(old_ids) then 1 else 0 end;
    recompute_latest := TG_OP <> 'INSERT';
    if recompute_latest then
      select max(w.created_at) into latest from public.wear_logs w
      where w.user_id = affected.user_id
        and affected.id = any(app_security.wear_log_ids(to_jsonb(w.item_ids)));
    else
      latest := NEW.created_at;
    end if;
    update public.clothing_items c set
      wear_count = greatest(0, coalesce(c.wear_count, 0) + delta),
      last_worn_at = case when recompute_latest then latest else greatest(c.last_worn_at, latest) end
    where c.id = affected.id and c.user_id = affected.user_id;
  end loop;
  if TG_OP = 'DELETE' then return OLD; end if;
  return NEW;
end;
$$;
revoke all on function app_security.sync_wear_log_counters() from public, anon;
grant execute on function app_security.sync_wear_log_counters() to authenticated, service_role;

drop trigger if exists sync_wear_log_counters on public.wear_logs;
create trigger sync_wear_log_counters after insert or update or delete on public.wear_logs
for each row execute function app_security.sync_wear_log_counters();

-- Old clients may still send cached counters with garment edits. Keep server statistics.
create or replace function app_security.protect_wear_statistics()
returns trigger language plpgsql security invoker set search_path = '' as $$
begin
  if current_user in ('authenticated', 'anon') and pg_trigger_depth() = 1 then
    if TG_OP = 'INSERT' then NEW.wear_count := 0; NEW.last_worn_at := null;
    else NEW.wear_count := OLD.wear_count; NEW.last_worn_at := OLD.last_worn_at; end if;
  end if;
  return NEW;
end;
$$;
revoke all on function app_security.protect_wear_statistics() from public, anon;
grant execute on function app_security.protect_wear_statistics() to authenticated, service_role;
drop trigger if exists protect_wear_statistics on public.clothing_items;
create trigger protect_wear_statistics before insert or update on public.clothing_items
for each row execute function app_security.protect_wear_statistics();
