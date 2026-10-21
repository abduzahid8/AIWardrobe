set local lock_timeout = '3s';
create or replace function app_security.sync_wear_log_counters()
returns trigger language plpgsql security invoker set search_path = '' as $$
declare
  old_owner uuid; new_owner uuid;
  old_ids uuid[] := '{}'; new_ids uuid[] := '{}';
  affected record; delta integer; latest timestamptz; recompute_latest boolean;
begin
  -- Auth owns the identity cascade. All garments of this owner are being
  -- removed too; no statistics remain to maintain and no extra grants are needed.
  if TG_OP = 'DELETE' and current_user = 'supabase_auth_admin' then return OLD; end if;
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
