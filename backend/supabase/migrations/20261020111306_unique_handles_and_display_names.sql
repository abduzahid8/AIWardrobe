-- Handles are account identifiers; display names are independent, non-unique text.
-- Keep username as a legacy display-name mirror for installed older clients.
set lock_timeout = '3s';
set statement_timeout = '30s';
alter table public.profiles add column handle text;
alter table public.profiles add column display_name text not null default '';
update public.profiles set display_name = coalesce(username, '');

-- Preserve the oldest valid legacy candidate in each case-insensitive group.
with candidates as (
 select id, lower(btrim(username)) as candidate,
 row_number() over(partition by lower(btrim(username)) order by created_at nulls last, id) as position
 from public.profiles where lower(btrim(username)) ~ '^[a-z0-9_]{3,30}$'
)
update public.profiles p set handle = c.candidate from candidates c where p.id=c.id and c.position=1;
alter table public.profiles add constraint profiles_handle_key unique(handle);
do $$
declare account record; candidate text;
begin
 for account in select id from public.profiles where handle is null order by id loop
  loop
   candidate := 'user_' || left(replace(pg_catalog.gen_random_uuid()::text, '-', ''), 25);
   begin
    update public.profiles set handle=candidate where id=account.id;
    exit;
   exception when unique_violation then null;
   end;
  end loop;
 end loop;
end $$;
alter table public.profiles alter column handle set not null;
alter table public.profiles add constraint profiles_handle_format check(handle ~ '^[a-z0-9_]{3,30}$');
alter table public.profiles add constraint profiles_display_name_length check(char_length(display_name) <= 200);

create or replace function public.sync_profile_names() returns trigger
language plpgsql set search_path = '' as $$
begin
 if TG_OP='INSERT' then
  if NEW.handle is null then
   NEW.handle := 'user_' || left(replace(pg_catalog.gen_random_uuid()::text, '-', ''), 25);
  end if;
  if NEW.display_name='' and coalesce(NEW.username,'')<>'' then NEW.display_name:=NEW.username; end if;
 elsif NEW.username is distinct from OLD.username and NEW.display_name is not distinct from OLD.display_name then
  -- A legacy profile-name edit must never rename the unique handle.
  NEW.display_name:=coalesce(NEW.username,'');
 end if;
 NEW.handle:=lower(btrim(NEW.handle));
 NEW.username:=NEW.display_name;
 return NEW;
end $$;
revoke all on function public.sync_profile_names() from public,anon,authenticated;
create trigger sync_profile_names before insert or update on public.profiles
 for each row execute function public.sync_profile_names();

create or replace function public.handle_new_user() returns trigger
language plpgsql security definer set search_path = '' as $$
declare requested text; chosen text; shown text; generated boolean; constraint_name text;
begin
 requested:=NEW.raw_user_meta_data->>'handle';
 if requested is null and coalesce(NEW.raw_app_meta_data->>'provider','email')='email'
    and lower(btrim(NEW.raw_user_meta_data->>'username')) ~ '^[a-z0-9_]{3,30}$' then
  requested:=NEW.raw_user_meta_data->>'username';
 end if;
 generated:=requested is null;
 if not generated then
  requested:=lower(btrim(requested));
  if requested !~ '^[a-z0-9_]{3,30}$' then
   raise exception 'invalid_handle' using errcode='23514';
  end if;
 end if;
 shown:=coalesce(NEW.raw_user_meta_data->>'display_name',NEW.raw_user_meta_data->>'full_name',
                  NEW.raw_user_meta_data->>'name',NEW.raw_user_meta_data->>'username','');
 loop
  chosen:=case when generated then 'user_'||left(replace(pg_catalog.gen_random_uuid()::text,'-',''),25) else requested end;
  begin
   insert into public.profiles(id,email,username,display_name,handle,gender,profile_image)
   values(NEW.id,NEW.email,shown,shown,chosen,NEW.raw_user_meta_data->>'gender',NEW.raw_user_meta_data->>'profile_image')
   on conflict(id) do nothing;
   return NEW;
  exception when unique_violation then
   get stacked diagnostics constraint_name = CONSTRAINT_NAME;
   if not generated or constraint_name <> 'profiles_handle_key' then raise; end if;
  end;
 end loop;
end $$;
revoke all on function public.handle_new_user() from public,anon,authenticated;

-- Intentionally expose only a boolean availability lookup, backed by the unique index.
create or replace function public.is_username_available(check_username text) returns boolean
language sql stable security definer set search_path = '' as $$
 select coalesce(lower(btrim(check_username)) ~ '^[a-z0-9_]{3,30}$',false)
 and not exists(select 1 from public.profiles where handle=lower(btrim(check_username)));
$$;
revoke all on function public.is_username_available(text) from public;
grant execute on function public.is_username_available(text) to anon,authenticated,service_role;
