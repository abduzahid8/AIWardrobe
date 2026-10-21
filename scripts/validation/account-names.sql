-- Synthetic Auth-trigger/profile checks. Every fixture rolls back.
begin;
set local lock_timeout='3s';set local statement_timeout='20s';
select set_config('request.jwt.claims','{"role":"service_role"}',true);
select set_config('readiness.name_owner',gen_random_uuid()::text,true);
select set_config('readiness.name_other',gen_random_uuid()::text,true);
select set_config('readiness.name_handle','audit_'||left(replace(gen_random_uuid()::text,'-',''),20),true);
insert into auth.users(id,email,raw_user_meta_data) values
 (current_setting('readiness.name_owner')::uuid,'names-'||current_setting('readiness.name_owner')||'@example.invalid',jsonb_build_object('handle',upper(current_setting('readiness.name_handle')),'display_name','Shared Full Name')),
 (current_setting('readiness.name_other')::uuid,'names-'||current_setting('readiness.name_other')||'@example.invalid','{"full_name":"Shared Full Name"}');
do $$declare owner uuid:=current_setting('readiness.name_owner')::uuid; other uuid:=current_setting('readiness.name_other')::uuid; h text:=current_setting('readiness.name_handle');begin
 if not exists(select 1 from profiles where id=owner and handle=h and display_name='Shared Full Name' and username=display_name) then raise exception 'Requested handle/display mapping failed';end if;
 if not exists(select 1 from profiles where id=other and handle ~ '^user_[a-z0-9_]{25}$' and display_name='Shared Full Name') then raise exception 'OAuth generated handle failed';end if;
 if public.is_username_available(upper(h)) or public.is_username_available('!') or public.is_username_available(null) then raise exception 'Availability contract failed';end if;
 begin
  insert into auth.users(id,email,raw_user_meta_data) values(gen_random_uuid(),gen_random_uuid()||'@example.invalid',jsonb_build_object('handle',upper(h)));
  raise exception 'Duplicate signup accepted';
 exception when unique_violation then null;end;
 begin
  insert into auth.users(id,email,raw_user_meta_data) values(gen_random_uuid(),gen_random_uuid()||'@example.invalid','{"handle":"bad handle"}');
  raise exception 'Invalid explicit signup handle accepted';
 exception when check_violation then null;end;
 begin update profiles set handle=upper(h) where id=other;raise exception 'Duplicate profile update accepted';exception when unique_violation then null;end;
 begin update profiles set handle='bad handle' where id=owner;raise exception 'Invalid handle update accepted';exception when check_violation then null;end;
 update profiles set display_name='Apple New Full Name' where id=owner;
 if not exists(select 1 from profiles where id=owner and handle=h and username='Apple New Full Name') then raise exception 'Display edit renamed handle';end if;
 update profiles set username='Legacy Profile Name' where id=owner;
 if not exists(select 1 from profiles where id=owner and handle=h and display_name='Legacy Profile Name') then raise exception 'Legacy profile edit renamed handle';end if;
 update profiles set handle=upper(left(h,24)||'_new') where id=owner;
 if not exists(select 1 from profiles where id=owner and handle=left(h,24)||'_new' and display_name='Legacy Profile Name') then raise exception 'Handle rename changed display name';end if;
 if not public.is_username_available(h) then raise exception 'Released handle stayed reserved';end if;
 if has_function_privilege('anon','public.sync_profile_names()','execute') or has_function_privilege('authenticated','public.handle_new_user()','execute') then raise exception 'Internal trigger exposed';end if;
end $$;
select jsonb_build_object('passed',true,'scope','Auth trigger, case-insensitive unique constraint, display/legacy separation, format, availability, internal ACL; rolled back') as account_name_checks;
rollback;
