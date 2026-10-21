-- Isolated Supabase-interface database only. Account deletion must not need
-- Auth's internal role to access the private statistics helper or garment rows.
\set ON_ERROR_STOP on
begin;
do $$ begin
 if not exists(select 1 from pg_roles where rolname='supabase_auth_admin') then create role supabase_auth_admin nologin; end if;
end $$;
grant usage on schema auth to supabase_auth_admin;
grant delete on auth.users to supabase_auth_admin;
grant select(id) on auth.users to supabase_auth_admin;
insert into auth.users(id,email,raw_user_meta_data) values
('76666666-6666-4666-8666-666666666666','wear-delete@example.invalid','{"username":"wear-delete"}');
insert into public.clothing_items(id,user_id) values
('77777777-7777-4777-8777-777777777777','76666666-6666-4666-8666-666666666666');
insert into public.wear_logs(id,user_id,item_ids,date) values
('78888888-8888-4888-8888-888888888888','76666666-6666-4666-8666-666666666666',
'["77777777-7777-4777-8777-777777777777"]','2026-10-16');
set local role supabase_auth_admin;
delete from auth.users where id='76666666-6666-4666-8666-666666666666';
reset role;
do $$ begin
 if exists(select 1 from public.wear_logs where user_id='76666666-6666-4666-8666-666666666666') then raise exception 'Wear log survived account deletion'; end if;
 if exists(select 1 from public.clothing_items where user_id='76666666-6666-4666-8666-666666666666') then raise exception 'Garment survived account deletion'; end if;
end $$;
rollback;
\echo Auth deletion with wear history passed
