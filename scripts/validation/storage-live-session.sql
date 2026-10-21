-- Run after avatar-storage.sql in the isolated readiness database only.
set role authenticated;
set request.jwt.claim.sub = '11111111-1111-4111-8111-111111111111';
set request.jwt.claims = '{"session_id":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"}';
do $$ begin
  if not app_security.has_active_session() then raise exception 'Active owner session rejected'; end if;
end $$;
set request.jwt.claims = '{"session_id":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2"}';
do $$ begin
  if app_security.has_active_session() then raise exception 'Another user session accepted'; end if;
end $$;
set request.jwt.claims = '{"session_id":"invalid"}';
do $$ begin
  if app_security.has_active_session() then raise exception 'Malformed session accepted'; end if;
end $$;
set request.jwt.claims = '{}';
do $$ begin
  if app_security.has_active_session() then raise exception 'Missing session accepted'; end if;
end $$;
reset role;
delete from auth.sessions where id='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1';
set role authenticated;
set request.jwt.claims = '{"session_id":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"}';
do $$ begin
  if app_security.has_active_session() then raise exception 'Revoked session accepted'; end if;
  if exists(select 1 from storage.objects where bucket_id='avatars') then raise exception 'Revoked session can read objects'; end if;
  begin
    insert into storage.objects(id,bucket_id,name) values
    ('cccccccc-cccc-4ccc-8ccc-ccccccccccc1','avatars','11111111-1111-4111-8111-111111111111/revoked.png');
    raise exception 'Revoked session can upload';
  exception when insufficient_privilege then null;
  end;
end $$;
reset role;
set role service_role;
do $$ begin
  if (select count(*) from storage.objects where bucket_id='avatars') <> 1 then raise exception 'Service cleanup cannot enumerate'; end if;
end $$;
reset role;
do $$ begin
 if has_schema_privilege('anon','app_security','usage') then raise exception 'Anonymous helper schema exposed'; end if;
 if has_function_privilege('anon','app_security.has_active_session()','execute') then raise exception 'Anonymous helper execution exposed'; end if;
end $$;
