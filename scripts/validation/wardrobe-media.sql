-- Local Auth/Storage interface checks, never production.
\set ON_ERROR_STOP on
begin;
grant usage on schema storage to authenticated,anon;
grant select,insert,update,delete on storage.objects to authenticated,anon;
insert into auth.users(id,email,raw_user_meta_data) values
('31111111-1111-4111-8111-111111111111','media-owner@example.invalid','{"username":"media-owner"}'),
('32222222-2222-4222-8222-222222222222','media-other@example.invalid','{"username":"media-other"}');
insert into auth.sessions(id,user_id) values
('3aaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1','31111111-1111-4111-8111-111111111111'),
('3aaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2','32222222-2222-4222-8222-222222222222');
set local role authenticated;
set local request.jwt.claim.sub='31111111-1111-4111-8111-111111111111';
set local request.jwt.claims='{"session_id":"3aaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"}';
insert into storage.objects(id,bucket_id,name) values
('3bbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb1','wardrobe-images','31111111-1111-4111-8111-111111111111/test/original.jpg');
set local request.jwt.claim.sub='32222222-2222-4222-8222-222222222222';
set local request.jwt.claims='{"session_id":"3aaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2"}';
do $$ declare changed integer; begin
 if exists(select 1 from storage.objects where bucket_id='wardrobe-images') then raise exception 'Other owner can list media'; end if;
 update storage.objects set name='stolen.jpg' where id='3bbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb1';
 get diagnostics changed=row_count;
 if changed<>0 then raise exception 'Other owner changed media'; end if;
 begin
  insert into storage.objects(id,bucket_id,name) values
  ('3bbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb2','wardrobe-images','31111111-1111-4111-8111-111111111111/spoof.jpg');
  raise exception 'Other owner wrote into owner folder';
 exception when insufficient_privilege then null; end;
end $$;
reset role;
delete from auth.sessions where id='3aaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1';
set local role authenticated;
set local request.jwt.claim.sub='31111111-1111-4111-8111-111111111111';
set local request.jwt.claims='{"session_id":"3aaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"}';
do $$ begin
 if exists(select 1 from storage.objects where bucket_id='wardrobe-images') then raise exception 'Revoked session can read media'; end if;
 begin
  insert into storage.objects(id,bucket_id,name) values
  ('3bbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb3','wardrobe-images','31111111-1111-4111-8111-111111111111/revoked.jpg');
  raise exception 'Revoked session can upload media';
 exception when insufficient_privilege then null; end;
end $$;
rollback;
\echo Private wardrobe media ownership and session checks passed
