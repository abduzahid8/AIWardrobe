-- Run only against the isolated local database with the Auth/Storage harness.
grant usage on schema storage to authenticated, anon;
grant select, insert, update, delete on storage.objects to authenticated, anon;
insert into auth.users(id,email,raw_user_meta_data) values
('11111111-1111-4111-8111-111111111111','avatar-owner@example.invalid','{"username":"avatar-owner"}'),
('22222222-2222-4222-8222-222222222222','avatar-control@example.invalid','{"username":"avatar-control"}') on conflict do nothing;
insert into auth.sessions(id,user_id) values
('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1','11111111-1111-4111-8111-111111111111'),
('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2','22222222-2222-4222-8222-222222222222');
set role authenticated;
set request.jwt.claims = '{"session_id":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"}';
set request.jwt.claim.sub = '11111111-1111-4111-8111-111111111111';
set request.jwt.claims = '{"session_id":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"}';
insert into storage.objects(id,bucket_id,name) values
 ('bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb1','avatars','11111111-1111-4111-8111-111111111111/avatar.png');
set request.jwt.claim.sub = '22222222-2222-4222-8222-222222222222';
set request.jwt.claims = '{"session_id":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2"}';
do $$
declare changed integer;
begin
  if exists(select 1 from storage.objects where bucket_id='avatars') then raise exception 'Another user can list avatar metadata'; end if;
  update storage.objects set name='22222222-2222-4222-8222-222222222222/stolen.png'
    where id='bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb1';
  get diagnostics changed = row_count;
  if changed <> 0 then raise exception 'Another user can overwrite avatar'; end if;
  delete from storage.objects where id='bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb1';
  get diagnostics changed = row_count;
  if changed <> 0 then raise exception 'Another user can delete avatar'; end if;
  begin
    insert into storage.objects(id,bucket_id,name) values
     ('bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb2','avatars','11111111-1111-4111-8111-111111111111/spoof.png');
    raise exception 'Another user can insert into owner folder';
  exception when insufficient_privilege then null;
  end;
end;
$$;
set request.jwt.claim.sub = '11111111-1111-4111-8111-111111111111';
set request.jwt.claims = '{"session_id":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"}';
do $$
begin
  if (select count(*) from storage.objects where bucket_id='avatars') <> 1 then raise exception 'Owner cannot read avatar'; end if;
  begin
    update storage.objects set name='22222222-2222-4222-8222-222222222222/moved.png'
      where id='bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb1';
    raise exception 'Owner can move object into another user folder';
  exception when insufficient_privilege then null;
  end;
end;
$$;
reset role;
set role anon;
set request.jwt.claim.sub = '';
do $$
begin
  begin
    insert into storage.objects(id,bucket_id,name) values
     ('bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb3','avatars','anonymous/avatar.png');
    raise exception 'Anonymous avatar upload allowed';
  exception when insufficient_privilege then null;
  end;
end;
$$;
reset role;
