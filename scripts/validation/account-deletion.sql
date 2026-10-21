-- Isolated local database only. Tests storage ownership and relational cascade.
insert into auth.users(id) values
 ('11111111-1111-4111-8111-111111111111'),
 ('22222222-2222-4222-8222-222222222222');
insert into public.profiles(id) select id from auth.users;
insert into public.influencer_referrals(user_id) values ('11111111-1111-4111-8111-111111111111');
insert into storage.objects(id,bucket_id,name,owner,owner_id) values
 ('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1','user_uploads','11111111-1111-4111-8111-111111111111/nested/a.png','11111111-1111-4111-8111-111111111111',null),
 ('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2','AIWARDROBE','legacy.png',null,'11111111-1111-4111-8111-111111111111'),
 ('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa3','user_uploads','11111111-1111-4111-8111-111111111111/service.png',null,null),
 ('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa4','AIWARDROBE','11111111-1111-4111-8111-111111111111/other-owner.png',null,'22222222-2222-4222-8222-222222222222');

do $$
begin
  if has_function_privilege('anon','public.list_account_deletion_objects(uuid)','execute')
     or has_function_privilege('authenticated','public.list_account_deletion_objects(uuid)','execute') then
    raise exception 'Storage enumeration leaked to client roles';
  end if;
end;
$$;
set role service_role;
do $$
begin
  if (select count(*) from public.list_account_deletion_objects('11111111-1111-4111-8111-111111111111')) <> 3 then
    raise exception 'Owner enumeration failed or selected another owner';
  end if;
end;
$$;
reset role;
delete from auth.users where id = '11111111-1111-4111-8111-111111111111';
do $$
begin
  if exists(select 1 from public.influencer_referrals) then raise exception 'Referral blocked account deletion'; end if;
  if (select count(*) from public.profiles) <> 1 then raise exception 'Profile cascade damaged another account'; end if;
end;
$$;
