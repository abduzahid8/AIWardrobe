-- Only enumerate metadata; remove file bytes through the Storage API.
create or replace function public.list_account_deletion_objects(p_user_id uuid)
returns table(bucket_id text, name text)
language sql stable security invoker
set search_path = ''
as $$
  select o.bucket_id, o.name
  from storage.objects o
  where o.owner_id = p_user_id::text
     or o.owner = p_user_id
     or (nullif(o.owner_id, '') is null and o.owner is null
         and starts_with(o.name, p_user_id::text || '/'))
  order by o.bucket_id, o.name
  limit 500;
$$;
revoke all on function public.list_account_deletion_objects(uuid) from public, anon, authenticated;
grant execute on function public.list_account_deletion_objects(uuid) to service_role;

-- This legacy hosted table was created outside the migration ledger.
do $$
begin
  if to_regclass('public.influencer_referrals') is not null then
    alter table public.influencer_referrals
      drop constraint if exists influencer_referrals_user_id_fkey;
    alter table public.influencer_referrals
      add constraint influencer_referrals_user_id_fkey
      foreign key (user_id) references public.profiles(id) on delete cascade;
  end if;
end;
$$;
notify pgrst, 'reload schema';
