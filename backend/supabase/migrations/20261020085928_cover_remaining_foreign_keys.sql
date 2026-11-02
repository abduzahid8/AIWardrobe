-- Cover remaining referential lookups before user/payment/referral cardinality grows.
-- Existing tables are small; fail promptly rather than waiting behind production writes.
set local lock_timeout='3s';
set local statement_timeout='30s';
-- These tables are installed by optional, unnumbered migrations on some
-- projects. Fresh numbered-migration replays must still complete.
do $$
begin
  if to_regclass('public.admin_permissions') is not null then
    create index if not exists admin_permissions_granted_by_fkey_idx on public.admin_permissions(granted_by);
  end if;
  if to_regclass('public.influencer_referrals') is not null then
    create index if not exists influencer_referrals_user_id_fkey_idx on public.influencer_referrals(user_id);
  end if;
end;
$$;
create index if not exists payments_subscription_id_fkey_idx on public.payments(subscription_id);
do $$
begin
  if exists (
    select 1 from pg_attribute
    where attrelid = 'public.profiles'::regclass
      and attname = 'admin_assigned_by' and not attisdropped
  ) then
    create index if not exists profiles_admin_assigned_by_fkey_idx on public.profiles(admin_assigned_by);
  end if;
end;
$$;
create index if not exists promo_redemptions_promo_code_id_fkey_idx on public.promo_redemptions(promo_code_id);
