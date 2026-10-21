-- One authoritative read avoids client-side ranking and multi-request races.
begin;
create or replace function public.get_subscription_status()
returns table(tier text, expiry_date timestamptz, trial_started_at timestamptz)
language plpgsql security definer set search_path = public as $$
declare
  owner uuid := auth.uid(); chosen_tier text; chosen_expiry timestamptz; trial_start timestamptz;
begin
  if owner is null then raise exception 'auth required' using errcode = '42501'; end if;
  select p.trial_started_at into trial_start from public.profiles p where p.id = owner;
  if not found then raise exception 'profile not found' using errcode = '42501'; end if;
  select s.tier, s.end_date into chosen_tier, chosen_expiry from public.subscriptions s
  where s.user_id = owner and s.tier in ('lite', 'premium', 'vip') and s.status in ('active', 'trial', 'cancelled')
    and (s.status <> 'cancelled' or s.end_date > now())
    and (s.end_date is null or s.end_date > now())
    and (s.status <> 'trial' or s.trial_end_date is null or s.trial_end_date > now())
  order by case s.tier when 'vip' then 3 when 'premium' then 2 else 1 end desc, s.end_date desc nulls first limit 1;
  return query select coalesce(chosen_tier, 'free'), chosen_expiry, trial_start;
end;
$$;
revoke execute on function public.get_subscription_status() from public, anon;
grant execute on function public.get_subscription_status() to authenticated;
commit;
