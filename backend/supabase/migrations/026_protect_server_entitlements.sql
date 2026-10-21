-- Subscription entitlements and admin privileges are server-owned.
-- Apply with the matching client change that stops writing subscription rows.
begin;

drop policy if exists "Users can insert own subscriptions" on public.subscriptions;
drop policy if exists "Users can update own subscriptions" on public.subscriptions;
revoke insert, update, delete on public.subscriptions from anon, authenticated;

-- RLS limits rows, not columns. An owner must not promote themselves through
-- an otherwise legitimate profile edit. Keep service-role webhook writes intact.
create or replace function public.protect_profile_authority()
returns trigger
language plpgsql
set search_path = public
as $$
begin
    if auth.role() in ('anon', 'authenticated') then
        if TG_OP = 'INSERT' then
            if coalesce(NEW.subscription_tier, 'free') <> 'free'
               or NEW.subscription_expires_at is not null
               or NEW.is_admin
               or to_jsonb(NEW)->>'admin_role' is not null then
                raise exception 'server-owned profile fields' using errcode = '42501';
            end if;
        elsif NEW.subscription_tier is distinct from OLD.subscription_tier
           or NEW.subscription_expires_at is distinct from OLD.subscription_expires_at
           or NEW.is_admin is distinct from OLD.is_admin
           or NEW.email is distinct from OLD.email
           or (to_jsonb(NEW)->'admin_role') is distinct from (to_jsonb(OLD)->'admin_role')
           or (to_jsonb(NEW)->'admin_assigned_by') is distinct from (to_jsonb(OLD)->'admin_assigned_by')
           or (to_jsonb(NEW)->'admin_assigned_at') is distinct from (to_jsonb(OLD)->'admin_assigned_at') then
            raise exception 'server-owned profile fields' using errcode = '42501';
        end if;
        if TG_OP = 'INSERT' and NEW.trial_started_at is not null then
            NEW.trial_started_at := now();
        elsif TG_OP = 'UPDATE' and NEW.trial_started_at is distinct from OLD.trial_started_at then
            if OLD.trial_started_at is null and NEW.trial_started_at is not null then
                NEW.trial_started_at := now();
            else
                raise exception 'server-owned trial start' using errcode = '42501';
            end if;
        end if;
    end if;
    return NEW;
end;
$$;

drop trigger if exists protect_profile_authority on public.profiles;
create trigger protect_profile_authority
before insert or update on public.profiles
for each row execute function public.protect_profile_authority();

-- Email is editable profile data in old installations. Only the server-managed
-- flag may authorize catalog/config administration.
create or replace function public.is_admin()
returns boolean
language sql
stable
set search_path = public
as $$
    select exists (
        select 1 from public.profiles where id = auth.uid() and is_admin = true
    );
$$;

commit;
