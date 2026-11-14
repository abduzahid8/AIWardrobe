-- Phase 6 — Growth & Monetization (LOCAL ONLY, additive, backward compatible)
--
-- New tables (all user-scoped via owner_id = auth.uid(), RLS enforced):
--   growth_activation_state  one row per user: activation milestones (6A)
--   growth_notification_prefs one row per user: channel preferences (6L)
--   growth_referrals          referral ledger, idempotency key unique (6AI/6BD)
--   growth_savings_ledger     verified savings with provenance (6AN)
--   growth_feedback           low-friction feedback events (6BR)
--
-- Nothing here touches live RevenueCat, stores, notifications, or
-- experiments. Apply only to hosted DB with explicit approval.

-- ── Activation state (6A) ───────────────────────────────────────
create table if not exists public.growth_activation_state (
    owner_id uuid not null references auth.users (id) on delete cascade,
    wardrobe_started boolean not null default false,
    enough_items_for_outfit boolean not null default false,
    first_outfit_generated boolean not null default false,
    first_outfit_saved boolean not null default false,
    first_wear_confirmed boolean not null default false,
    first_purchase_check boolean not null default false,
    first_personalization_signal boolean not null default false,
    first_aha_moment boolean not null default false,
    last_milestone_at timestamptz null,
    updated_at timestamptz not null default now(),
    primary key (owner_id)
);

-- ── Notification preferences (6L) ───────────────────────────────
create table if not exists public.growth_notification_prefs (
    owner_id uuid not null references auth.users (id) on delete cascade,
    outfit_reminders text not null default 'normal',
    event_reminders text not null default 'normal',
    price_alerts text not null default 'normal',
    fit_feedback text not null default 'important_only',
    weekly_review text not null default 'normal',
    travel_packing text not null default 'normal',
    shopping_reminders text not null default 'important_only',
    updated_at timestamptz not null default now(),
    primary key (owner_id)
);

-- ── Referral ledger (6AI/6BD) ───────────────────────────────────
create table if not exists public.growth_referrals (
    id uuid not null default gen_random_uuid() primary key,
    inviter_user_id uuid not null references auth.users (id) on delete cascade,
    invitee_user_id uuid not null references auth.users (id) on delete cascade,
    invite_token text not null,
    status text not null default 'invited',
    reward_idempotency_key text null unique,
    created_at timestamptz not null default now(),
    constraint growth_referrals_no_self check (inviter_user_id <> invitee_user_id)
);

-- ── Verified savings ledger (6AN) ───────────────────────────────
create table if not exists public.growth_savings_ledger (
    id uuid not null default gen_random_uuid() primary key,
    owner_id uuid not null references auth.users (id) on delete cascade,
    kind text not null,
    amount_cents integer not null check (amount_cents > 0),
    currency text not null,
    provenance text not null,
    verified_at timestamptz not null default now()
);

-- ── Feedback events (6BR) ───────────────────────────────────────
create table if not exists public.growth_feedback (
    id uuid not null default gen_random_uuid() primary key,
    owner_id uuid not null references auth.users (id) on delete cascade,
    feature text not null,
    helpful boolean not null,
    reason text null,
    created_at timestamptz not null default now()
);

-- ── RLS ─────────────────────────────────────────────────────────
alter table public.growth_activation_state enable row level security;
alter table public.growth_notification_prefs enable row level security;
alter table public.growth_referrals enable row level security;
alter table public.growth_savings_ledger enable row level security;
alter table public.growth_feedback enable row level security;

drop policy if exists growth_owner_isolation on public.growth_activation_state;
create policy growth_owner_isolation on public.growth_activation_state
    for all using (auth.uid() = owner_id) with check (auth.uid() = owner_id);

drop policy if exists growth_owner_isolation on public.growth_notification_prefs;
create policy growth_owner_isolation on public.growth_notification_prefs
    for all using (auth.uid() = owner_id) with check (auth.uid() = owner_id);

drop policy if exists growth_referral_participant on public.growth_referrals;
create policy growth_referral_participant on public.growth_referrals
    for all using (auth.uid() = inviter_user_id or auth.uid() = invitee_user_id)
    with check (auth.uid() = inviter_user_id or auth.uid() = invitee_user_id);

drop policy if exists growth_owner_isolation on public.growth_savings_ledger;
create policy growth_owner_isolation on public.growth_savings_ledger
    for all using (auth.uid() = owner_id) with check (auth.uid() = owner_id);

drop policy if exists growth_owner_isolation on public.growth_feedback;
create policy growth_owner_isolation on public.growth_feedback
    for all using (auth.uid() = owner_id) with check (auth.uid() = owner_id);

-- Explicit least-privilege grants: do not depend on hosted default privileges.
revoke all on public.growth_activation_state from anon, authenticated;
grant select, insert, update, delete on public.growth_activation_state to authenticated;
grant all on public.growth_activation_state to service_role;
revoke all on public.growth_notification_prefs from anon, authenticated;
grant select, insert, update, delete on public.growth_notification_prefs to authenticated;
grant all on public.growth_notification_prefs to service_role;
revoke all on public.growth_referrals from anon, authenticated;
grant select on public.growth_referrals to authenticated;
grant all on public.growth_referrals to service_role;
revoke all on public.growth_savings_ledger from anon, authenticated;
grant select on public.growth_savings_ledger to authenticated;
grant all on public.growth_savings_ledger to service_role;
revoke all on public.growth_feedback from anon, authenticated;
grant select, insert, update, delete on public.growth_feedback to authenticated;
grant all on public.growth_feedback to service_role;
