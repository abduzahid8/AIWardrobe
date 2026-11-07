-- Phase 3 Personal Style OS — additive local migration (DO NOT APPLY REMOTELY
-- without explicit approval). Creates durable tables for style evidence,
-- profiles, outfit history, goals, purchase decisions/outcomes, events,
-- price watches, recommendation feedback, item lifecycle, style prefs.
-- All tables user-scoped with RLS; no backfill, no DDL on existing tables.

-- ============ style_evidence (3A) ============
create table if not exists public.style_evidence (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  source_type text not null check (source_type in (
    'declared_preference','outfit_worn','outfit_saved','outfit_liked',
    'outfit_disliked','product_saved','product_skipped','product_bought',
    'product_returned','manual_feedback','wardrobe_item','reference_outfit',
    'moodboard','social_import')),
  entity_type text not null check (entity_type in (
    'garment','outfit','product','visual_family','brand','color',
    'silhouette','material','pattern','style_tag')),
  entity_key text not null,
  polarity text not null default 'positive' check (polarity in ('positive','negative','neutral')),
  strength double precision not null default 0.5 check (strength >= 0 and strength <= 1),
  context jsonb,
  confidence double precision not null default 0.5 check (confidence >= 0 and confidence <= 1),
  observed_at timestamptz not null default now(),
  expires_at timestamptz,
  provenance text not null default '',
  created_at timestamptz not null default now()
);
create index if not exists style_evidence_user_key_idx
  on public.style_evidence (user_id, entity_key, observed_at desc);

-- ============ style_profiles (3B snapshot cache) ============
create table if not exists public.style_profiles (
  user_id uuid primary key references auth.users (id) on delete cascade,
  version int not null default 1,
  dimensions jsonb not null default '{}',
  emerging text[] not null default '{}',
  evidence_count int not null default 0,
  computed_at timestamptz not null default now()
);

-- ============ outfit_history (3F) ============
create table if not exists public.outfit_history (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  outfit_id text not null,
  item_ids text[] not null default '{}',
  worn_date date not null,
  planned_date date,
  context_id text,
  rating text check (rating in ('loved','good','okay','didnt_work')),
  weather_temp double precision,
  weather_condition text,
  repeat_count int not null default 1,
  combination_key text not null default '',
  planned boolean not null default false,
  confirmed boolean not null default true,
  note text,
  created_at timestamptz not null default now()
);
create index if not exists outfit_history_user_worn_idx
  on public.outfit_history (user_id, worn_date desc);

-- ============ wardrobe_goals (3R) ============
create table if not exists public.wardrobe_goals (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  type text not null,
  target text,
  time_horizon_days int,
  budget jsonb,
  priority text not null default 'medium' check (priority in ('low','medium','high')),
  status text not null default 'active' check (status in ('active','paused','done','abandoned')),
  progress jsonb not null default '{}',
  constraints text[] not null default '{}',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- ============ purchase_decisions + purchase_outcomes (3T/3U) ============
create table if not exists public.purchase_decisions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  product_id text not null,
  stage text not null default 'discovered' check (stage in (
    'discovered','considering','compared','waiting_for_price','planned',
    'bought','skipped','returned','archived')),
  decision_reason text,
  target_price double precision,
  currency text,
  evidence text[] not null default '{}',
  updated_at timestamptz not null default now()
);
create index if not exists purchase_decisions_user_stage_idx
  on public.purchase_decisions (user_id, stage);

create table if not exists public.purchase_outcomes (
  product_id text not null,
  user_id uuid not null references auth.users (id) on delete cascade,
  worth_it boolean,
  fits_well boolean,
  wear_count int not null default 0,
  would_buy_again boolean,
  asked_at timestamptz not null default now(),
  answered_at timestamptz,
  calibration text[] not null default '{}',
  primary key (user_id, product_id)
);

-- ============ wardrobe_events (3O/3P) ============
create table if not exists public.wardrobe_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  title text not null,
  starts_at timestamptz not null,
  context_id text not null,
  dress_code text,
  climate_hint text,
  source text not null default 'manual' check (source in ('manual','calendar','import')),
  selected_outfit_id text,
  locked boolean not null default false,
  preparation_status text not null default 'draft',
  missing_item_ids text[] not null default '{}',
  created_at timestamptz not null default now()
);
create index if not exists wardrobe_events_user_starts_idx
  on public.wardrobe_events (user_id, starts_at);

-- ============ price_watches (extends Phase 2 foundation) ============
create table if not exists public.price_watches (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  product_id text not null,
  target_price double precision,
  currency text,
  active boolean not null default true,
  created_at timestamptz not null default now()
);

-- ============ recommendation_feedback (3E/3Z/§36) ============
create table if not exists public.recommendation_feedback (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  suggestion_id text not null,
  topic_key text not null default '',
  prompt_type text not null default '',
  outcome text not null check (outcome in (
    'accepted','ignored','dismissed','purchased','returned',
    'worn_repeatedly','rated_poorly')),
  at timestamptz not null default now()
);
create index if not exists recommendation_feedback_user_topic_idx
  on public.recommendation_feedback (user_id, topic_key, at desc);

-- ============ wardrobe_item_lifecycle (3H) ============
create table if not exists public.wardrobe_item_lifecycle (
  item_id text not null,
  user_id uuid not null references auth.users (id) on delete cascade,
  acquired_at date,
  acquisition_price double precision,
  currency text,
  source text,
  condition text,
  repair_needed boolean not null default false,
  last_worn_at date,
  wear_count int not null default 0,
  status text not null default 'active' check (status in (
    'active','seasonal','storage','repair','sell','donate','archived')),
  satisfaction int check (satisfaction is null or (satisfaction >= 1 and satisfaction <= 5)),
  fit_status text,
  replacement_candidate boolean not null default false,
  primary key (user_id, item_id)
);

-- ============ user_style_preferences (declared overrides, §38) ============
create table if not exists public.user_style_preferences (
  user_id uuid primary key references auth.users (id) on delete cascade,
  prefs jsonb not null default '{}',
  novelty_mode text not null default 'balanced' check (novelty_mode in ('safe','balanced','experimental')),
  updated_at timestamptz not null default now()
);

-- ============ RLS (user-scoped; deny-by-default like sibling migrations) ============
alter table public.style_evidence enable row level security;
alter table public.style_profiles enable row level security;
alter table public.outfit_history enable row level security;
alter table public.wardrobe_goals enable row level security;
alter table public.purchase_decisions enable row level security;
alter table public.purchase_outcomes enable row level security;
alter table public.wardrobe_events enable row level security;
alter table public.price_watches enable row level security;
alter table public.recommendation_feedback enable row level security;
alter table public.wardrobe_item_lifecycle enable row level security;
alter table public.user_style_preferences enable row level security;

do $$
declare t text;
begin
  for t in select unnest(array[
    'style_evidence','style_profiles','outfit_history','wardrobe_goals',
    'purchase_decisions','purchase_outcomes','wardrobe_events','price_watches',
    'recommendation_feedback','wardrobe_item_lifecycle','user_style_preferences'
  ]) loop
    execute format(
      'create policy %I_owner_all on public.%I for all using (auth.uid() = user_id) with check (auth.uid() = user_id)',
      t, t);
  end loop;
end $$;

-- Explicit least-privilege grants: do not depend on hosted default privileges.
revoke all on public.style_evidence from anon, authenticated;
grant select, insert, update, delete on public.style_evidence to authenticated;
grant all on public.style_evidence to service_role;
revoke all on public.style_profiles from anon, authenticated;
grant select, insert, update, delete on public.style_profiles to authenticated;
grant all on public.style_profiles to service_role;
revoke all on public.outfit_history from anon, authenticated;
grant select, insert, update, delete on public.outfit_history to authenticated;
grant all on public.outfit_history to service_role;
revoke all on public.wardrobe_goals from anon, authenticated;
grant select, insert, update, delete on public.wardrobe_goals to authenticated;
grant all on public.wardrobe_goals to service_role;
revoke all on public.purchase_decisions from anon, authenticated;
grant select, insert, update, delete on public.purchase_decisions to authenticated;
grant all on public.purchase_decisions to service_role;
revoke all on public.purchase_outcomes from anon, authenticated;
grant select, insert, update, delete on public.purchase_outcomes to authenticated;
grant all on public.purchase_outcomes to service_role;
revoke all on public.wardrobe_events from anon, authenticated;
grant select, insert, update, delete on public.wardrobe_events to authenticated;
grant all on public.wardrobe_events to service_role;
revoke all on public.price_watches from anon, authenticated;
grant select, insert, update, delete on public.price_watches to authenticated;
grant all on public.price_watches to service_role;
revoke all on public.recommendation_feedback from anon, authenticated;
grant select, insert, update, delete on public.recommendation_feedback to authenticated;
grant all on public.recommendation_feedback to service_role;
revoke all on public.wardrobe_item_lifecycle from anon, authenticated;
grant select, insert, update, delete on public.wardrobe_item_lifecycle to authenticated;
grant all on public.wardrobe_item_lifecycle to service_role;
revoke all on public.user_style_preferences from anon, authenticated;
grant select, insert, update, delete on public.user_style_preferences to authenticated;
grant all on public.user_style_preferences to service_role;
