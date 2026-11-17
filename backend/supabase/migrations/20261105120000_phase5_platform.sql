-- Phase 5 Production Platform — additive local migration (DO NOT APPLY
-- REMOTELY without explicit approval). Durable tables for decision events
-- (sanitized), experiment assignments, and recommendation outcomes.
-- All tables user-scoped with RLS; no backfill, no DDL on existing tables.

-- ============ decision_events (5A, sanitized only) ============
create table if not exists public.decision_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  event_name text not null,
  request_id text not null,
  trace_id text,
  feature text not null,
  decision_type text not null,
  evidence_summary text[] not null default '{}',
  model_provider text,
  model_version text,
  rules_version text not null default 'v1',
  confidence double precision check (confidence is null or (confidence >= 0 and confidence <= 1)),
  candidate_count int,
  latency_ms int,
  fallback_used text,
  experiment_assignment text,
  outcome text,
  error_class text,
  created_at timestamptz not null default now()
);
create index if not exists decision_events_user_feature_idx
  on public.decision_events (user_id, feature, created_at desc);

-- ============ experiment_assignments (5J, deterministic local mirror) ============
create table if not exists public.experiment_assignments (
  experiment_id text not null,
  user_id uuid not null references auth.users (id) on delete cascade,
  variant text not null,
  assigned_at timestamptz not null default now(),
  primary key (experiment_id, user_id)
);

-- ============ recommendation_outcomes (5AC attribution) ============
create table if not exists public.recommendation_outcomes (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  recommendation_id text not null,
  verdict text not null,
  outcome text not null,
  days_to_outcome int,
  confidence text not null default 'low' check (confidence in ('low','medium','high')),
  note text not null default '',
  created_at timestamptz not null default now()
);
create index if not exists recommendation_outcomes_rec_idx
  on public.recommendation_outcomes (user_id, recommendation_id);

-- ============ RLS (user-scoped; deny-by-default like sibling migrations) ============
alter table public.decision_events enable row level security;
alter table public.experiment_assignments enable row level security;
alter table public.recommendation_outcomes enable row level security;

do $$
declare t text;
begin
  for t in select unnest(array[
    'decision_events','experiment_assignments','recommendation_outcomes'
  ]) loop
    execute format(
      'create policy %I_owner_all on public.%I for all using (auth.uid() = user_id) with check (auth.uid() = user_id)',
      t, t);
  end loop;
end $$;

-- Explicit least-privilege grants: do not depend on hosted default privileges.
revoke all on public.decision_events from anon, authenticated;
grant select, insert, update, delete on public.decision_events to authenticated;
grant all on public.decision_events to service_role;
revoke all on public.experiment_assignments from anon, authenticated;
grant select, insert, update, delete on public.experiment_assignments to authenticated;
grant all on public.experiment_assignments to service_role;
revoke all on public.recommendation_outcomes from anon, authenticated;
grant select, insert, update, delete on public.recommendation_outcomes to authenticated;
grant all on public.recommendation_outcomes to service_role;
