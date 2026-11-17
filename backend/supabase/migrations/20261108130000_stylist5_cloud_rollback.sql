-- Exact release rollback remains available without the operator's local cache.
begin;
create table if not exists public.stylist5_release_rollback (
 release_id text primary key,
 baseline_version integer not null,
 baseline_source_sha256 text not null check(baseline_source_sha256 ~ '^[a-f0-9]{64}$'),
 baseline_source text not null,
 baseline_retrieval text not null,
 baseline_visual jsonb not null check(jsonb_typeof(baseline_visual)='array'),
 candidate_source_sha256 text not null check(candidate_source_sha256 ~ '^[a-f0-9]{64}$'),
 deployed_version integer not null,
 deployed_commit text not null,
 created_at timestamptz not null default now()
);
alter table public.stylist5_release_rollback enable row level security;
revoke all on public.stylist5_release_rollback from public,anon,authenticated;
grant select,insert on public.stylist5_release_rollback to service_role;
commit;
