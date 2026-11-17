-- Phase 4 Multimodal Stylist — additive local migration (DO NOT APPLY
-- REMOTELY without explicit approval). Durable tables for visual assets,
-- visual evidence (with user-correction supersession), reference looks,
-- inference cache ledger, try-on requests, and stylist plans.
-- All tables user-scoped with RLS; no backfill, no DDL on existing tables.

-- ============ visual_assets (4B) ============
create table if not exists public.visual_assets (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  kind text not null check (kind in (
    'original_upload','cropped_garment','background_removed','thumbnail',
    'normalized','tryon_result','inspiration','outfit_photo','merchant_image')),
  original_asset_id uuid references public.visual_assets (id) on delete set null,
  transformation_chain text[] not null default '{}',
  local_uri text,
  remote_uri text,
  content_hash text,
  width int,
  height int,
  mime_type text,
  source text not null default 'user' check (source in ('user','merchant','generated','import')),
  retention_class text not null default 'durable' check (retention_class in ('ephemeral','session','durable')),
  user_owned boolean not null default true,
  created_at timestamptz not null default now()
);
create index if not exists visual_assets_user_kind_idx
  on public.visual_assets (user_id, kind, created_at desc);

-- ============ visual_evidence (4A) ============
create table if not exists public.visual_evidence (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  source_asset_id uuid not null references public.visual_assets (id) on delete cascade,
  entity_type text not null check (entity_type in (
    'wardrobe_item','catalog_product','outfit','inspiration','mirror_photo','receipt_or_page')),
  entity_id text,
  attribute_type text not null check (attribute_type in (
    'garment_type','category','color','pattern','material','silhouette',
    'fit_intent','formality','style_tags','brand','condition','layering',
    'outfit_role','visual_similarity')),
  value text not null default 'unknown',
  confidence double precision not null default 0 check (confidence >= 0 and confidence <= 1),
  source text not null check (source in (
    'user','metadata','deterministic_image_analysis','local_model','remote_model','merchant')),
  model_version text,
  superseded_by uuid references public.visual_evidence (id) on delete set null,
  user_confirmed boolean not null default false,
  user_rejected boolean not null default false,
  created_at timestamptz not null default now()
);
create index if not exists visual_evidence_asset_attr_idx
  on public.visual_evidence (source_asset_id, attribute_type, created_at desc);

-- ============ reference_looks (4M) ============
create table if not exists public.reference_looks (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  asset_id uuid references public.visual_assets (id) on delete set null,
  source_label text not null default 'import',
  style_representation jsonb not null default '{}',
  user_rating int check (user_rating is null or (user_rating >= 1 and user_rating <= 5)),
  contexts text[] not null default '{}',
  recreated_outfit_ids text[] not null default '{}',
  match_confidence text check (match_confidence in ('low','medium','high')),
  created_at timestamptz not null default now()
);

-- ============ inference_cache_ledger (§29, server-side counterpart) ============
create table if not exists public.inference_cache_ledger (
  cache_key text primary key,
  user_id uuid not null references auth.users (id) on delete cascade,
  task text not null,
  model_version text not null,
  schema_version text not null default 'v1',
  result jsonb not null default '{}',
  cached_at timestamptz not null default now()
);

-- ============ tryon_requests (4S/4T) ============
create table if not exists public.tryon_requests (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  person_asset_id uuid references public.visual_assets (id) on delete set null,
  garment_variant_id text not null,
  garment_image_asset_id uuid references public.visual_assets (id) on delete set null,
  category text not null,
  preservation_constraints text[] not null default '{}',
  provider text not null default 'default',
  quality_preset text not null default 'standard' check (quality_preset in ('draft','standard','high')),
  status text not null default 'pending',
  output_asset_id uuid references public.visual_assets (id) on delete set null,
  synthetic boolean not null default true,
  created_at timestamptz not null default now()
);

-- ============ stylist_plans (4W) ============
create table if not exists public.stylist_plans (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  intent text not null,
  tools_selected text[] not null default '{}',
  candidate_counts jsonb not null default '{}',
  validation_stages text[] not null default '{}',
  evidence_summary text[] not null default '{}',
  created_at timestamptz not null default now()
);

-- ============ RLS (user-scoped; deny-by-default like sibling migrations) ============
alter table public.visual_assets enable row level security;
alter table public.visual_evidence enable row level security;
alter table public.reference_looks enable row level security;
alter table public.inference_cache_ledger enable row level security;
alter table public.tryon_requests enable row level security;
alter table public.stylist_plans enable row level security;

do $$
declare t text;
begin
  for t in select unnest(array[
    'visual_assets','visual_evidence','reference_looks',
    'inference_cache_ledger','tryon_requests','stylist_plans'
  ]) loop
    execute format(
      'create policy %I_owner_all on public.%I for all using (auth.uid() = user_id) with check (auth.uid() = user_id)',
      t, t);
  end loop;
end $$;

-- Explicit least-privilege grants: do not depend on hosted default privileges.
revoke all on public.visual_assets from anon, authenticated;
grant select, insert, update, delete on public.visual_assets to authenticated;
grant all on public.visual_assets to service_role;
revoke all on public.visual_evidence from anon, authenticated;
grant select, insert, update, delete on public.visual_evidence to authenticated;
grant all on public.visual_evidence to service_role;
revoke all on public.reference_looks from anon, authenticated;
grant select, insert, update, delete on public.reference_looks to authenticated;
grant all on public.reference_looks to service_role;
revoke all on public.inference_cache_ledger from anon, authenticated;
grant select, insert, update, delete on public.inference_cache_ledger to authenticated;
grant all on public.inference_cache_ledger to service_role;
revoke all on public.tryon_requests from anon, authenticated;
grant select, insert, update, delete on public.tryon_requests to authenticated;
grant all on public.tryon_requests to service_role;
revoke all on public.stylist_plans from anon, authenticated;
grant select, insert, update, delete on public.stylist_plans to authenticated;
grant all on public.stylist_plans to service_role;

-- Composite owner FKs prevent cross-user links, including service writes.
-- PostgreSQL 15+ supports column-specific SET NULL, retaining the row owner.
alter table public.visual_assets add constraint visual_assets_id_owner unique (id,user_id);
alter table public.visual_evidence add constraint visual_evidence_id_owner unique (id,user_id);
alter table public.visual_assets drop constraint visual_assets_original_asset_id_fkey;
alter table public.visual_assets add constraint visual_assets_original_owner_fkey
 foreign key (original_asset_id,user_id) references public.visual_assets(id,user_id) on delete set null (original_asset_id);
alter table public.visual_evidence drop constraint visual_evidence_source_asset_id_fkey;
alter table public.visual_evidence add constraint visual_evidence_source_owner_fkey
 foreign key (source_asset_id,user_id) references public.visual_assets(id,user_id) on delete cascade;
alter table public.visual_evidence drop constraint visual_evidence_superseded_by_fkey;
alter table public.visual_evidence add constraint visual_evidence_superseded_owner_fkey
 foreign key (superseded_by,user_id) references public.visual_evidence(id,user_id) on delete set null (superseded_by);
alter table public.reference_looks drop constraint reference_looks_asset_id_fkey;
alter table public.reference_looks add constraint reference_looks_asset_owner_fkey
 foreign key (asset_id,user_id) references public.visual_assets(id,user_id) on delete set null (asset_id);
alter table public.tryon_requests drop constraint tryon_requests_person_asset_id_fkey;
alter table public.tryon_requests add constraint tryon_requests_person_owner_fkey
 foreign key (person_asset_id,user_id) references public.visual_assets(id,user_id) on delete set null (person_asset_id);
alter table public.tryon_requests drop constraint tryon_requests_garment_image_asset_id_fkey;
alter table public.tryon_requests add constraint tryon_requests_garment_owner_fkey
 foreign key (garment_image_asset_id,user_id) references public.visual_assets(id,user_id) on delete set null (garment_image_asset_id);
alter table public.tryon_requests drop constraint tryon_requests_output_asset_id_fkey;
alter table public.tryon_requests add constraint tryon_requests_output_owner_fkey
 foreign key (output_asset_id,user_id) references public.visual_assets(id,user_id) on delete set null (output_asset_id);
