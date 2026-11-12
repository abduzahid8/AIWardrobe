-- Stylist 6 feedback taste (mission gate D): make explicit thumbs feedback
-- linkable to future rankings. The existing stylist_look_feedback rows carry
-- only request/look/concept identity, so the ranker could never connect a
-- like/dislike to the items or styling direction the user rated.
--
-- Additive only, in the ADD COLUMN IF NOT EXISTS style: no existing column,
-- constraint, grant or RLS policy changes. RLS stays own-row-only
-- (feedback_own_read/insert/update/delete from 20261107230000); the Stylist 6
-- cloud handler reads through the service-role key, exactly like inventory.
--
-- item_ids      text[] — the server-issued item ids of the rated look
--               (<= 8 entries, write path bounds them). Exact item identity
--               is the primary linking key for per-item taste adjustments.
-- look_signature jsonb — versioned styling-feature document written by the
--               client from src/domain/outfitVisualSignature.ts:
--               { version:1, direction:'signature'|'refresh'|'discovery'|
--               'alternative'|null, signature:'sq1:...'|null,
--               features:[[subtype,volume,color,texture,pattern]|null x4] }.
--               The sq1 signature and per-slot features are the item-id-
--               independent styling identity of the rated look, recorded for
--               attribution/diagnostics; scoring uses explicit ratings, item
--               ids, concept ids and direction labels only.
--
-- Read path: the cloud handler reads the last 60 rows for the JWT user with
-- `order created_at desc` — already served by feedback_owner_recent
-- (user_id, created_at desc) created in 20261107230000, so no new index here.
--
-- Apply only this reviewed migration with the Supabase migration API; hosted
-- migration application is a deploy-time step (no local postgres is required
-- or used to validate it).
begin;
set local lock_timeout='2s';
alter table public.stylist_look_feedback
  add column if not exists item_ids text[] not null default '{}';
alter table public.stylist_look_feedback
  add column if not exists look_signature jsonb not null default '{}';
do $$
begin
  if not exists (
    select 1 from pg_constraint
    where conname = 'stylist_feedback_item_ids_bounded'
      and conrelid = 'public.stylist_look_feedback'::regclass
  ) then
    alter table public.stylist_look_feedback
      add constraint stylist_feedback_item_ids_bounded check (cardinality(item_ids) <= 8);
  end if;
end $$;
notify pgrst,'reload schema';
commit;

-- Down/rollback (additive columns only; dropping them loses no pre-existing
-- data and restores the exact 20261107230000 shape):
--   begin;
--   set local lock_timeout='2s';
--   alter table public.stylist_look_feedback drop column if exists look_signature;
--   alter table public.stylist_look_feedback drop column if exists item_ids;
--   notify pgrst,'reload schema';
--   commit;
