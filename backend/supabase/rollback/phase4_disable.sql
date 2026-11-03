-- Approved activation rollback only: revoke new client paths; retain every row and index.
-- Restore grants by re-running the grant section of the pinned forward migration.
begin;
revoke all on public.visual_assets from anon, authenticated;
revoke all on public.visual_evidence from anon, authenticated;
revoke all on public.reference_looks from anon, authenticated;
revoke all on public.inference_cache_ledger from anon, authenticated;
revoke all on public.tryon_requests from anon, authenticated;
revoke all on public.stylist_plans from anon, authenticated;
commit;
