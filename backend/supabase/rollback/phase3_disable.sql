-- Approved activation rollback only: revoke new client paths; retain every row and index.
-- Restore grants by re-running the grant section of the pinned forward migration.
begin;
revoke all on public.style_evidence from anon, authenticated;
revoke all on public.style_profiles from anon, authenticated;
revoke all on public.outfit_history from anon, authenticated;
revoke all on public.wardrobe_goals from anon, authenticated;
revoke all on public.purchase_decisions from anon, authenticated;
revoke all on public.purchase_outcomes from anon, authenticated;
revoke all on public.wardrobe_events from anon, authenticated;
revoke all on public.price_watches from anon, authenticated;
revoke all on public.recommendation_feedback from anon, authenticated;
revoke all on public.wardrobe_item_lifecycle from anon, authenticated;
revoke all on public.user_style_preferences from anon, authenticated;
commit;
