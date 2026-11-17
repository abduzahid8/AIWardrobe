-- Approved activation rollback only: revoke new client paths; retain every row and index.
-- Restore grants by re-running the grant section of the pinned forward migration.
begin;
revoke all on public.growth_activation_state from anon, authenticated;
revoke all on public.growth_notification_prefs from anon, authenticated;
revoke all on public.growth_referrals from anon, authenticated;
revoke all on public.growth_savings_ledger from anon, authenticated;
revoke all on public.growth_feedback from anon, authenticated;
commit;
