-- Approved activation rollback only: revoke new client paths; retain every row and index.
-- Restore grants by re-running the grant section of the pinned forward migration.
begin;
revoke all on public.decision_events from anon, authenticated;
revoke all on public.experiment_assignments from anon, authenticated;
revoke all on public.recommendation_outcomes from anon, authenticated;
commit;
