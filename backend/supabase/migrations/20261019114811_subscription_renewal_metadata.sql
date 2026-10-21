-- Older hosted schemas predate this field. NULL preserves unknown renewal state
-- for legacy subscriptions; verified RevenueCat snapshots set an explicit value.
alter table public.subscriptions add column if not exists auto_renew boolean;
