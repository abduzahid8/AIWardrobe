-- Equality on owner, then the immutable ID cursor and result order.
-- Existing owner/date indexes still serve chronological queries elsewhere.
set local lock_timeout = '3s';
set local statement_timeout = '30s';
create index if not exists clothing_items_owner_cursor_idx
  on public.clothing_items(user_id, id desc);
create index if not exists wear_logs_owner_cursor_idx
  on public.wear_logs(user_id, id desc);
