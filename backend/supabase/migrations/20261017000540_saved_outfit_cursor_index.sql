set local lock_timeout = '3s';
set local statement_timeout = '30s';
create index if not exists saved_outfits_owner_cursor_idx
  on public.saved_outfits(user_id, id desc);
