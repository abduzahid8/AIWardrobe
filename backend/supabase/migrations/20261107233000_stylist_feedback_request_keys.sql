-- The app's generationRequestId is a bounded time/sequence key, not a UUID.
-- Existing UUID keys remain valid text; ownership, primary key and policies persist.
begin;
set local lock_timeout='2s';
alter table public.stylist_look_feedback alter column request_id type text using request_id::text;
alter table public.stylist_look_feedback add constraint stylist_feedback_request_key_length
  check(length(request_id) between 1 and 100);
commit;
