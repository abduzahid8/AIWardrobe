-- Apply only after separately approved rollback. Clients tolerate a missing RPC.
begin;
drop function if exists public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric);
commit;
