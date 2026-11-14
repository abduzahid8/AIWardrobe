-- Optional capability rollback ONLY AFTER restoring the verified v195 Edge
-- source. v195 tolerates a missing RPC and resumes its legacy bounded path.
-- Additive derived metadata is retained; original catalog/RLS are untouched.
begin;
drop function if exists public.catalog_discovery_candidates(text,text,text,text,integer,integer,text,numeric,jsonb);
notify pgrst,'reload schema';
commit;
