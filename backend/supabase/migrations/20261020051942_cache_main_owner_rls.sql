-- Keep existing roles, commands and ownership predicates. Cache stable caller identity per statement.
-- Explicit UPDATE WITH CHECK is equivalent to the previous implicit USING fallback.
set local lock_timeout='3s';
set local statement_timeout='30s';
do $$
declare p record; existing record; predicate text;
begin
 for p in select * from (values
  ('clothing_items', 'Users can view own items', 'SELECT', 'user_id'),
  ('clothing_items', 'Users can insert own items', 'INSERT', 'user_id'),
  ('clothing_items', 'Users can update own items', 'UPDATE', 'user_id'),
  ('clothing_items', 'Users can delete own items', 'DELETE', 'user_id'),
  ('saved_outfits', 'Users can view own outfits', 'SELECT', 'user_id'),
  ('saved_outfits', 'Users can insert own outfits', 'INSERT', 'user_id'),
  ('saved_outfits', 'Users can update own outfits', 'UPDATE', 'user_id'),
  ('saved_outfits', 'Users can delete own outfits', 'DELETE', 'user_id'),
  ('wear_logs', 'Users can view own wear logs', 'SELECT', 'user_id'),
  ('wear_logs', 'Users can insert own wear logs', 'INSERT', 'user_id'),
  ('wear_logs', 'Users can update own wear logs', 'UPDATE', 'user_id'),
  ('wear_logs', 'Users can delete own wear logs', 'DELETE', 'user_id'),
  ('profiles', 'Users can view own profile', 'SELECT', 'id'),
  ('profiles', 'Users can insert own profile', 'INSERT', 'id'),
  ('profiles', 'Users can update own profile', 'UPDATE', 'id')
 ) as policies(tablename,policyname,command,owner_column) loop
  select cmd,qual,with_check into strict existing from pg_policies
   where schemaname='public' and tablename=p.tablename and policyname=p.policyname;
  if existing.cmd<>p.command or
     existing.qual is distinct from (case when p.command='INSERT' then null else '(auth.uid() = '||p.owner_column||')' end) or
     existing.with_check is distinct from (case when p.command='INSERT' then '(auth.uid() = '||p.owner_column||')' else null end) then
   raise exception 'Unexpected policy definition for %.%; review before changing it',p.tablename,p.policyname;
  end if;
  predicate=format('((select auth.uid()) = %I)',p.owner_column);
  if p.command='INSERT' then
   execute format('alter policy %I on public.%I with check (%s)',p.policyname,p.tablename,predicate);
  elsif p.command='UPDATE' then
   execute format('alter policy %I on public.%I using (%s) with check (%s)',p.policyname,p.tablename,predicate,predicate);
  else
   execute format('alter policy %I on public.%I using (%s)',p.policyname,p.tablename,predicate);
  end if;
 end loop;
end $$;
