-- JWTs survive Auth deletion until expiry. Require the signed session to exist
-- for authenticated Storage operations, in addition to every existing policy.
create schema app_security;
revoke all on schema app_security from public;
grant usage on schema app_security to authenticated, service_role;

create function app_security.has_active_session() returns boolean
language plpgsql stable security definer set search_path = '' as $$
declare session_claim text := auth.jwt() ->> 'session_id';
begin
  if session_claim is null or session_claim !~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$' then
    return false;
  end if;
  return exists (
    select 1 from auth.sessions s
    where s.id = session_claim::uuid and s.user_id = (select auth.uid())
  );
end;
$$;
revoke all on function app_security.has_active_session() from public, anon;
grant execute on function app_security.has_active_session() to authenticated, service_role;

create policy authenticated_storage_requires_active_session
on storage.objects as restrictive for all to authenticated
using ((select app_security.has_active_session()))
with check ((select app_security.has_active_session()));
