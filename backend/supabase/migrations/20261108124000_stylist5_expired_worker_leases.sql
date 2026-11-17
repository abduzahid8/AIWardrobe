-- Expired runners cannot strand rows at the bounded attempt limit forever.
begin;
create or replace function public.catalog_visual_expired_lease() returns trigger
language plpgsql security invoker set search_path='' as $$
begin
 if old.state='leased' and new.state='pending' and new.attempts>=3 then
  new.state:='unknown';new.reason:='inference_failed';new.updated_at:=now();
 end if;
 return new;
end;$$;
revoke all on function public.catalog_visual_expired_lease() from public,anon,authenticated;
grant execute on function public.catalog_visual_expired_lease() to service_role;
create trigger catalog_visual_expired_lease before update of state on public.catalog_visual_jobs
 for each row execute function public.catalog_visual_expired_lease();
commit;
