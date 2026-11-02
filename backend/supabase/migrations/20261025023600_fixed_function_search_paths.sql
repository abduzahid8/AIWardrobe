-- Pin object resolution without changing bodies, grants or execution roles.
set lock_timeout='3s';
set statement_timeout='15s';
alter function public.featured_capsules_set_updated_at() set search_path='';
alter function public.record_brand_click(text,text,text,numeric,text,text,text) set search_path='';
alter function public.set_updated_at() set search_path='';
alter function public.update_guide_page_updated_at() set search_path='';
alter function public.update_home_occasions_updated_at() set search_path='';
-- This legacy helper exists on some deployed projects but is not created by
-- the numbered migration chain used for a fresh database.
do $body$
begin
  if to_regprocedure('public.update_updated_at_column()') is not null then
    execute format('alter function %s set search_path = %L', 'public.update_updated_at_column()'::regprocedure, '');
  end if;
end
$body$;
