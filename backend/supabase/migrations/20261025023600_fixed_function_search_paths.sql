-- Pin object resolution without changing bodies, grants or execution roles.
set lock_timeout='3s';
set statement_timeout='15s';
alter function public.featured_capsules_set_updated_at() set search_path='';
alter function public.record_brand_click(text,text,text,numeric,text,text,text) set search_path='';
alter function public.set_updated_at() set search_path='';
alter function public.update_guide_page_updated_at() set search_path='';
alter function public.update_home_occasions_updated_at() set search_path='';
alter function public.update_updated_at_column() set search_path='';
