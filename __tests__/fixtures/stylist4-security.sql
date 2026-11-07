\set ON_ERROR_STOP on
begin;
insert into auth.users(id) values('00000000-0000-4000-8000-000000000401'),('00000000-0000-4000-8000-000000000402') on conflict do nothing;
set local role authenticated;
select set_config('request.jwt.claim.sub','00000000-0000-4000-8000-000000000401',true);
insert into public.stylist_look_feedback(request_id,look_id,concept_id,engine_version,rating)
values('00000000-0000-4000-8000-000000000411','outfit_1','refined-ivy','stylist-4-test',1);
-- Exercise the actual mobile request-key shape as well as old UUID callers.
insert into public.stylist_look_feedback(request_id,look_id,concept_id,engine_version,rating)
values('moi0kaaa-1','outfit_2','refined-ivy','stylist-4-test',1);
do $$ begin
  if has_table_privilege(current_user,'public.catalog_visual_fashion','INSERT') or has_table_privilege(current_user,'public.catalog_image_embeddings','SELECT')
    or has_function_privilege(current_user,'public.catalog_stylist_candidates(text[],text,jsonb,integer,integer,text,numeric)','EXECUTE') then raise exception 'client privilege escalation';end if;
  begin
    insert into public.stylist_look_feedback(user_id,request_id,look_id,concept_id,engine_version,rating)
    values('00000000-0000-4000-8000-000000000402','00000000-0000-4000-8000-000000000412','outfit_1','refined-ivy','test',1);
    raise exception 'other-owner insert accepted';
  exception when insufficient_privilege then null;end;
end $$;
select set_config('request.jwt.claim.sub','00000000-0000-4000-8000-000000000402',true);
do $$ begin
  if exists(select 1 from public.stylist_look_feedback) then raise exception 'other-owner feedback visible';end if;
  update public.stylist_look_feedback set rating=-1;
  if found then raise exception 'other-owner feedback mutable';end if;
end $$;
reset role;
set local role service_role;
do $$ begin
  begin perform public.catalog_stylist_candidates(array['invalid'],'test');raise exception 'invalid slot accepted';
    exception when invalid_parameter_value then null;end;
  begin perform public.catalog_stylist_candidates(array['top'],'test','{}',100,200);raise exception 'unbounded offset accepted';
    exception when invalid_parameter_value then null;end;
  begin perform public.catalog_stylist_candidates(array['top'],'test','{}',101,0);raise exception 'unbounded limit accepted';
    exception when invalid_parameter_value then null;end;
end $$;
reset role;
do $$ begin
  if exists(select 1 from pg_proc where oid='public.catalog_stylist_candidates(text[],text,jsonb,integer,integer,text,numeric)'::regprocedure and prosecdef) then raise exception 'definer retrieval';end if;
  if exists(select 1 from pg_class where oid in ('public.stylist_look_feedback'::regclass,'public.catalog_visual_fashion'::regclass,'public.catalog_image_embeddings'::regclass) and not relrowsecurity) then raise exception 'RLS missing';end if;
end $$;
select 'PASS: feedback ownership, private embeddings, read-only visual cache, bounded invoker retrieval' result;
rollback;
