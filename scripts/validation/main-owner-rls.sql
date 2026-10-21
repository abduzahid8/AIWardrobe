-- Two synthetic users, confined to one rolled-back transaction. No real account data is read.
begin;
set local lock_timeout='3s';
set local statement_timeout='15s';
select set_config('readiness.owner',gen_random_uuid()::text,true);
select set_config('readiness.other',gen_random_uuid()::text,true);
insert into auth.users(id,email,raw_user_meta_data) values
 (current_setting('readiness.owner')::uuid,'rls-'||current_setting('readiness.owner')||'@example.invalid','{}'),
 (current_setting('readiness.other')::uuid,'rls-'||current_setting('readiness.other')||'@example.invalid','{}');
insert into public.profiles(id,email) values
 (current_setting('readiness.owner')::uuid,'rls-'||current_setting('readiness.owner')||'@example.invalid'),
 (current_setting('readiness.other')::uuid,'rls-'||current_setting('readiness.other')||'@example.invalid') on conflict(id) do nothing;
select set_config('request.jwt.claims',json_build_object('sub',current_setting('readiness.owner'),'role','authenticated')::text,true);
set local role authenticated;
do $$
declare t text; n int;
begin
 foreach t in array array['clothing_items','saved_outfits','wear_logs'] loop
  if t='wear_logs' then
   execute format('insert into public.%I(user_id,date) values ($1,$2)',t) using current_setting('readiness.owner')::uuid,'2026-10-20';
  else
   execute format('insert into public.%I(user_id) values ($1)',t) using current_setting('readiness.owner')::uuid;
  end if;
  execute format('select count(*) from public.%I where user_id=$1',t) into n using current_setting('readiness.owner')::uuid;
  if n<>1 then raise exception 'Own SELECT failed for %',t; end if;
  execute format('update public.%I set user_id=user_id where user_id=$1',t) using current_setting('readiness.owner')::uuid;
  get diagnostics n=row_count;if n<>1 then raise exception 'Own UPDATE failed for %',t;end if;
  begin
   execute format('update public.%I set user_id=$1 where user_id=$2',t) using current_setting('readiness.other')::uuid,current_setting('readiness.owner')::uuid;
   raise exception 'Owner reassignment accepted for %',t;
  exception when insufficient_privilege then null;end;
  begin
   if t='wear_logs' then execute format('insert into public.%I(user_id,date) values ($1,$2)',t) using current_setting('readiness.other')::uuid,'2026-10-20';
   else execute format('insert into public.%I(user_id) values ($1)',t) using current_setting('readiness.other')::uuid;end if;
   raise exception 'Cross-owner INSERT accepted for %',t;
  exception when insufficient_privilege then null;end;
 end loop;
end $$;
select set_config('request.jwt.claims',json_build_object('sub',current_setting('readiness.other'),'role','authenticated')::text,true);
do $$declare t text;n int;begin
 foreach t in array array['clothing_items','saved_outfits','wear_logs'] loop
  execute format('select count(*) from public.%I where user_id=$1',t) into n using current_setting('readiness.owner')::uuid;
  if n<>0 then raise exception 'Cross-owner SELECT accepted for %',t;end if;
  execute format('delete from public.%I where user_id=$1',t) using current_setting('readiness.owner')::uuid;
  get diagnostics n=row_count;if n<>0 then raise exception 'Cross-owner DELETE accepted for %',t;end if;
 end loop;
 select count(*) into n from public.profiles where id=current_setting('readiness.owner')::uuid;
 if n<>0 then raise exception 'Cross-owner profile SELECT accepted';end if;
end $$;
select set_config('request.jwt.claims',json_build_object('sub',current_setting('readiness.owner'),'role','authenticated')::text,true);
do $$declare t text;n int;begin
 foreach t in array array['clothing_items','saved_outfits','wear_logs'] loop
  execute format('delete from public.%I where user_id=$1',t) using current_setting('readiness.owner')::uuid;
  get diagnostics n=row_count;if n<>1 then raise exception 'Own DELETE failed for %',t;end if;
 end loop;
 select count(*) into n from public.profiles where id=current_setting('readiness.owner')::uuid;
 if n<>1 then raise exception 'Own profile SELECT failed';end if;
 update public.profiles set username='readiness_rls' where id=current_setting('readiness.owner')::uuid;
 get diagnostics n=row_count;if n<>1 then raise exception 'Own profile UPDATE failed';end if;
end $$;
select 'passed: owner CRUD, cross-owner read/insert/delete rejection, owner reassignment rejection and profile read/update' as assertion;
rollback;
