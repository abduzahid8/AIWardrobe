-- Real roles and synthetic profiles in a single rolled-back live transaction.
begin;
set local lock_timeout='3s';
set local statement_timeout='15s';
select set_config('readiness.owner',gen_random_uuid()::text,true);
select set_config('readiness.other',gen_random_uuid()::text,true);
select set_config('request.jwt.claims','{"role":"service_role"}',true);
insert into auth.users(id,email,raw_user_meta_data) values
 (current_setting('readiness.owner')::uuid,'status-'||current_setting('readiness.owner')||'@example.invalid','{}'),
 (current_setting('readiness.other')::uuid,'status-'||current_setting('readiness.other')||'@example.invalid','{}');
do $$declare n int;begin
 select count(*) into n from public.profiles where id in(current_setting('readiness.owner')::uuid,current_setting('readiness.other')::uuid);
 if n<>2 then raise exception 'Auth profile trigger failed';end if;
end $$;
update public.profiles set subscription_tier='premium',subscription_expires_at=now()+interval '1 day',locked_until=now()+interval '1 hour'
 where id in(current_setting('readiness.owner')::uuid,current_setting('readiness.other')::uuid);
set local role authenticated;
select set_config('request.jwt.claims',json_build_object('role','authenticated','sub',current_setting('readiness.owner'))::text,true);
do $$begin
 if public.has_premium_access(current_setting('readiness.owner')::uuid) is distinct from true then raise exception 'Own premium access failed';end if;
 if public.is_account_locked(current_setting('readiness.owner')::uuid) is distinct from true then raise exception 'Own lock status failed';end if;
 if public.has_premium_access(current_setting('readiness.other')::uuid) is distinct from false then raise exception 'Cross-owner premium status leaked';end if;
 if public.is_account_locked(current_setting('readiness.other')::uuid) is distinct from false then raise exception 'Cross-owner lock status leaked';end if;
 if public.has_premium_access(gen_random_uuid()) is distinct from false then raise exception 'Missing user considered premium';end if;
end $$;
set local role postgres;
select set_config('request.jwt.claims','{"role":"service_role"}',true);
update public.profiles set subscription_tier='vip',subscription_expires_at=now()-interval '1 day',locked_until=now()-interval '1 hour' where id=current_setting('readiness.owner')::uuid;
set local role authenticated;
select set_config('request.jwt.claims',json_build_object('role','authenticated','sub',current_setting('readiness.owner'))::text,true);
do $$begin
 if public.has_premium_access(current_setting('readiness.owner')::uuid) is distinct from false then raise exception 'Expired paid status accepted';end if;
 if public.is_account_locked(current_setting('readiness.owner')::uuid) is distinct from false then raise exception 'Expired lock accepted';end if;
end $$;
set local role postgres;
select set_config('request.jwt.claims','{"role":"service_role"}',true);
update public.profiles set subscription_tier='free',subscription_expires_at=null,locked_until=null where id=current_setting('readiness.owner')::uuid;
set local role authenticated;
select set_config('request.jwt.claims',json_build_object('role','authenticated','sub',current_setting('readiness.owner'))::text,true);
do $$begin
 if public.has_premium_access(current_setting('readiness.owner')::uuid) is distinct from false then raise exception 'Free profile considered premium';end if;
 if public.is_account_locked(current_setting('readiness.owner')::uuid) is distinct from false then raise exception 'Unlocked profile considered locked';end if;
end $$;
set local role service_role;
select set_config('request.jwt.claims','{"role":"service_role"}',true);
do $$begin
 if public.has_premium_access(current_setting('readiness.other')::uuid) is distinct from true then raise exception 'Service verification broken';end if;
 if public.is_account_locked(current_setting('readiness.other')::uuid) is distinct from true then raise exception 'Service lock verification broken';end if;
end $$;
set local role anon;
select set_config('request.jwt.claims','{"role":"anon"}',true);
do $$begin
 begin perform public.has_premium_access(gen_random_uuid());raise exception 'Anonymous premium helper allowed';exception when insufficient_privilege then null;end;
 begin perform public.is_account_locked(gen_random_uuid());raise exception 'Anonymous lock helper allowed';exception when insufficient_privilege then null;end;
 if has_function_privilege('anon','public.handle_new_user()','execute') then raise exception 'Trigger function remains anonymous RPC';end if;
 if has_function_privilege('authenticated','public.handle_new_user()','execute') then raise exception 'Trigger function remains authenticated RPC';end if;
end $$;
select 'passed: trigger creation, own and service status, cross-owner isolation, missing/expired/free status, anonymous RPC rejection' as assertion;
rollback;
