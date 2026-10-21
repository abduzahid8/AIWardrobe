-- Isolated database only. All fixtures roll back, including on assertion failure.
\set ON_ERROR_STOP on
begin;
insert into auth.users(id,email,raw_user_meta_data) values
('11111111-1111-4111-8111-111111111111','wear-owner@example.invalid','{"username":"wear-owner"}'),
('22222222-2222-4222-8222-222222222222','wear-other@example.invalid','{"username":"wear-other"}');
insert into public.clothing_items(id,user_id) values
('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1','11111111-1111-4111-8111-111111111111'),
('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2','11111111-1111-4111-8111-111111111111'),
('bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb1','22222222-2222-4222-8222-222222222222');
set local role authenticated;
set local request.jwt.claim.sub = '11111111-1111-4111-8111-111111111111';
insert into public.wear_logs(id,user_id,item_ids,date,created_at) values
('cccccccc-cccc-4ccc-8ccc-ccccccccccc1','11111111-1111-4111-8111-111111111111',
'["aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1","aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1","bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb1","invalid",null,{}]',
'2026-10-15','2026-10-15T12:00:00Z');
do $$ begin
 if (select wear_count from public.clothing_items where id='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1') <> 1 then raise exception 'Insert/duplicate normalization failed'; end if;
end $$;
-- A retry must be idempotent even if metadata changes.
insert into public.wear_logs(id,user_id,item_ids,date,created_at) values
('cccccccc-cccc-4ccc-8ccc-ccccccccccc1','11111111-1111-4111-8111-111111111111','["aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"]','2026-10-15','2026-10-15T12:00:00Z')
on conflict(id) do update set item_ids=excluded.item_ids,occasion='retry';
update public.clothing_items set wear_count=900,last_worn_at='2099-01-01';
do $$ begin
 if (select wear_count from public.clothing_items where id='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1') <> 1 then raise exception 'Retry or direct statistics overwrite failed'; end if;
 if (select last_worn_at from public.clothing_items where id='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1') <> '2026-10-15T12:00:00Z'::timestamptz then raise exception 'Direct timestamp overwrite allowed'; end if;
end $$;
insert into public.wear_logs(id,user_id,item_ids,date,created_at) values
('cccccccc-cccc-4ccc-8ccc-ccccccccccc2','11111111-1111-4111-8111-111111111111','["aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"]','2026-10-16','2026-10-16T12:00:00Z');
update public.wear_logs set item_ids='["aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2"]' where id='cccccccc-cccc-4ccc-8ccc-ccccccccccc2';
do $$ begin
 if (select wear_count from public.clothing_items where id='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1') <> 1 then raise exception 'Old membership decrement failed'; end if;
 if (select wear_count from public.clothing_items where id='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2') <> 1 then raise exception 'New membership increment failed'; end if;
 if (select last_worn_at from public.clothing_items where id='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1') <> '2026-10-15T12:00:00Z'::timestamptz then raise exception 'Latest timestamp correction failed'; end if;
end $$;
delete from public.wear_logs;
do $$ begin
 if exists(select 1 from public.clothing_items where wear_count<>0 or last_worn_at is not null) then raise exception 'Delete failed to clear counters'; end if;
end $$;
reset role;
do $$ begin
 if (select wear_count from public.clothing_items where id='bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb1') <> 0 then raise exception 'Cross-owner counter changed'; end if;
end $$;
rollback;
\echo Wear counter transaction and ownership assertions passed
