-- Isolated readiness database only; run after seed-capacity.sql.
insert into public.clothing_items(id,user_id,category)
select md5('cursor-heavy-item-' || n)::uuid, md5('load-user-1')::uuid, 'top'
from generate_series(1,10000) n on conflict do nothing;
insert into public.wear_logs(id,user_id,item_ids,date,created_at)
select md5('cursor-log-' || u || '-' || n)::uuid, md5('load-user-' || u)::uuid,
       '[]'::jsonb, '2026-10-16', '2026-10-16T12:00:00Z'::timestamptz
from generate_series(1,10000) u cross join generate_series(1,10) n
on conflict do nothing;
insert into public.wear_logs(id,user_id,item_ids,date,created_at)
select md5('cursor-heavy-log-' || n)::uuid, md5('load-user-1')::uuid,
       '[]'::jsonb, '2026-10-16', '2026-10-16T12:00:00Z'::timestamptz
from generate_series(1,10000) n on conflict do nothing;
analyze public.clothing_items;
analyze public.wear_logs;
