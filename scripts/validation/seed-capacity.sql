-- Synthetic data for the isolated readiness DB only.
insert into auth.users(id, email, raw_user_meta_data)
select md5('load-user-' || n)::uuid, 'load-' || n || '@example.invalid',
       jsonb_build_object('username', 'load-' || n, 'gender', 'prefer_not_to_say')
from generate_series(1, 10000) n
on conflict do nothing;

insert into public.clothing_items(id, user_id, category, image_url)
select md5('load-item-' || u || '-' || n)::uuid, md5('load-user-' || u)::uuid,
       case n % 4 when 0 then 'top' when 1 then 'bottom' when 2 then 'shoes' else 'outerwear' end,
       'https://example.invalid/item.jpg'
from generate_series(1, 10000) u cross join generate_series(1, 50) n
on conflict do nothing;

insert into public.saved_outfits(id, user_id, items)
select md5('load-outfit-' || u || '-' || n)::uuid, md5('load-user-' || u)::uuid, '[]'::jsonb
from generate_series(1, 10000) u cross join generate_series(1, 5) n
on conflict do nothing;
analyze public.clothing_items;
analyze public.saved_outfits;
analyze public.profiles;
