do $$
declare client_used integer; edge_used integer;
begin
    select used into client_used from public.usage_counters where user_id = md5('load-user-3')::uuid and feature = 'homeSwipes' and day = (now() at time zone 'utc')::date;
    select used into edge_used from public.usage_counters where user_id = md5('load-user-4')::uuid and feature = 'edge:wardrobeScans' and day = (now() at time zone 'utc')::date;
    if client_used is distinct from 3 or edge_used is distinct from 5 then
        raise exception 'concurrent quota failed: client=%, edge=%', client_used, edge_used;
    end if;
end;
$$;
select (select count(*) from auth.users) as users,
       (select count(*) from public.clothing_items) as wardrobe_items,
       (select count(*) from public.saved_outfits) as saved_outfits,
       (select count(*) from public.usage_counters where feature = 'rl:local-capacity') as distinct_users_exercised;
