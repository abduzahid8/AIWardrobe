begin;
set local lock_timeout = '3s';
set local statement_timeout = '15s';

-- One allowance per displayed card and UTC day, even when the reply is lost.
-- Only the authenticated RPC below can create receipts; clients cannot edit them.
create table public.home_swipe_receipts (
    user_id uuid not null references auth.users(id) on delete cascade,
    day date not null,
    card_key text not null check (char_length(card_key) between 1 and 512),
    charged_used integer check (charged_used > 0),
    primary key (user_id, day, card_key)
);
create index home_swipe_receipts_retention_idx on public.home_swipe_receipts(day);
alter table public.home_swipe_receipts enable row level security;
revoke all on public.home_swipe_receipts from public, anon, authenticated;
grant all on public.home_swipe_receipts to service_role;

create function public.authorize_home_swipe(p_card_key text, p_day date)
returns table (allowed boolean, used integer, remaining integer)
language plpgsql security definer set search_path = ''
as $$
declare
    v_user uuid := auth.uid();
    v_day date := (now() at time zone 'utc')::date;
    v_created boolean;
    v_receipt integer;
    v_result record;
    v_used integer;
    v_limit integer;
begin
    if v_user is null then raise exception 'auth required' using errcode = '42501'; end if;
    if p_card_key is null or char_length(p_card_key) not between 1 and 512
       or p_day is null or p_day <> v_day then
        raise exception 'invalid swipe identity or day' using errcode = '22023';
    end if;
    loop
        v_created := null;
        insert into public.home_swipe_receipts(user_id, day, card_key)
        values (v_user, v_day, p_card_key)
        on conflict do nothing returning true into v_created;
        if v_created then
            -- This invocation owns the receipt. Counter + receipt commit together.
            select * into strict v_result from public.check_and_increment_usage('homeSwipes', 0, 1);
            if v_result.allowed then
                update public.home_swipe_receipts set charged_used = v_result.used
                where user_id = v_user and day = v_day and card_key = p_card_key;
            else
                -- A denied card may be retried after a legitimate plan upgrade.
                delete from public.home_swipe_receipts
                where user_id = v_user and day = v_day and card_key = p_card_key;
            end if;
            return query select v_result.allowed::boolean, v_result.used::integer, v_result.remaining::integer;
            return;
        end if;
        select r.charged_used into v_receipt from public.home_swipe_receipts r
        where r.user_id = v_user and r.day = v_day and r.card_key = p_card_key;
        if found then
            -- Never debit a replay, including after the last free allowance.
            select coalesce(c.used, 0) into v_used from public.usage_counters c
            where c.user_id = v_user and c.day = v_day and c.feature = 'homeSwipes';
            v_used := greatest(coalesce(v_used, 0), v_receipt);
            v_limit := case when public.effective_subscription_tier(v_user) = 'free' then 3 else -1 end;
            return query select true, v_used, case when v_limit = -1 then -1 else greatest(0, v_limit - v_used) end;
            return;
        end if;
        -- A competing denied request deleted its temporary receipt: retry ownership.
    end loop;
end;
$$;
revoke execute on function public.authorize_home_swipe(text, date) from public, anon;
grant execute on function public.authorize_home_swipe(text, date) to authenticated;

-- Reuse the existing minute job; each deletion remains bounded and skips locks.
create or replace function public.prune_usage_counters()
returns void language sql security definer set search_path = '' as $$
    with expired as (
        select ctid from public.usage_counters
        where day < (now() at time zone 'utc')::date - 2
        order by day limit 10000 for update skip locked
    )
    delete from public.usage_counters counters using expired where counters.ctid = expired.ctid;
    with expired as (
        select ctid from public.home_swipe_receipts
        where day < (now() at time zone 'utc')::date - 2
        order by day limit 10000 for update skip locked
    )
    delete from public.home_swipe_receipts receipts using expired where receipts.ctid = expired.ctid;
$$;
commit;
