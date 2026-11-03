-- Local preparation only. No ingestion, scheduler, notifications or production activation.
begin;
create table public.purchase_fit_evidence (
 id text primary key check(length(id)<=128), user_id uuid not null references auth.users(id) on delete cascade,
 product_id text, product_family_key text, variant_id text, brand text, category text not null,
 subcategory text,size_label text,cut text,materials text[],stretch text,
 source text not null check(source in ('owned','purchase','try_on','return','manual','body_profile','reference_garment')),
 preferred_fit text check(preferred_fit in ('snug','regular','relaxed')),
 overall_fit text not null check(overall_fit in ('too_small','snug','perfect','relaxed','too_large','unknown')),
 zones jsonb check(zones is null or (jsonb_typeof(zones)='object' and octet_length(zones::text)<=4096)),
 kept boolean,returned boolean,sold boolean,return_reason text check(length(return_reason)<=240),
 confidence numeric not null check(confidence>=0 and confidence<=1),is_reference boolean not null default false,
 created_at timestamptz not null default now(),source_version text
);
create index purchase_fit_evidence_owner_recent on public.purchase_fit_evidence(user_id,created_at desc);
create table public.purchase_fit_profiles (
 user_id uuid primary key references auth.users(id) on delete cascade,
 version integer not null default 1 check(version>0),aggregate jsonb not null check(jsonb_typeof(aggregate)='object' and octet_length(aggregate::text)<=32768),updated_at timestamptz not null default now()
);
create table public.purchase_price_watches (
 id text primary key,user_id uuid not null references auth.users(id) on delete cascade,
 product_id text not null,variant_id text,selected_size text,merchant text,currency text not null check(currency~'^[A-Z]{3}$'),
 threshold numeric not null check(threshold>=0 and threshold<'Infinity'::numeric),status text not null default 'active' check(status in ('active','paused','triggered')),
 created_at timestamptz not null default now(),last_checked_at timestamptz,last_notified_offer_id text
);
create index purchase_price_watches_owner on public.purchase_price_watches(user_id,status);
alter table public.purchase_fit_evidence enable row level security;
alter table public.purchase_fit_profiles enable row level security;
alter table public.purchase_price_watches enable row level security;
create policy purchase_fit_evidence_own on public.purchase_fit_evidence for all to authenticated using(user_id=(select auth.uid())) with check(user_id=(select auth.uid()));
create policy purchase_fit_profiles_own on public.purchase_fit_profiles for all to authenticated using(user_id=(select auth.uid())) with check(user_id=(select auth.uid()));
create policy purchase_price_watches_own on public.purchase_price_watches for all to authenticated using(user_id=(select auth.uid())) with check(user_id=(select auth.uid()));
revoke all on public.purchase_fit_evidence,public.purchase_fit_profiles,public.purchase_price_watches from public,anon,authenticated;
grant select,insert,update,delete on public.purchase_fit_evidence,public.purchase_fit_profiles,public.purchase_price_watches to authenticated;
grant all on public.purchase_fit_evidence,public.purchase_fit_profiles,public.purchase_price_watches to service_role;

create table public.shop_merchants (
 id text primary key,name text not null,region text,is_active boolean not null default true,
 confidence text not null default 'low' check(confidence in ('low','medium','high')),
 return_policy jsonb check(return_policy is null or octet_length(return_policy::text)<=4096)
);
create table public.shop_offers (
 id text primary key,variant_id text not null,catalog_listing_id text references public.shop_catalog(id) on delete set null check(catalog_listing_id is null or catalog_listing_id=variant_id),merchant_id text not null references public.shop_merchants(id),
 merchant_product_id text,url text not null check(url~'^https?://'),region text,currency text not null check(currency~'^[A-Z]{3}$'),
 current_price numeric not null check(current_price>0 and current_price<'Infinity'::numeric),original_price numeric check(original_price>0 and original_price<'Infinity'::numeric),
 shipping_price numeric check(shipping_price>=0 and shipping_price<'Infinity'::numeric),tax_price numeric check(tax_price>=0 and tax_price<'Infinity'::numeric),fees numeric check(fees>=0 and fees<'Infinity'::numeric),
 discount numeric check(discount>=0 and discount<'Infinity'::numeric),coupon numeric check(coupon>=0 and coupon<'Infinity'::numeric),
 component_provenance jsonb check(component_provenance is null or octet_length(component_provenance::text)<=4096),
 canonical_identifier jsonb check(canonical_identifier is null or octet_length(canonical_identifier::text)<=512),
 in_stock boolean,size_stock jsonb check(size_stock is null or (jsonb_typeof(size_stock)='object' and octet_length(size_stock::text)<=4096)),
 is_active boolean not null default true,last_seen_at timestamptz not null default now()
);
create unique index shop_offers_merchant_ref on public.shop_offers(merchant_id,merchant_product_id,coalesce(region,'')) where merchant_product_id is not null;
create index shop_offers_variant on public.shop_offers(variant_id,is_active);
create table public.shop_price_history (
 offer_id text not null references public.shop_offers(id) on delete cascade,observed_at timestamptz not null default clock_timestamp(),
 price numeric not null check(price>0 and price<'Infinity'::numeric),currency text not null check(currency~'^[A-Z]{3}$'),original_price numeric,in_stock boolean,
 primary key(offer_id,observed_at)
);
-- The (offer_id, observed_at) primary key also serves reverse-chronological reads.
alter table public.shop_merchants enable row level security;
alter table public.shop_offers enable row level security;
alter table public.shop_price_history enable row level security;
create policy shop_merchants_read on public.shop_merchants for select to anon,authenticated using(is_active);
create policy shop_offers_read on public.shop_offers for select to anon,authenticated using(is_active and exists(select 1 from public.shop_merchants m where m.id=merchant_id and m.is_active) and exists(select 1 from public.shop_catalog c where c.id=variant_id and c.is_active));
create policy shop_price_history_read on public.shop_price_history for select to anon,authenticated using(exists(select 1 from public.shop_offers o where o.id=offer_id and o.is_active));
revoke all on public.shop_merchants,public.shop_offers,public.shop_price_history from public,anon,authenticated;
grant select on public.shop_merchants,public.shop_offers,public.shop_price_history to anon,authenticated;
grant all on public.shop_merchants,public.shop_offers,public.shop_price_history to service_role;
create function public.record_shop_offer_price() returns trigger language plpgsql security invoker set search_path='' as $$
begin
 if tg_op='UPDATE' and (new.variant_id is distinct from old.variant_id or new.merchant_id is distinct from old.merchant_id) then raise exception using errcode='22023',message='offer_identity_immutable'; end if;
 if tg_op='INSERT' or new.current_price is distinct from old.current_price or new.original_price is distinct from old.original_price or new.currency is distinct from old.currency or new.in_stock is distinct from old.in_stock or (new.last_seen_at>old.last_seen_at and not exists(select 1 from public.shop_price_history h where h.offer_id=new.id and h.observed_at>=(date_trunc('day',clock_timestamp() at time zone 'UTC') at time zone 'UTC'))) then
  insert into public.shop_price_history(offer_id,observed_at,price,currency,original_price,in_stock) values(new.id,clock_timestamp(),new.current_price,new.currency,new.original_price,new.in_stock);
 end if;
 return new;
end; $$;
revoke all on function public.record_shop_offer_price() from public,anon,authenticated;
grant execute on function public.record_shop_offer_price() to service_role;
create trigger record_shop_offer_price after insert or update on public.shop_offers for each row execute function public.record_shop_offer_price();
comment on table public.purchase_fit_evidence is 'Private fit feedback. No body measurement payload; owner-scoped. Remote sync requires separate activation.';
comment on table public.shop_price_history is 'Observed native price changes and at most one unchanged-price checkpoint per actually observed UTC day; no forecasts or synthetic historical backfill.';
commit;
