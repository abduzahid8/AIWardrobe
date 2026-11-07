create role anon nologin;
create role authenticated nologin;
create role service_role nologin;
create table public.shop_catalog (
 id text primary key,brand text not null default '',name text not null,
 price numeric not null default 0,currency text not null default 'USD',category text not null default 'tops',
 garment_type text not null default 'upper_body',description text not null default '',image_url text not null,
 is_active boolean not null default true,sort_order integer not null default 0,
 source text not null default 'product-only',source_url text not null default '',external_id text
);
alter table public.shop_catalog enable row level security;
create policy fixture_active on public.shop_catalog for select to anon,authenticated using(is_active and source<>'secret-product-only');
grant select on public.shop_catalog to anon,authenticated;
insert into public.shop_catalog(id,brand,name,price,image_url,source_url)
select 'top-'||lpad(n::text,5,'0'),'Top Brand '||(n%3),'Oxford shirt model '||(n%30)||case when n%2=0 then ' Black' else ' White' end,20+n%100,
'https://image.invalid/top-'||n||'.jpg','https://merchant.invalid/products/top-'||(n%30) from generate_series(1,7200)n;
insert into public.shop_catalog(id,brand,name,price,category,garment_type,image_url,source_url)
select 'shoe-'||lpad(n::text,5,'0'),'Shoe Brand '||(n%6),(array['Penny Loafer Black','Leather Sneaker White','Chelsea Boot Brown'])[n%3+1],40+n%100,'shoes','shoes',
'https://image.invalid/shoe-'||n||'.jpg','https://merchant.invalid/products/shoe-'||n from generate_series(1,120)n;
insert into public.shop_catalog(id,brand,name,price,category,garment_type,image_url)
select 'bottom-'||n,'Bottom Brand '||(n%4),'Navy chino trousers '||n,30+n%100,'bottoms','lower_body','https://image.invalid/bottom-'||n||'.jpg' from generate_series(1,400)n;
insert into public.shop_catalog(id,brand,name,price,category,garment_type,image_url)
select 'layer-'||n,'Layer Brand '||(n%3),'Lightweight cotton blazer '||n,50+n%100,'outerwear','upper_body','https://image.invalid/layer-'||n||'.jpg' from generate_series(1,80)n;
insert into public.shop_catalog(id,name,image_url,source,is_active) values
('inactive','Shirt','x','product-only',false),('rls-secret','Shirt','x','secret-product-only',true),
('person','Photo shows a model wearing shirt','x','product-only',true),('unknown-provenance','Shirt','x','generic-feed',true);
