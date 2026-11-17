insert into public.shop_merchants(id,name) values('merchant','Test Merchant');
insert into public.shop_merchants(id,name,is_active) values('hidden','Hidden',false);
insert into public.shop_offers(id,variant_id,merchant_id,merchant_product_id,url,currency,current_price) values('offer','shoe-00001','merchant','sku','https://merchant.invalid/products/one','USD',100),('hidden-offer','shoe-00001','hidden','sku','https://hidden.invalid/product','USD',50);
do $$ begin
 if (select count(*) from public.shop_price_history where offer_id='offer')<>1 then raise exception 'initial history missing';end if;
end $$;
update public.shop_offers set current_price=100,last_seen_at=now() where id='offer';
do $$ begin if (select count(*) from public.shop_price_history where offer_id='offer')<>1 then raise exception 'redundant same-price history';end if;end $$;
update public.shop_offers set current_price=80 where id='offer';
do $$ begin
 if (select count(*) from public.shop_price_history where offer_id='offer')<>2 then raise exception 'price change missing';end if;
 begin update public.shop_offers set variant_id='shoe-00002' where id='offer';raise exception 'mutable offer identity';exception when sqlstate '22023' then null;end;
end $$;
set role authenticated;
select set_config('request.jwt.claim.sub','11111111-1111-1111-1111-111111111111',false);
insert into public.purchase_fit_evidence(id,user_id,product_id,category,source,overall_fit,confidence) values('private-fit','11111111-1111-1111-1111-111111111111','garment','top','owned','perfect',0.9);
insert into public.purchase_fit_profiles(user_id,aggregate) values('11111111-1111-1111-1111-111111111111','{"version":1}');
insert into public.purchase_price_watches(id,user_id,product_id,currency,threshold) values('private-watch','11111111-1111-1111-1111-111111111111','shoe','MYR',350);
do $$ begin
 if(select count(*) from public.purchase_fit_evidence)<>1 then raise exception 'owner cannot read evidence';end if;
 if(select count(*) from public.shop_offers)<>1 then raise exception 'hidden merchant leak';end if;
 begin insert into public.shop_offers(id,variant_id,merchant_id,url,currency,current_price) values('unauthorized','shoe-00001','merchant','https://merchant.invalid/products/one','USD',1);raise exception 'public writes allowed';exception when insufficient_privilege then null;end;
end $$;
select set_config('request.jwt.claim.sub','22222222-2222-2222-2222-222222222222',false);
do $$ begin
 if(select count(*) from public.purchase_fit_evidence)<>0 or (select count(*) from public.purchase_fit_profiles)<>0 or (select count(*) from public.purchase_price_watches)<>0 then raise exception 'private account data leak';end if;
 begin insert into public.purchase_fit_evidence(id,user_id,category,source,overall_fit,confidence) values('forged','11111111-1111-1111-1111-111111111111','top','owned','perfect',0.9);raise exception 'owner spoof allowed';exception when insufficient_privilege then null;end;
end $$;
reset role;
set role anon;
do $$ begin
 begin perform count(*) from public.purchase_fit_evidence;raise exception 'anonymous private read allowed';exception when insufficient_privilege then null;end;
 if(select count(*) from public.shop_offers)<>1 then raise exception 'public active offer read failed';end if;
 if(select count(*) from public.shop_price_history)<>2 then raise exception 'hidden historical offers exposed';end if;
end $$;
reset role;

-- Existing catalog hard-deletes remain possible; archival price data is retained but hidden.
insert into public.shop_catalog(id,name,image_url) values('disposable-quote','Test shirt','https://image.invalid/disposable.jpg');
insert into public.shop_offers(id,variant_id,catalog_listing_id,merchant_id,url,currency,current_price) values('archival','disposable-quote','disposable-quote','merchant','https://merchant.invalid/disposable','USD',50);
delete from public.shop_catalog where id='disposable-quote';
do $$ begin
 if not exists(select 1 from public.shop_offers where id='archival' and variant_id='disposable-quote' and catalog_listing_id is null) then raise exception 'catalog delete blocked or price archive destroyed';end if;
 if(select count(*) from public.shop_price_history where offer_id='archival')<>1 then raise exception 'archival price history lost';end if;
end $$;
set role anon;
do $$ begin if exists(select 1 from public.shop_offers where id='archival')then raise exception 'orphan quote published';end if;end $$;
reset role;
