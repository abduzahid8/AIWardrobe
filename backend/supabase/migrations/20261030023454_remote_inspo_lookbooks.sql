-- Remote editable outfit looks and their shoppable product cards.
create table if not exists public.inspo_looks (
    id text primary key,
    brand text not null,
    collection_name text not null,
    title text not null,
    image_url text not null,
    source_url text not null,
    sort_order integer not null default 0,
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.inspo_look_products (
    id text primary key,
    look_id text not null references public.inspo_looks(id) on delete cascade,
    name text not null,
    category text not null default 'TOP',
    image_url text not null,
    price numeric(12,2),
    currency text,
    product_url text not null,
    sort_order integer not null default 0,
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists inspo_looks_active_order_idx
    on public.inspo_looks (is_active, sort_order);
create index if not exists inspo_look_products_parent_order_idx
    on public.inspo_look_products (look_id, is_active, sort_order);

alter table public.inspo_looks enable row level security;
alter table public.inspo_look_products enable row level security;
grant select on public.inspo_looks, public.inspo_look_products to anon, authenticated;
grant insert, update, delete on public.inspo_looks, public.inspo_look_products to authenticated;
grant all on public.inspo_looks, public.inspo_look_products to service_role;

drop policy if exists inspo_looks_public_read on public.inspo_looks;
create policy inspo_looks_public_read on public.inspo_looks
    for select to anon, authenticated using (is_active = true);

drop policy if exists inspo_looks_admin_manage on public.inspo_looks;
create policy inspo_looks_admin_manage on public.inspo_looks
    for all to authenticated using ((select public.is_admin()))
    with check ((select public.is_admin()));

drop policy if exists inspo_look_products_public_read on public.inspo_look_products;
create policy inspo_look_products_public_read on public.inspo_look_products
    for select to anon, authenticated using (
        is_active = true and exists (
            select 1 from public.inspo_looks l
            where l.id = look_id and l.is_active = true
        )
    );

drop policy if exists inspo_look_products_admin_manage on public.inspo_look_products;
create policy inspo_look_products_admin_manage on public.inspo_look_products
    for all to authenticated using ((select public.is_admin()))
    with check ((select public.is_admin()));

create or replace function public.set_inspo_look_updated_at()
returns trigger language plpgsql set search_path = '' as $$
begin
    new.updated_at = pg_catalog.now();
    return new;
end;
$$;

drop trigger if exists inspo_looks_updated_at on public.inspo_looks;
create trigger inspo_looks_updated_at before update on public.inspo_looks
    for each row execute function public.set_inspo_look_updated_at();
drop trigger if exists inspo_look_products_updated_at on public.inspo_look_products;
create trigger inspo_look_products_updated_at before update on public.inspo_look_products
    for each row execute function public.set_inspo_look_updated_at();

insert into public.inspo_looks (id, brand, collection_name, title, image_url, source_url, sort_order)
values
('asket-fw26-01', 'Asket', 'Fall/Winter 2026', 'Look 01', 'https://www.asket.com/images/2160x1543/da8501083a/asket_fw26_looks_mens_15_exteneded_header.jpg?auto=format&q=80&w=1000', 'https://www.asket.com/en-us/journal/inspiration/mens-autumn-winter-looks-2026', 10),
('canali-ss27-01', 'Canali', 'Spring/Summer 2027 preview', 'Look 01', 'https://canali.vtexassets.com/assets/vtex.file-manager-graphql/images/d3f94624-638e-4535-b4b7-dfd5acf8bbc8___55fcdc9efef8053172d5934e355f9aff.jpg', 'https://intl.canali.com/collection/spring-summer/preview', 20),
('canali-ss27-02', 'Canali', 'Spring/Summer 2027 preview', 'Look 02', 'https://canali.vtexassets.com/assets/vtex.file-manager-graphql/images/ee5972ac-2a1d-410b-a3f5-a4321110cbcc___ce130248b80403683c6a2681c8d6d980.jpg', 'https://intl.canali.com/collection/spring-summer/preview', 21),
('canali-ss27-03', 'Canali', 'Spring/Summer 2027 preview', 'Look 03', 'https://canali.vtexassets.com/assets/vtex.file-manager-graphql/images/5d1a8385-d2e1-43c6-a8d4-7526dd4a5b21___0c1b82e15fd91a049a68b5b7fe104504.jpg', 'https://intl.canali.com/collection/spring-summer/preview', 22),
('cos-fw26-01', 'COS', 'Fall/Winter 2026', 'Look 01', 'https://media.cos.com/sb/251958/2479x3718/e234513963/aw26-campaign-shoot_08.jpg?imwidth=1000', 'https://www.cos.com/en-gb/features/collections/men-s-autumn-winter-2026-collection', 30),
('cos-fw26-02', 'COS', 'Fall/Winter 2026', 'Look 02', 'https://media.cos.com/sb/251958/2549x3823/9b47c79103/aw26-campaign-shoot_07.jpg?imwidth=1000', 'https://www.cos.com/en-gb/features/collections/men-s-autumn-winter-2026-collection', 31),
('todd-snyder-fw26-01', 'Todd Snyder', 'Fall/Winter 2026', 'Look 01', 'https://www.toddsnyder.com/cdn/shop/files/Lookbook_Image_3of3_1280x1920_9acc0a90-29dd-45c9-9bd3-8b385b86e134.jpg?v=1771015043&width=600', 'https://www.toddsnyder.com/blogs/editorials/fall-winter-2026-lookbook-american-form', 40),
('todd-snyder-fw26-02', 'Todd Snyder', 'Fall/Winter 2026', 'Look 02', 'https://www.toddsnyder.com/cdn/shop/files/Lookbook_Image_2of3_1280x1920_35c6efb9-83d0-4374-8984-a3a88e643e49.jpg?v=1771015059&width=600', 'https://www.toddsnyder.com/blogs/editorials/fall-winter-2026-lookbook-american-form', 41),
('todd-snyder-fw26-03', 'Todd Snyder', 'Fall/Winter 2026', 'Look 03', 'https://www.toddsnyder.com/cdn/shop/files/Lookbook_Image_1of3_1280x1920_922db451-9502-4aa8-9c32-d1781b30503f.jpg?v=1771015079&width=600', 'https://www.toddsnyder.com/blogs/editorials/fall-winter-2026-lookbook-american-form', 42)
on conflict (id) do nothing;

insert into public.inspo_look_products (id, look_id, name, category, image_url, price, currency, product_url, sort_order)
values
('asket-fw26-01-jacket', 'asket-fw26-01', 'The Field Jacket · Khaki Green', 'LAYER', 'https://images.asket.com/products/images/488_FJT-MC-KIG-thumbnail-original.jpg', 650, 'USD', 'https://www.asket.com/en-us/mens-field-jacket-khaki-green', 10),
('asket-fw26-01-sweater', 'asket-fw26-01', 'The Heavy Wool Sweater · Black', 'TOP', 'https://images.asket.com/products/images/131_HWS-MA-BLA-thumbnail-original.jpg', 260, 'USD', 'https://www.asket.com/en-us/mens-heavy-wool-sweater-black', 20),
('asket-fw26-01-jeans', 'asket-fw26-01', 'The Loose Jeans · Raw Denim', 'BOTTOM', 'https://images.asket.com/products/images/246_RD4-MK-RWD-thumbnail-original.jpg', 230, 'USD', 'https://www.asket.com/en-us/mens-loose-jeans-raw-denim', 30),
('canali-ss27-01-jacket', 'canali-ss27-01', 'Dark brown Canali Travel wool flannel modern-fit double-breasted jacket', 'LAYER', 'https://canalius.vtexassets.com/arquivos/ids/373348/15486-AN04763-902_a_.jpg.jpg?v=639207720575630000', 2150, 'USD', 'https://us.canali.com/dark-brown-canali-travel-wool-flannel-modern-fit-double-breasted-jacket-15486an04763902/p', 10),
('canali-ss27-01-trousers', 'canali-ss27-01', 'Dark brown Canali Travel wool flannel regular-fit pants', 'BOTTOM', 'https://canalius.vtexassets.com/arquivos/ids/373421/71119-AN04763-902_a_.jpg.jpg?v=639207721021500000', 725, 'USD', 'https://us.canali.com/dark-brown-canali-travel-wool-flannel-regular-fit-pants-71119an04763902/p', 20),
('canali-ss27-01-polo', 'canali-ss27-01', 'Ochre structured ribbed merino wool polo shirt', 'TOP', 'https://canali.vtexassets.com/arquivos/ids/539022/C1134-MK02483-502_a_.jpg', 450, 'GBP', 'https://gb.canali.com/ochre-structured-ribbed-merino-wool-polo-shirt-c1134mk02483502/p', 30),
('canali-ss27-01-cap', 'canali-ss27-01', 'White linen and silk blend baseball cap', 'ACCESSORY', 'https://canali.vtexassets.com/arquivos/ids/530108/H155-YA00132-001_a_.jpg', 470, 'GBP', 'https://gb.canali.com/white-linen-and-silk-blend-baseball-cap-h155ya00132001/p', 40),
('canali-ss27-01-loafers', 'canali-ss27-01', 'Black calfskin Penny loafers', 'SHOES', 'https://canali.vtexassets.com/arquivos/ids/531343/611242-RA00527-112_a_.jpg', 450, 'GBP', 'https://gb.canali.com/black-calfskin-penny-loafers-611242ra00527112/p', 50),
('canali-ss27-01-jacket-blue', 'canali-ss27-01', 'Relaxed fit tricot effect navy blue 180s wool jacket', 'LAYER', 'https://canali.vtexassets.com/arquivos/ids/546153/31220F-CJ06154-301_a_.jpg', 2250, 'GBP', 'https://gb.canali.com/relaxed-fit-tricot-effect-navy-blue-180s-wool-jacket-31220fcj06154301/p', 60),
('canali-ss27-02-jacket', 'canali-ss27-02', 'Dark brown Canali Travel wool flannel modern-fit double-breasted jacket', 'LAYER', 'https://canalius.vtexassets.com/arquivos/ids/373348/15486-AN04763-902_a_.jpg.jpg?v=639207720575630000', 2150, 'USD', 'https://us.canali.com/dark-brown-canali-travel-wool-flannel-modern-fit-double-breasted-jacket-15486an04763902/p', 10),
('canali-ss27-02-trousers', 'canali-ss27-02', 'White cotton gabardine double-pleated pants', 'BOTTOM', 'https://canalius.vtexassets.com/arquivos/ids/370388/D91606-PT01666-005_a_.jpg?v=639178242230330000', 650, 'USD', 'https://us.canali.com/white-cotton-gabardine-double-pleated-pants-d91606pt01666005/p', 20),
('canali-ss27-02-cap', 'canali-ss27-02', 'Gray pure wool check baseball cap with piping', 'ACCESSORY', 'https://canalius.vtexassets.com/arquivos/ids/530108/H155-YA00132-001_a_.jpg?v=639142896547300000', 575, 'USD', 'https://us.canali.com/gray-pure-wool-check-baseball-cap-with-piping-h152ya00129200/p', 30),
('canali-ss27-02-shoes', 'canali-ss27-02', 'Brown suede loafers with rubber sole', 'SHOES', 'https://canalius.vtexassets.com/arquivos/ids/351636/461214-RB01269-590_a_.jpg?v=639064947249670000', 775, 'USD', 'https://us.canali.com/brown-suede-loafers-with-rubber-sole-461214rb01269590/p', 40),
('canali-ss27-03-jacket', 'canali-ss27-03', 'Beige wool, silk and linen modern-fit jacket', 'LAYER', 'https://canalius.vtexassets.com/arquivos/ids/359735/23270-AE00386-708_a_.jpg?v=639075485517170000', 2295, 'USD', 'https://us.canali.com/beige-wool-silk-and-linen-modern-fit-jacket-23270ae00386708/p', 10),
('canali-ss27-03-trousers', 'canali-ss27-03', 'Beige cotton gabardine double-pleated pants', 'BOTTOM', 'https://canalius.vtexassets.com/arquivos/ids/370348/D91606-PT01666-727_a_.jpg.jpg?v=639178241864900000', 650, 'USD', 'https://us.canali.com/beige-cotton-gabardine-double-pleated-pants-d91606pt01666727/p', 20),
('canali-ss27-03-polo', 'canali-ss27-03', 'Green polo shirt in soft cotton piqué', 'TOP', 'https://canalius.vtexassets.com/arquivos/ids/373502/T0863P-MJ02239-817_a_.jpg?v=639210061066030000', 450, 'USD', 'https://us.canali.com/green-polo-shirt-in-soft-cotton-pique-t0863pmj02239817/p', 30),
('canali-ss27-03-cap', 'canali-ss27-03', 'White linen and silk blend baseball cap', 'ACCESSORY', 'https://canalius.vtexassets.com/arquivos/ids/368231/H155-YA00132-001_a_.jpg?v=639142815347370000', 695, 'USD', 'https://us.canali.com/white-linen-and-silk-blend-baseball-cap-h155ya00132001/p', 40),
('canali-ss27-03-shoes', 'canali-ss27-03', 'Dark brown deerskin penny loafers with EVA rubber sole', 'SHOES', 'https://canalius.vtexassets.com/arquivos/ids/368956/611244-RF01356-510_a_.jpg.jpg?v=639148611052600000', 825, 'USD', 'https://us.canali.com/dark-brown-deerskin-penny-loafers-with-eva-rubber-sole-611244rf01356510/p', 50),
('cos-fw26-01-jacket', 'cos-fw26-01', 'Black leather blouson jacket', 'LAYER', 'https://media.cos.com/assets/001/cd/80/cd80d347d78f47dbf50717a0da4ec115fa85c4cf_xxl-1.jpg?imwidth=1000', 479, 'USD', 'https://www.cos.com/en-us/men/menswear/coatsjackets/jackets/product/leather-blouson-jacket-black-1301223001', 10),
('cos-fw26-01-trousers', 'cos-fw26-01', 'Double-faced wool wide-leg trousers · Charcoal', 'BOTTOM', 'https://media.cos.com/assets/001/ca/40/ca40a598fdae43bae88b3c79256ad57f50be954d_xxl-1.jpg?imwidth=1000', 139, 'GBP', 'https://www.cos.com/en-gb/men/menswear/trousers/relaxed-fit/wideleg/product/double-faced-wool-relaxed-fit-wide-leg-trousers-charcoal-1342991002', 20),
('cos-fw26-01-polo', 'cos-fw26-01', 'Brushed-wool polo shirt · Dark Brown', 'TOP', 'https://media.cos.com/assets/001/58/52/5852d4d1a2a4d679d0f05813559214cba8643bbb_xxl-1.jpg?imwidth=1000', 85, 'GBP', 'https://www.cos.com/en-gb/men/menswear/knitwear/jumpers/product/brushed-wool-polo-shirt-dark-brown-1302327001', 30),
('cos-fw26-01-loafers', 'cos-fw26-01', 'Classic leather penny loafers · Black', 'SHOES', 'https://media.cos.com/assets/001/9a/56/9a56bf8e8277650ea46d6a9d84c21383af545092_xxl-1.jpg?imwidth=1000', 139, 'GBP', 'https://www.cos.com/en-gb/men/accessories/shoes/formal/product/classic-penny-loafers-black-1288359001', 40),
('cos-fw26-02-jacket', 'cos-fw26-02', 'Black leather blouson jacket', 'LAYER', 'https://media.cos.com/assets/001/cd/80/cd80d347d78f47dbf50717a0da4ec115fa85c4cf_xxl-1.jpg?imwidth=1000', 479, 'USD', 'https://www.cos.com/en-us/men/menswear/coatsjackets/jackets/product/leather-blouson-jacket-black-1301223001', 10),
('cos-fw26-02-trousers', 'cos-fw26-02', 'Double-faced wool wide-leg trousers · Charcoal', 'BOTTOM', 'https://media.cos.com/assets/001/ca/40/ca40a598fdae43bae88b3c79256ad57f50be954d_xxl-1.jpg?imwidth=1000', 139, 'GBP', 'https://www.cos.com/en-gb/men/menswear/trousers/relaxed-fit/wideleg/product/double-faced-wool-relaxed-fit-wide-leg-trousers-charcoal-1342991002', 20),
('cos-fw26-02-polo', 'cos-fw26-02', 'Brushed-wool polo shirt · Dark Brown', 'TOP', 'https://media.cos.com/assets/001/58/52/5852d4d1a2a4d679d0f05813559214cba8643bbb_xxl-1.jpg?imwidth=1000', 85, 'GBP', 'https://www.cos.com/en-gb/men/menswear/knitwear/jumpers/product/brushed-wool-polo-shirt-dark-brown-1302327001', 30),
('cos-fw26-02-loafers', 'cos-fw26-02', 'Classic leather penny loafers · Black', 'SHOES', 'https://media.cos.com/assets/001/9a/56/9a56bf8e8277650ea46d6a9d84c21383af545092_xxl-1.jpg?imwidth=1000', 139, 'GBP', 'https://www.cos.com/en-gb/men/accessories/shoes/formal/product/classic-penny-loafers-black-1288359001', 40),
('todd-snyder-fw26-01-jacket', 'todd-snyder-fw26-01', 'Italian Wool Dad Jacket · Cream Gingham', 'LAYER', 'https://cdn.shopify.com/s/files/1/0186/1574/files/SEPT26_OU0275696_TAN_F1_R.jpg?v=1786631030&width=1000', 698, 'USD', 'https://www.toddsnyder.com/products/gingham-collared-blousoncream', 10),
('todd-snyder-fw26-01-shirt', 'todd-snyder-fw26-01', 'Cotton-Cashmere Sport Shirt · Cream', 'TOP', 'https://cdn.shopify.com/s/files/1/0186/1574/files/SEPT26_SH22810953_CREAM_F1_R.jpg?v=1786630198&width=1000', 198, 'USD', 'https://www.toddsnyder.com/products/cotton-cashmere-sport-shirt-solidcream', 20),
('todd-snyder-fw26-01-polo', 'todd-snyder-fw26-01', 'Holborn Sweater Polo · Chalk Mark', 'TOP', 'https://cdn.shopify.com/s/files/1/0186/1574/files/SEPT26_SW3491019_CHALK_MARK_F1_R.jpg?v=1786630157&width=1000', 328, 'USD', 'https://www.toddsnyder.com/products/holborn-polochalk-mark', 30),
('todd-snyder-fw26-01-derby', 'todd-snyder-fw26-01', 'Italian City Derby Shoe · Brown', 'SHOES', 'https://cdn.shopify.com/s/files/1/0186/1574/files/DRS008_BROWN_1_R.jpg?v=1769714662&width=1000', 398, 'USD', 'https://www.toddsnyder.com/products/todd-snyder-piped-seam-derbybrown', 40),
('todd-snyder-fw26-02-jacket', 'todd-snyder-fw26-02', 'Italian Wool Dad Jacket · Cream Gingham', 'LAYER', 'https://cdn.shopify.com/s/files/1/0186/1574/files/SEPT26_OU0275696_TAN_F1_R.jpg?v=1786631030&width=1000', 698, 'USD', 'https://www.toddsnyder.com/products/gingham-collared-blousoncream', 10),
('todd-snyder-fw26-02-shirt', 'todd-snyder-fw26-02', 'Cotton-Cashmere Sport Shirt · Cream', 'TOP', 'https://cdn.shopify.com/s/files/1/0186/1574/files/SEPT26_SH22810953_CREAM_F1_R.jpg?v=1786630198&width=1000', 198, 'USD', 'https://www.toddsnyder.com/products/cotton-cashmere-sport-shirt-solidcream', 20),
('todd-snyder-fw26-02-polo', 'todd-snyder-fw26-02', 'Holborn Sweater Polo · Chalk Mark', 'TOP', 'https://cdn.shopify.com/s/files/1/0186/1574/files/SEPT26_SW3491019_CHALK_MARK_F1_R.jpg?v=1786630157&width=1000', 328, 'USD', 'https://www.toddsnyder.com/products/holborn-polochalk-mark', 30),
('todd-snyder-fw26-02-derby', 'todd-snyder-fw26-02', 'Italian City Derby Shoe · Brown', 'SHOES', 'https://cdn.shopify.com/s/files/1/0186/1574/files/DRS008_BROWN_1_R.jpg?v=1769714662&width=1000', 398, 'USD', 'https://www.toddsnyder.com/products/todd-snyder-piped-seam-derbybrown', 40),
('todd-snyder-fw26-03-jacket', 'todd-snyder-fw26-03', 'Italian Wool Dad Jacket · Cream Gingham', 'LAYER', 'https://cdn.shopify.com/s/files/1/0186/1574/files/SEPT26_OU0275696_TAN_F1_R.jpg?v=1786631030&width=1000', 698, 'USD', 'https://www.toddsnyder.com/products/gingham-collared-blousoncream', 10),
('todd-snyder-fw26-03-shirt', 'todd-snyder-fw26-03', 'Cotton-Cashmere Sport Shirt · Cream', 'TOP', 'https://cdn.shopify.com/s/files/1/0186/1574/files/SEPT26_SH22810953_CREAM_F1_R.jpg?v=1786630198&width=1000', 198, 'USD', 'https://www.toddsnyder.com/products/cotton-cashmere-sport-shirt-solidcream', 20),
('todd-snyder-fw26-03-polo', 'todd-snyder-fw26-03', 'Holborn Sweater Polo · Chalk Mark', 'TOP', 'https://cdn.shopify.com/s/files/1/0186/1574/files/SEPT26_SW3491019_CHALK_MARK_F1_R.jpg?v=1786630157&width=1000', 328, 'USD', 'https://www.toddsnyder.com/products/holborn-polochalk-mark', 30),
('todd-snyder-fw26-03-derby', 'todd-snyder-fw26-03', 'Italian City Derby Shoe · Brown', 'SHOES', 'https://cdn.shopify.com/s/files/1/0186/1574/files/DRS008_BROWN_1_R.jpg?v=1769714662&width=1000', 398, 'USD', 'https://www.toddsnyder.com/products/todd-snyder-piped-seam-derbybrown', 40)
on conflict (id) do nothing;
