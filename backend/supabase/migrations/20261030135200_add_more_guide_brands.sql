-- Add official Fall/Winter 2026 menswear previews from new brands.
insert into public.inspo_looks
    (id, brand, collection_name, title, image_url, source_url, sort_order, is_active)
values
    ('paul-shark-fw26-01', 'Paul & Shark', 'Fall/Winter 2026', 'Look 01',
     'https://www.paulandshark.com/on/demandware.static/-/Sites-catalog_paulandshark_front/default/dw7cbbd085/fw26_menu_m.jpg',
     'https://www.paulandshark.com/en_US/men/fall-winter-collection/', 50, true),
    ('paul-shark-fw26-02', 'Paul & Shark', 'Fall/Winter 2026', 'Look 02',
     'https://www.paulandshark.com/on/demandware.static/-/Sites-catalog_paulandshark_front/default/dwb94a8201/fw26_menu2.jpg',
     'https://www.paulandshark.com/en_US/men/fall-winter-collection/', 51, true),
    ('etro-fw26-27-01', 'ETRO', 'Fall/Winter 2026/2027', 'Look 01',
     'https://content.etro.com/Adaptations/1024/ETRO_FW2627_PRECOLL-12_4_5.jpg',
     'https://www.etro.com/us-en/men/', 60, true)
on conflict (id) do update set
    brand = excluded.brand,
    collection_name = excluded.collection_name,
    title = excluded.title,
    image_url = excluded.image_url,
    source_url = excluded.source_url,
    sort_order = excluded.sort_order,
    is_active = excluded.is_active;

insert into public.inspo_look_products
    (id, look_id, name, category, image_url, price, currency, product_url, sort_order, is_active)
values
    ('paul-shark-fw26-01-jacket', 'paul-shark-fw26-01', 'Wool Field Jacket with Suede Details · Navy', 'LAYER', 'https://media-cdn.paulandshark.com/dam/ecomm/M/16312292-050-f.jpg', 1795, 'USD', 'https://www.paulandshark.com/en_US/wool-field-jacket-with-suede-details/16312292-050.html', 10, true),
    ('paul-shark-fw26-01-knit', 'paul-shark-fw26-01', 'Soft Cool Touch Wool Turtleneck · Navy', 'TOP', 'https://media-cdn.paulandshark.com/dam/ecomm/M/15311064-050-f.jpg', 495, 'USD', 'https://www.paulandshark.com/en_US/soft-cool-touch-wool-turtleneck/15311064-050.html', 20, true),
    ('paul-shark-fw26-01-trousers', 'paul-shark-fw26-01', 'Bellagio Trousers in Wool Twill · Blue', 'BOTTOM', 'https://media-cdn.paulandshark.com/dam/ecomm/M/16314069-013-f.jpg', 545, 'USD', 'https://www.paulandshark.com/en_US/bellagio-trousers-in-wool-twill/16314069-013.html', 30, true),
    ('paul-shark-fw26-01-shoes', 'paul-shark-fw26-01', 'Club Riviera Leather Sneakers · Blue', 'SHOES', 'https://media-cdn.paulandshark.com/dam/ecomm/M/15318007-050-f3.jpg', 465, 'USD', 'https://www.paulandshark.com/en_US/club-riviera-sneakers/15318007-050.html', 40, true),
    ('paul-shark-fw26-02-coat', 'paul-shark-fw26-02', 'Maestro Cashmere Car Coat · Navy', 'LAYER', 'https://media-cdn.paulandshark.com/dam/ecomm/M/16312315-050-f.jpg', 2745, 'USD', 'https://www.paulandshark.com/en_US/maestro-cashmere-car-coat/16312315-050.html', 10, true),
    ('paul-shark-fw26-02-knit', 'paul-shark-fw26-02', 'Half-Zip Sweater in Wool and Cashmere', 'TOP', 'https://media-cdn.paulandshark.com/dam/ecomm/M/16311138-050-f.jpg', 695, 'USD', 'https://www.paulandshark.com/en_US/half-zip-sweater-in-wool-and-cashmere-with-suede-details/16311138-050.html', 20, true),
    ('paul-shark-fw26-02-trousers', 'paul-shark-fw26-02', 'Lucca Wool Trousers · Navy', 'BOTTOM', 'https://media-cdn.paulandshark.com/dam/ecomm/M/16314079-013-f.jpg', 775, 'USD', 'https://www.paulandshark.com/en_US/lucca-wool-trousers/16314079-013.html', 30, true),
    ('paul-shark-fw26-02-shoes', 'paul-shark-fw26-02', 'Club Riviera Leather Sneakers · Blue', 'SHOES', 'https://media-cdn.paulandshark.com/dam/ecomm/M/15318007-050-f3.jpg', 465, 'USD', 'https://www.paulandshark.com/en_US/club-riviera-sneakers/15318007-050.html', 40, true),
    ('etro-fw26-27-01-jacket', 'etro-fw26-27-01', 'Single-Breasted Jacket in Stretch Cotton Velvet', 'LAYER', 'https://content.etro.com/Adaptations/768/MRCA008199TUEI1V0707_SF_01.jpg', 1990, 'USD', 'https://www.etro.com/us-en/single-breasted-jacket-in-stretch-cotton-velvet-MRCA008199TUEI1V0707.html', 10, true),
    ('etro-fw26-27-01-trousers', 'etro-fw26-27-01', 'Stretch Velvet Trousers', 'BOTTOM', 'https://content.etro.com/Adaptations/768/MREA000599TUEI1V0707_SF_01.jpg', 775, 'USD', 'https://www.etro.com/us-en/stretch-velvet-trousers-MREA000599TUEI1V0707.html', 20, true),
    ('etro-fw26-27-01-shirt', 'etro-fw26-27-01', 'Silk Twill Shirt with Placed Paisley Pochette Print', 'TOP', 'https://content.etro.com/Adaptations/768/MRIC007199SP112X0890_SF_01.jpg', 1700, 'USD', 'https://www.etro.com/us-en/silk-twill-shirt-with-placed-paisley-pochette-print-MRIC007199SP112X0890.html', 30, true),
    ('etro-fw26-27-01-sweater', 'etro-fw26-27-01', 'Polo-Style Sweater in Cashmere with Wavy Motif', 'TOP', 'https://content.etro.com/Adaptations/768/MRKF0028AU12PV0456_SF_01.jpg', 1600, 'USD', 'https://www.etro.com/us-en/polo-style-sweater-in-cashmere-with-wavy-motif-MRKF0028AU12PV0456.html', 40, true)
on conflict (id) do update set
    look_id = excluded.look_id,
    name = excluded.name,
    category = excluded.category,
    image_url = excluded.image_url,
    price = excluded.price,
    currency = excluded.currency,
    product_url = excluded.product_url,
    sort_order = excluded.sort_order,
    is_active = excluded.is_active;
