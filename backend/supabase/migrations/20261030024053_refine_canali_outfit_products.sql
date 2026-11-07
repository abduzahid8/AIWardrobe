-- Refine Canali SS27 picks against current standalone Canali product pages.
UPDATE public.inspo_look_products
SET name = 'Dark brown Canali Travel wool flannel modern-fit double-breasted jacket',
    image_url = 'https://canalius.vtexassets.com/arquivos/ids/373348/15486-AN04763-902_a_.jpg.jpg?v=639207720575630000',
    price = 2150, currency = 'USD',
    product_url = 'https://us.canali.com/dark-brown-canali-travel-wool-flannel-modern-fit-double-breasted-jacket-15486an04763902/p'
WHERE id = 'canali-ss27-01-jacket';

UPDATE public.inspo_look_products
SET name = 'Dark brown Canali Travel wool flannel regular-fit pants',
    image_url = 'https://canalius.vtexassets.com/arquivos/ids/373421/71119-AN04763-902_a_.jpg.jpg?v=639207721021500000',
    price = 725, currency = 'USD',
    product_url = 'https://us.canali.com/dark-brown-canali-travel-wool-flannel-regular-fit-pants-71119an04763902/p'
WHERE id = 'canali-ss27-01-trousers';

UPDATE public.inspo_look_products
SET name = 'White linen and silk blend baseball cap',
    image_url = 'https://canalius.vtexassets.com/arquivos/ids/368231/H155-YA00132-001_a_.jpg?v=639142815347370000',
    price = 695, currency = 'USD',
    product_url = 'https://us.canali.com/white-linen-and-silk-blend-baseball-cap-h155ya00132001/p'
WHERE id = 'canali-ss27-01-cap';

-- Second runway look: brown tailoring layered under an outer coat, light trousers,
-- cap and suede loafers. Canali does not expose direct runway-to-SKU mappings.
INSERT INTO public.inspo_look_products (id, look_id, name, category, image_url, price, currency, product_url, sort_order)
VALUES
('canali-ss27-02-jacket', 'canali-ss27-02', 'Dark brown Canali Travel wool flannel modern-fit double-breasted jacket', 'LAYER', 'https://canalius.vtexassets.com/arquivos/ids/373348/15486-AN04763-902_a_.jpg.jpg?v=639207720575630000', 2150, 'USD', 'https://us.canali.com/dark-brown-canali-travel-wool-flannel-modern-fit-double-breasted-jacket-15486an04763902/p', 10),
('canali-ss27-02-trousers', 'canali-ss27-02', 'White cotton gabardine double-pleated pants', 'BOTTOM', 'https://canalius.vtexassets.com/arquivos/ids/370388/D91606-PT01666-005_a_.jpg?v=639178242230330000', 650, 'USD', 'https://us.canali.com/white-cotton-gabardine-double-pleated-pants-d91606pt01666005/p', 20),
('canali-ss27-02-cap', 'canali-ss27-02', 'Gray pure wool check baseball cap with piping', 'ACCESSORY', 'https://canalius.vtexassets.com/arquivos/ids/368231/H155-YA00132-001_a_.jpg?v=639142815347370000', 575, 'USD', 'https://us.canali.com/gray-pure-wool-check-baseball-cap-with-piping-h152ya00129200/p', 30),
('canali-ss27-02-shoes', 'canali-ss27-02', 'Brown suede loafers with rubber sole', 'SHOES', 'https://canalius.vtexassets.com/arquivos/ids/351636/461214-RB01269-590_a_.jpg?v=639064947249670000', 775, 'USD', 'https://us.canali.com/brown-suede-loafers-with-rubber-sole-461214rb01269590/p', 40)
ON CONFLICT (id) DO NOTHING;

-- Third runway look: light tailoring, green polo, white cap and loafers.
INSERT INTO public.inspo_look_products (id, look_id, name, category, image_url, price, currency, product_url, sort_order)
VALUES
('canali-ss27-03-jacket', 'canali-ss27-03', 'Beige wool, silk and linen modern-fit jacket', 'LAYER', 'https://canalius.vtexassets.com/arquivos/ids/359735/23270-AE00386-708_a_.jpg?v=639075485517170000', 2295, 'USD', 'https://us.canali.com/beige-wool-silk-and-linen-modern-fit-jacket-23270ae00386708/p', 10),
('canali-ss27-03-trousers', 'canali-ss27-03', 'Beige cotton gabardine double-pleated pants', 'BOTTOM', 'https://canalius.vtexassets.com/arquivos/ids/370348/D91606-PT01666-727_a_.jpg.jpg?v=639178241864900000', 650, 'USD', 'https://us.canali.com/beige-cotton-gabardine-double-pleated-pants-d91606pt01666727/p', 20),
('canali-ss27-03-polo', 'canali-ss27-03', 'Green polo shirt in soft cotton piqué', 'TOP', 'https://canalius.vtexassets.com/arquivos/ids/373502/T0863P-MJ02239-817_a_.jpg?v=639210061066030000', 450, 'USD', 'https://us.canali.com/green-polo-shirt-in-soft-cotton-pique-t0863pmj02239817/p', 30),
('canali-ss27-03-cap', 'canali-ss27-03', 'White linen and silk blend baseball cap', 'ACCESSORY', 'https://canalius.vtexassets.com/arquivos/ids/368231/H155-YA00132-001_a_.jpg?v=639142815347370000', 695, 'USD', 'https://us.canali.com/white-linen-and-silk-blend-baseball-cap-h155ya00132001/p', 40),
('canali-ss27-03-shoes', 'canali-ss27-03', 'Dark brown deerskin penny loafers with EVA rubber sole', 'SHOES', 'https://canalius.vtexassets.com/arquivos/ids/368956/611244-RF01356-510_a_.jpg.jpg?v=639148611052600000', 825, 'USD', 'https://us.canali.com/dark-brown-deerskin-penny-loafers-with-eva-rubber-sole-611244rf01356510/p', 50)
ON CONFLICT (id) DO NOTHING;
