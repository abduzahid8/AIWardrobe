import json,re,html
from pathlib import Path
allp=json.load(open('work/stylist-fix/thursday.json'))['products']+json.load(open('work/stylist-fix/thursday-dress.json'))['products']
handles=['mens-dress-shoe-chairman-black-leather','mens-dress-shoe-executive-black-leather','mens-lincoln-penny-loafer-black-leather','mens-lincoln-penny-loafer-dark-brown','mens-saint-double-monk-dress-shoe-cognac','mens-statesman-derby-shoe-brandy-leather','mens-renegade-derby-shoe-black-leather','mens-duke-chelsea-boot-black-coffee','mens-journeyman-wholecut-chelsea-boot-black','mens-president-lace-up-boot-bull-brown-stormking','mens-era-leather-sneaker-cloud','mens-era-leather-sneaker-navy']
products={p['handle']:p for p in allp};rows=[]
for handle in handles:
 p=products[handle];available=[v for v in p['variants'] if v['available']]
 if not available:continue
 # A row represents a product at one unambiguous price, never an unavailable teaser price.
 prices={float(v['price']) for v in available}
 if len(prices)!=1:continue
 features=re.split(r'<div class="product-shipping-info-wrapper">',p['body_html'])[0]
 body=html.unescape(re.sub('<[^>]+>',' ',features));body=re.sub(r'\s+',' ',body).strip()
 noun=handle.replace('mens-','').replace('-',' ')
 evidence=[t.replace(':',' ') for t in p.get('tags',[]) if re.match(r'^(color|material|pf-style|style|sole):',t) or re.match(r'^(oxford|derby) search_term$',t)]
 noun += ' ' + ' '.join(evidence)
 rows.append(dict(id='thursday-'+str(p['id']),brand='Thursday Boot Co.',name=p['title'],price=prices.pop(),currency='USD',category='shoes',garment_type='shoes',description=f'{noun}. {body}'[:1900],image_url=p['images'][0]['src'],is_active=True,sort_order=9000+len(rows),source='thursday-official-products',source_url='https://thursdayboots.com/products/'+handle,external_id=str(p['id'])))
Path('data/shop_catalog_thursday.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
# Do not activate stale listings. Repair only products already active.
sql="""update public.shop_catalog set garment_type='shoes',category='shoes'
where is_active and name ~* '\\m(shoes?|sneakers?|trainers?|loafers?|derbys?|derbies|brogues?|moccasins?|espadrilles?|boots?)\\M'
and name !~* '\\m(shirts?|jeans|pants?|trousers?|bags?)\\M'
and (garment_type <> 'shoes' or category <> 'shoes');
"""
json_sql=json.dumps(rows,ensure_ascii=False).replace("'","''")
sql+="""insert into public.shop_catalog (id,brand,name,price,currency,category,garment_type,description,image_url,is_active,sort_order,source,source_url,external_id,last_seen_at)
select id,brand,name,price,currency,category,garment_type,description,image_url,is_active,sort_order,source,source_url,external_id,now()
from jsonb_populate_recordset(null::public.shop_catalog, '"""+json_sql+"""'::jsonb)
on conflict(id) do update set price=excluded.price,currency=excluded.currency,description=excluded.description,image_url=excluded.image_url,is_active=excluded.is_active,last_seen_at=excluded.last_seen_at;
select garment_type,category,count(*) from public.shop_catalog where is_active group by 1,2 order by 3 desc;
"""
Path('work/stylist-fix/catalog-repair.sql').write_text(sql)
print('Verified live footwear rows prepared:',len(rows))
