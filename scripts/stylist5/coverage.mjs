import {query} from './private-api.mjs';
const result=await query(`select jsonb_build_object(
 'activeProducts',(select count(*) from public.shop_catalog where is_active),
 'currentProfiles',count(*),
 'currentEmbeddings',count(*) filter(where e.image_sha256 is not null),
 'shoeProfiles',count(*) filter(where i.slot='shoes'),
 'capsule',(select count(*) from public.catalog_stylist_capsule c join public.shop_catalog p on p.id=c.product_id where p.is_active and c.source_hash=md5(jsonb_build_array(p.name,p.description,p.brand,p.category,p.garment_type,p.image_url)::text)),
 'fieldCounts',(select jsonb_object_agg(field,n) from (select key field,count(*) n from public.catalog_visual_fashion q join public.shop_catalog t on t.id=q.product_id cross join lateral jsonb_each(q.profile->'attributes') a where t.is_active and q.source_hash=md5(jsonb_build_array(t.name,t.description,t.brand,t.category,t.garment_type,t.image_url)::text) group by key) f)
 ) coverage from public.catalog_visual_fashion v join public.shop_catalog s on s.id=v.product_id
 join public.catalog_garment_intelligence i on i.product_id=s.id
 left join public.catalog_image_embeddings e on e.image_sha256=v.image_sha256 and e.model_revision=v.model_revision
 where s.is_active and v.source_hash=md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text)`);
console.log(JSON.stringify(result[0].coverage,null,2));
