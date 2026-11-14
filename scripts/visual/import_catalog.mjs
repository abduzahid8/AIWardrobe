/** Guarded service-only import. Price/product rows are never updated. */
import fs from 'node:fs';
const input=process.argv[2];if(!input)throw Error('provide enrichment JSONL');
const rows=fs.readFileSync(input,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
if(!rows.length||rows.length>256)throw Error('batch bound 1..256');
const token=process.env.SUPABASE_ACCESS_TOKEN || fs.readFileSync(process.env.SUPABASE_ACCESS_TOKEN_FILE,'utf8').trim();
const quote=x=>"'"+JSON.stringify(x).replaceAll("'","''")+"'::jsonb";
for(const {product_id,profile:p,embedding} of rows){
 if(typeof product_id!=='string'||p?.version!==1||!(/^[a-f0-9]{40}$/.test(p.revision))||!(/^[a-f0-9]{64}$/.test(p.imageSha256))||!(/^[a-f0-9]{32}$/.test(p.sourceHash))||!Array.isArray(embedding)||embedding.length!==768||embedding.some(n=>!Number.isFinite(n))||JSON.stringify(p).length>16000)throw Error('invalid image evidence');
}
const query=`begin;
create temporary table imported as select * from jsonb_to_recordset(${quote(rows)}) r(product_id text,profile jsonb,embedding real[]);
insert into public.catalog_image_embeddings(image_sha256,model_revision,embedding)
select distinct on (profile->>'imageSha256',profile->>'revision') profile->>'imageSha256',profile->>'revision',embedding from imported
on conflict(image_sha256,model_revision) do nothing;
insert into public.catalog_visual_fashion(product_id,version,source_hash,image_url,image_sha256,model,model_revision,profile)
select r.product_id,1,r.profile->>'sourceHash',r.profile->>'imageUrl',r.profile->>'imageSha256',r.profile->>'model',r.profile->>'revision',r.profile
from imported r join public.shop_catalog s on s.id=r.product_id
where s.is_active and r.profile->>'sourceHash'=md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text)
  and r.profile->>'imageUrl'=s.image_url
on conflict(product_id,version) do update set source_hash=excluded.source_hash,image_url=excluded.image_url,image_sha256=excluded.image_sha256,
 model=excluded.model,model_revision=excluded.model_revision,profile=excluded.profile,created_at=now();
commit;
select count(*)::int as current_visual_products from public.catalog_visual_fashion;`;
const r=await fetch('https://api.supabase.com/v1/projects/fyqpifmrsftsfqibhwhy/database/query',{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:JSON.stringify({query}),signal:AbortSignal.timeout(60000)});
if(!r.ok)throw Error('import SQL HTTP '+r.status);console.log(JSON.stringify({requested:rows.length,result:await r.json(),paidCalls:0,authoritativeCatalogUnchanged:true}));
