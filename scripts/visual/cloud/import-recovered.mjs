/** Bounded, idempotent recovery of private journals into source-guarded caches. */
import fs from 'node:fs';import crypto from 'node:crypto';import assert from 'node:assert/strict';import {query,quote} from '../../stylist5/private-api.mjs';
const dir=process.env.STYLIST5_JOURNAL_DIR;if(!dir)throw Error('private journal directory required');
const rows=new Map();
for(const name of ['shoes-pixels','garments-pixels','roles-pixels','human-photo-review']){
 const file=dir+'/'+name+'.jsonl';if(!fs.existsSync(file))continue;
 for(const line of fs.readFileSync(file,'utf8').split('\n').filter(Boolean)){
  const row=JSON.parse(line),p=row.profile;
  assert(typeof row.product_id==='string'&&row.product_id.length<=200&&p.version===1&&p.model==='google/siglip2-base-patch16-224'&&p.revision==='75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'&&/^[a-f0-9]{32}$/.test(p.sourceHash)&&/^[a-f0-9]{64}$/.test(p.imageSha256)&&/^https:\/\//.test(p.imageUrl)&&JSON.stringify(p).length<16000,'journal evidence guard failed');
  assert(Array.isArray(row.embedding)&&row.embedding.length===768&&row.embedding.every(v=>Number.isFinite(v)&&Math.abs(v)<=1),'embedding guard failed');
  const norm=row.embedding.reduce((n,x)=>n+x*x,0);assert(norm>=0.9&&norm<=1.1,'embedding normalization guard failed');rows.set(row.product_id,row);
 }
}
assert(rows.size>=338,'expected recovered journals unavailable');
let imported=0,stale=0;
const all=[...rows.values()];
for(let offset=0;offset<all.length;offset+=24){
 const batch=all.slice(offset,offset+24),json=quote(JSON.stringify(batch))+'::jsonb';
 const result=await query(`begin;
 create temporary table recovered on commit drop as select r.* from jsonb_to_recordset(${json}) r(product_id text,profile jsonb,embedding real[])
 join public.shop_catalog s on s.id=r.product_id where s.is_active and r.profile->>'sourceHash'=md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text) and r.profile->>'imageUrl'=s.image_url;
 insert into public.catalog_image_embeddings(image_sha256,model_revision,embedding) select distinct on(profile->>'imageSha256',profile->>'revision') profile->>'imageSha256',profile->>'revision',embedding from recovered on conflict(image_sha256,model_revision) do nothing;
 insert into public.catalog_visual_fashion(product_id,version,source_hash,image_url,image_sha256,model,model_revision,profile)
 select product_id,1,profile->>'sourceHash',profile->>'imageUrl',profile->>'imageSha256',profile->>'model',profile->>'revision',profile from recovered
 on conflict(product_id,version) do update set source_hash=excluded.source_hash,image_url=excluded.image_url,image_sha256=excluded.image_sha256,model=excluded.model,model_revision=excluded.model_revision,profile=excluded.profile,created_at=now();
 select count(*)::integer imported from recovered;commit;`);
 const accepted=Number(result.find(r=>r.imported!==undefined)?.imported??0);imported+=accepted;stale+=batch.length-accepted;
 console.log(JSON.stringify({journalRowsHandled:Math.min(offset+24,all.length),imported,stale}));
}
console.log(JSON.stringify({recovered:rows.size,imported,stale,journalDigest:crypto.createHash('sha256').update(JSON.stringify(all.map(r=>({id:r.product_id,hash:r.profile.sourceHash,image:r.profile.imageSha256})))).digest('hex')}));
