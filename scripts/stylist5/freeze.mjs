/** Private fixed benchmark inputs. Never prints or commits product/user records. */
import fs from 'node:fs';import path from 'node:path';import crypto from 'node:crypto';import {query,quote} from './private-api.mjs';
const dir=process.argv[2];if(!dir||fs.existsSync(path.join(dir,'catalog.json')))throw Error('Provide a new private output directory; frozen inputs are immutable');
fs.mkdirSync(dir,{recursive:true,mode:0o700});
const rows=await query(`select s.*,i.slot fashion_slot,i.intelligence fashion_intelligence,
 v.profile visual_fashion,v.source_hash visual_source_hash,
 md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text) source_hash
 from public.shop_catalog s join public.catalog_garment_intelligence i on i.product_id=s.id and i.version=1
 left join public.catalog_visual_fashion v on v.product_id=s.id and v.version=1
  and v.source_hash=md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text) and v.image_url=s.image_url
 where s.is_active and i.slot in ('top','outerwear','bottom','shoes') order by s.id`);
fs.writeFileSync(path.join(dir,'catalog.json'),JSON.stringify(rows),{mode:0o600});
const summary={frozenAt:new Date().toISOString(),rows:rows.length,sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(dir,'catalog.json'))).digest('hex'),slots:Object.fromEntries(['shoes','top','bottom','outerwear'].map(s=>[s,rows.filter(r=>r.fashion_slot===s).length])),photoProfiles:rows.filter(r=>r.visual_fashion).length};
fs.writeFileSync(path.join(dir,'freeze.json'),JSON.stringify(summary,null,2));console.log(summary);
// Freeze the bounded v200 RPC pools with a fixed seed, before any data migration.
const scenarios=[['casual','Travel'],['semi_classic','Work'],['old_money','Work'],['classic','Work'],['semi_classic','Date night'],['minimalist','Everyday']];
for(let n=0;n<scenarios.length;n++){
 const [style,occasion]=scenarios[n],context={dressCode:/Work|Date/.test(occasion)?'smart':'casual'};
 const pool=await query(`select * from public.catalog_stylist_candidates(array['top','outerwear','bottom','shoes'],'stylist5-fixed-benchmark',${quote(JSON.stringify(context))}::jsonb,100,0,null,null)`);
 fs.writeFileSync(path.join(dir,`pool-${n}.json`),JSON.stringify(pool),{mode:0o600});
}
console.log({frozenScenarios:scenarios.length,providerCalls:0});
