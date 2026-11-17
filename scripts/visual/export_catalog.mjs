/** Bounded export of source rows needing offline image enrichment. */
import fs from 'node:fs';
const args=process.argv.slice(2),option=(key,fallback)=>{const i=args.indexOf(key);return i<0?fallback:args[i+1]};
const output=option('--output',null),limit=Number(option('--limit','32')),after=option('--after',''),slot=option('--slot','all');
if(!output||!Number.isInteger(limit)||limit<1||limit>256||after.length>200||!['all','top','outerwear','bottom','shoes'].includes(slot))throw Error('use --output file.json --limit 1..256 [--after product-id] [--slot top|outerwear|bottom|shoes]');
const token=process.env.SUPABASE_ACCESS_TOKEN||fs.readFileSync(process.env.SUPABASE_ACCESS_TOKEN_FILE,'utf8').trim();
const q=s=>"'"+s.replaceAll("'","''")+"'";
const query=`select s.id,s.name,s.brand,s.description,s.category,s.garment_type,s.image_url,i.slot,
 md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text) source_hash
 from public.shop_catalog s join public.catalog_garment_intelligence i on i.product_id=s.id
 left join public.catalog_visual_fashion v on v.product_id=s.id and v.version=1
 where s.is_active and s.id>${q(after)} and s.image_url like 'https://%'
 and i.slot in ('top','outerwear','bottom','shoes') ${slot==='all'?'':`and i.slot=${q(slot)}`}
 and (v.product_id is null or v.source_hash<>md5(jsonb_build_array(s.name,s.description,s.brand,s.category,s.garment_type,s.image_url)::text) or v.profile->>'taxonomyRevision' is distinct from 'stylist5-facets-3' or v.created_at<now()-interval '7 days')
 order by s.id limit ${limit}`;
const r=await fetch('https://api.supabase.com/v1/projects/fyqpifmrsftsfqibhwhy/database/query',{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:JSON.stringify({query}),signal:AbortSignal.timeout(30000)});
if(!r.ok)throw Error('export SQL HTTP '+r.status);const rows=await r.json();fs.writeFileSync(output,JSON.stringify(rows,null,2)+'\n',{mode:0o600});fs.writeFileSync(output+'.cursor.json',JSON.stringify({after:rows.at(-1)?.id||after}),{mode:0o600});
console.log(JSON.stringify({exported:rows.length,hasMore:rows.length===limit,paidCalls:0,originalCatalogUnchanged:true}));
