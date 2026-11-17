/** GET-only real catalog snapshot. Credentials never persisted or printed. */
import {readFileSync,writeFileSync} from 'node:fs';
import dotenv from 'dotenv';
const env=dotenv.parse(readFileSync('.env'));
const key=env.EXPO_PUBLIC_SUPABASE_ANON_KEY || env.SUPABASE_SERVICE_KEY;
const rows=[];
for(let offset=0;offset<30000;offset+=1000){
 const url=new URL('/rest/v1/shop_catalog',env.EXPO_PUBLIC_SUPABASE_URL);
 url.searchParams.set('select','id,name,brand,category,garment_type,description,source,source_url,image_url,price,currency,is_active');
 url.searchParams.set('is_active','eq.true');url.searchParams.set('order','id.asc');
 const r=await fetch(url,{headers:{apikey:key,Authorization:`Bearer ${key}`,Range:`${offset}-${offset+999}`},signal:AbortSignal.timeout(20000)});
 if(!r.ok)throw Error(`catalog_read_${r.status}`);
 const page=await r.json();rows.push(...page);if(page.length<1000)break;
}
writeFileSync('/tmp/aiwardrobe-stylist-catalog.json',JSON.stringify({observedAt:new Date().toISOString(),visibility:'public RLS read',rows}));
console.log(JSON.stringify({activeVisible:rows.length,snapshot:'/tmp/aiwardrobe-stylist-catalog.json'}));
