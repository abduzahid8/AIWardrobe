/** Real anonymous catalog reads through the staged production helper; no writes or AI. */
import 'dotenv/config';
import {readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import ts from 'typescript';
import {createClient} from '@supabase/supabase-js';
const root=path.resolve(process.argv[2]||'');if(!process.argv[2])throw Error('Staged function directory required');
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,key=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const helpers={},hashes={};
for(const name of ['occasionRules','outfitStructure']){
 const source=await readFile(path.join(root,'_shared',name+'.ts'),'utf8');
 new Function('exports',ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText)(helpers);
}
const entry=await readFile(path.join(root,'generate-outfits/index.ts'),'utf8');hashes.entry=createHash('sha256').update(entry).digest('hex');
const strip=s=>s.replace(/^import .*\n/gm,'').replace(/\bexport\s+/g,'');
const code=ts.transpileModule(strip(await readFile(path.join(root,'_shared/modelOutfitResponse.ts'),'utf8'))+'\n'+strip(entry),{compilerOptions:{module:ts.ModuleKind.None,target:ts.ScriptTarget.ES2022}}).outputText;
const api=new Function('serve','createClient','Deno',...Object.keys(helpers),code+'\nreturn {fetchShopItemsForAllSlotsEdge,macroCategory,getStyleContext,extractRejectKeywords};')(()=>{},createClient,{env:{get:()=>undefined}},...Object.values(helpers));
let phase='choose-target';const requests=[];
const client=createClient(base,key,{auth:{persistSession:false,autoRefreshToken:false},global:{fetch:async(input,init)=>{
 const url=new URL(typeof input==='string'?input:input instanceof URL?input.href:input.url),start=performance.now();
 const response=await fetch(input,{...init,signal:init?.signal??AbortSignal.timeout(15000)});
 requests.push({phase,status:response.status,limit:url.searchParams.get('limit'),order:url.searchParams.get('order'),cursor:url.searchParams.get('id'),latencyMs:Math.round(performance.now()-start)});return response;
}}});
const report={measuredAt:new Date().toISOString(),scope:'Actual public catalog through staged helper; an artificial target-ID predicate forces discovery past100rows. No catalog mutation, user account or NVIDIA request.',sourceHashes:hashes};
const originalLog=console.log;console.log=(...args)=>{if(!String(args[0]).startsWith('[Deno macroCategory]'))originalLog(...args);};
try{
 const selected=await client.from('shop_catalog').select('id,name,category,garment_type,description,brand,image_url').eq('is_active',true).or('category.eq.bottoms,garment_type.eq.lower_body').order('id',{ascending:true}).range(100,199);
 if(selected.error)throw Error('Catalog target discovery failed');
 const banned=api.extractRejectKeywords(api.getStyleContext('casual').reject);
 const target=selected.data.find(row=>row.image_url&&row.name&&!/^\d+$/.test(row.name)&&api.macroCategory(row.category,row.garment_type,row.name)==='bottom'&&!helpers.rejectForStyleOccasion({...row,macroCategory:'bottom'},'casual','Everyday')&&!banned.some(word=>(`${row.name} ${row.description||''} ${row.garment_type} ${row.category}`).toLowerCase().includes(word)));
 if(!target)throw Error('Eligible target beyond100rows unavailable');
 phase='actual-helper';const start=performance.now();
 const result=await api.fetchShopItemsForAllSlotsEdge(client,['bottom'],'casual',1,undefined,row=>row.id===target.id);
 const calls=requests.filter(r=>r.phase==='actual-helper');
 report.checks={targetReturned:result.length===1&&result[0].id==='shop_'+target.id,multiplePages:calls.length>=2,successfulRequests:calls.every(r=>r.status===200),stableOrder:calls.every(r=>r.order==='id.asc'),cursorAdvances:calls.slice(1).every(r=>r.cursor?.startsWith('gt.'))};
 report.helperLatencyMs=Math.round(performance.now()-start);report.targetId=target.id;report.passed=Object.values(report.checks).every(Boolean);
}catch(error){report.failure=error.name;report.passed=false;}
finally{console.log=originalLog;report.requests=requests;await writeFile('docs/reports/readiness-evidence/catalog-keyset-public-read-v110.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));if(!report.passed)process.exitCode=1;}
