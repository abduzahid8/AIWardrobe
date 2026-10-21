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
const api=new Function('serve','createClient','Deno',...Object.keys(helpers),code+'\nreturn {fetchShopItemsForAllSlotsEdge,filterShopByPrefs};')(()=>{},createClient,{env:{get:()=>undefined}},...Object.values(helpers));
let phase='preferences';const requests=[];
const client=createClient(base,key,{auth:{persistSession:false,autoRefreshToken:false},global:{fetch:async(input,init)=>{
 const url=new URL(typeof input==='string'?input:input instanceof URL?input.href:input.url),start=performance.now();
 const response=await fetch(input,{...init,signal:init?.signal??AbortSignal.timeout(15000)});
 requests.push({phase,status:response.status,limit:url.searchParams.get('limit'),order:url.searchParams.get('order'),cursor:url.searchParams.get('id'),latencyMs:Math.round(performance.now()-start)});return response;
}}});
const report={measuredAt:new Date().toISOString(),scope:'Actual public catalog and staged helper with preferences; Node execution, no writes or provider calls.',sourceHashes:hashes};
const originalLog=console.log;console.log=(...args)=>{if(!String(args[0]).startsWith('[Deno macroCategory]'))originalLog(...args);};
try{
 report.scope='Actual public catalog through the staged production helper with color and pattern preferences included in eligibility before page completion and slot trimming. Runs in Node, not hosted Edge. No writes, accounts or AI requests.';
 report.scenarios=[];
 for(const preferences of [{avoidColors:['black','navy','blue']},{dislikedPatterns:['striped','check','print']}]){
  phase=Object.keys(preferences)[0];const start=performance.now();
  const eligible=row=>!helpers.rejectForStyleOccasion(row,'casual','Everyday')&&!helpers.rejectForClimate(row,{temp:20})&&!helpers.rejectForActivity(row,'Everyday')&&api.filterShopByPrefs([row],preferences).length===1;
  const items=await api.fetchShopItemsForAllSlotsEdge(client,['bottom'],'casual',6,undefined,eligible);
  report.scenarios.push({preferences,latencyMs:Math.round(performance.now()-start),count:items.length,allPass:items.length===6&&items.every(eligible),ids:items.map(item=>item.id)});
 }
 report.passed=report.scenarios.length===2&&report.scenarios.every(row=>row.allPass)&&requests.every(row=>row.status===200);
}catch(error){report.failure=error.name;report.passed=false;}
finally{console.log=originalLog;report.requests=requests;await writeFile('docs/reports/readiness-evidence/catalog-preferences-public-read-v112.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));if(!report.passed)process.exitCode=1;}
