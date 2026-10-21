/** Real anonymous catalog reads through the staged production helper; no writes or AI. */
import 'dotenv/config';
import {readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import ts from 'typescript';
import {loadBudgetHelpers} from './load-outfit-budget-helpers.mjs';
import {createClient} from '@supabase/supabase-js';
const root=path.resolve(process.argv[2]||'');if(!process.argv[2])throw Error('Staged function directory required');
const reportName=process.env.BUDGET_CATALOG_READ_REPORT||'budget-catalog-discovery-real-read-v113.json';
if(!/^[a-z0-9-]+\.json$/.test(reportName))throw Error('Invalid report filename');
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,key=process.env.SUPABASE_SERVICE_KEY;
if(!key)throw Error('Backend credential required');
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
 requests.push({phase,status:response.status,limit:url.searchParams.get('limit'),order:url.searchParams.get('order'),cursor:url.searchParams.get('id'),rpc:url.pathname.includes('/rpc/'),latencyMs:Math.round(performance.now()-start)});return response;
}}});
const report={measuredAt:new Date().toISOString(),scope:'Actual public catalog and staged helper with preferences; Node execution, no writes or provider calls.',sourceHashes:hashes};
const originalLog=console.log;console.log=(...args)=>{if(!String(args[0]).startsWith('[Deno macroCategory]'))originalLog(...args);};
try{
 report.scope='Actual budget-ordered backend RPC and staged helper in Node. Compare bounded6choice discovery with exhaustive eligible discovery over the same real slot catalog. No accounts, catalog writes or AI calls; trusted FX cache may refresh through the production cache helper.';
 const {core}=await loadBudgetHelpers();
 const cache={};
 const cacheCode=ts.transpileModule(await readFile(path.join(root,'_shared/budgetFxCache.ts'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
 new Function('require','exports',cacheCode)(()=>core,cache);
 const fx=core.budgetFxContext(await cache.currentBudgetRates(client));
 report.reference=fx.snapshot.id;report.scenarios=[];
 const eligible=item=>!helpers.rejectForStyleOccasion(item,'casual','Everyday')&&!helpers.rejectForClimate(item,{temp:20})&&!helpers.rejectForActivity(item,'Everyday');
 const cost=item=>Number(item.price)>0?core.referenceAmount(Number(item.price),item.currency,'UZS',fx.snapshot,fx.checkedAt):Infinity;
 for(const slot of ['top','bottom','shoes']){
  phase=slot+'-bounded';const start=performance.now();
  const bounded=await api.fetchShopItemsForAllSlotsEdge(client,[slot],'casual',6,undefined,eligible,fx);
  const elapsed=Math.round(performance.now()-start);
  phase=slot+'-exhaustive';
  const all=await api.fetchShopItemsForAllSlotsEdge(client,[slot],'casual',100000,undefined,eligible,fx);
  const minimum=Math.min(...all.map(cost));
  report.scenarios.push({slot,boundedItems:bounded.map(item=>({id:item.id,name:item.name,price:item.price,currency:item.currency})),boundedChoices:bounded.length,eligibleFamilies:all.length,boundedLatencyMs:elapsed,globalMinimumUzs:minimum,boundedMinimumUzs:Math.min(...bounded.map(cost)),minimumPreserved:bounded.some(item=>cost(item)===minimum),allEligible:bounded.every(eligible)});
 }
 const anonymous=createClient(base,process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,{auth:{persistSession:false,autoRefreshToken:false}});
 const denied=await anonymous.rpc('budget_catalog_candidates',{p_slot:'bottom',p_rates:fx.snapshot.uzsPerUnit,p_limit:1});
 report.anonymousDenied=denied.error?.code==='42501';
 report.passed=report.anonymousDenied&&report.scenarios.length===3&&report.scenarios.every(row=>row.boundedChoices===6&&row.minimumPreserved&&row.allEligible)&&requests.every(row=>row.status===200);
}catch(error){report.failure=error.name;report.passed=false;}
finally{console.log=originalLog;report.requests=requests;await writeFile('docs/reports/readiness-evidence/'+reportName,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));if(!report.passed)process.exitCode=1;}
