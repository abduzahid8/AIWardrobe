/** Bounded read-only RPC load; no AI, auth fixtures or application writes. */
import 'dotenv/config';
import {readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import ts from 'typescript';
import {createClient} from '@supabase/supabase-js';
import {loadBudgetHelpers} from './load-outfit-budget-helpers.mjs';
const root=process.argv[2];if(!root||!process.env.SUPABASE_SERVICE_KEY)throw Error('Backend test configuration required');
const admin=createClient(process.env.EXPO_PUBLIC_SUPABASE_URL,process.env.SUPABASE_SERVICE_KEY,{auth:{persistSession:false,autoRefreshToken:false}});
const report={measuredAt:new Date().toISOString(),scope:'Direct read-only backend budget_catalog_candidates RPC at2,10,40RPS. No AI, media, user auth, billing, native UI or10000DAU certification. FX cache may refresh.',phases:[]};
const {core}=await loadBudgetHelpers(),cache={};
new Function('require','exports',ts.transpileModule(await readFile(path.join(root,'_shared/budgetFxCache.ts'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText)(()=>core,cache);
const fx=core.budgetFxContext(await cache.currentBudgetRates(admin));report.reference=fx.snapshot.id;
const stop=new AbortController();let total=0,failed=0;
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const percentile=(values,p)=>values.length?Math.round([...values].sort((a,b)=>a-b)[Math.min(values.length-1,Math.ceil(values.length*p)-1)]):null;
try{
 for(const [rps,seconds] of [[2,10],[10,20],[40,60]]){
  const phase={rps,seconds,scheduled:rps*seconds,attempted:0,succeeded:0,failed:0,dropped:0,latencies:[],errors:{}};
  const pending=new Set(),start=performance.now();
  for(let i=0;i<phase.scheduled&&!stop.signal.aborted;i++){
   const due=start+i*1000/rps;await sleep(Math.max(0,due-performance.now()));
   if(performance.now()-due>100||pending.size>=80){phase.dropped++;continue;}
   phase.attempted++;const slot=['top','bottom','shoes'][i%3];
   const request=(async()=>{
    const began=performance.now();let code;
    try{
     const {data,error}=await admin.rpc('budget_catalog_candidates',{p_slot:slot,p_rates:fx.snapshot.uzsPerUnit,p_limit:100}).abortSignal(AbortSignal.any([stop.signal,AbortSignal.timeout(5000)]));
     if(error)code=error.code||'transport';
     else if(!Array.isArray(data)||data.length!==100||data.some((row,index)=>typeof row._budget_cost!=='string'||!Number.isFinite(Number(row._budget_cost))||(index>0&&Number(row._budget_cost)<Number(data[index-1]._budget_cost))))code='invalid_ordered_page';
    }catch(error){code=error.name||'transport';}
    phase.latencies.push(performance.now()-began);total++;
    if(code){failed++;phase.failed++;phase.errors[code]=(phase.errors[code]||0)+1;}else phase.succeeded++;
    if(total>=100&&failed>=5&&failed/total>0.02)stop.abort();
   })();pending.add(request);request.finally(()=>pending.delete(request));
  }
  await Promise.all(pending);
  phase.elapsedMs=Math.round(performance.now()-start);phase.p50Ms=percentile(phase.latencies,.5);phase.p95Ms=percentile(phase.latencies,.95);phase.p99Ms=percentile(phase.latencies,.99);delete phase.latencies;
  report.phases.push(phase);console.log(JSON.stringify({phase}));
  if(stop.signal.aborted)break;
 }
 report.passed=report.phases.length===3&&report.phases.every(phase=>phase.attempted===phase.scheduled&&phase.failed===0&&phase.dropped===0);
}catch(error){report.failure=error.name;report.passed=false;}
finally{
 report.finishedAt=new Date().toISOString();report.safetyStop=stop.signal.aborted;
 await writeFile('docs/reports/readiness-evidence/budget-catalog-rpc-load-v113.json',JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify({passed:report.passed,phases:report.phases.length,total,failed,safetyStop:report.safetyStop}));if(!report.passed)process.exitCode=1;
}
