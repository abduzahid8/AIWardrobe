/** Actual installed FunctionsClient + production wrapper; controlled transport/clock, no network. */
import {readFile,writeFile} from 'node:fs/promises';
import {FunctionsClient} from '@supabase/functions-js';
import ts from 'typescript';
import assert from 'node:assert/strict';
const source=await readFile('src/services/externalAIService.ts','utf8');
const compiled=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
const checks=[];
for(const phase of ['fetch','body']){
 let trigger,budget,signal,entered=false,cleared=false;
 const stall=()=>new Promise((_resolve,reject)=>signal.addEventListener('abort',()=>reject(Object.assign(new Error('aborted'),{name:'AbortError'})),{once:true}));
 const functions=new FunctionsClient('https://controlled.invalid/functions/v1',{customFetch:async(_url,options)=>{
  signal=options.signal;entered=true;
  if(phase==='fetch')return stall();
  return {ok:true,headers:new Headers({'content-type':'application/json'}),json:stall};
 }});
 const exports={};
 Function('exports','require','__DEV__','setTimeout','clearTimeout','console',compiled)(exports,name=>{
  if(name==='../../lib/supabase')return{supabase:{functions}};
  if(name==='../utils/logger')return{default:{debug(){}}};
  throw Error('Unexpected dependency '+name);
 },false,(callback,ms)=>{trigger=callback;budget=ms;return 1;},()=>{cleared=true;},{error(){}});
 const pending=exports.ExternalAIService.classifyOnly('fixture');
 for(let i=0;i<20;i++)await Promise.resolve();
 assert.equal(entered,true);assert.equal(budget,60000);trigger();
 const result=await pending;assert.equal(result.success,false);assert.equal(result.failureKind,'timeout');assert.equal(result.classification,null);assert.equal(signal.aborted,true);assert.equal(cleared,true);
 checks.push({phase,success:false,failureKind:result.failureKind,transportAborted:true,deadlineCleared:true,clockBudgetMs:budget});
}
const report={passed:true,scope:'Installed Supabase FunctionsClient with exact production wrapper and controlled fetch/body stalls plus clock; no network/native UI/capacity claim',checks};
await writeFile('docs/reports/readiness-evidence/scan-client-sdk-deadline.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
