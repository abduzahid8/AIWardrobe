/** Installed FunctionsClient + production rate loader with controlled fetch/body stalls. */
import {readFile,writeFile} from 'node:fs/promises';import {FunctionsClient} from '@supabase/functions-js';import ts from 'typescript';import assert from 'node:assert/strict';
const compile=source=>ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
const core={};Function('exports',compile(await readFile('backend/supabase/functions/_shared/budgetFx.ts','utf8')))(core);
const deadlineCode=compile(await readFile('src/lib/requestDeadline.ts','utf8')),serviceCode=compile(await readFile('src/services/budgetRates.ts','utf8')),checks=[];
for(const phase of ['fetch','body']){
 let trigger,budget,signal,entered=false,cleared=false;
 const stall=()=>new Promise((_resolve,reject)=>signal.addEventListener('abort',()=>reject(Object.assign(new Error('aborted'),{name:'AbortError'})),{once:true}));
 const functions=new FunctionsClient('https://controlled.invalid/functions/v1',{customFetch:async(_url,options)=>{signal=options.signal;entered=true;if(phase==='fetch')return stall();return{ok:true,headers:new Headers({'content-type':'application/json'}),json:stall};}});
 const deadline={};Function('exports','setTimeout','clearTimeout',deadlineCode)(deadline,(callback,ms)=>{trigger=callback;budget=ms;return 1;},()=>{cleared=true;});
 const api={};Function('exports','require',serviceCode)(api,name=>name==='../../lib/supabase'?{supabase:{functions}}:name==='../lib/requestDeadline'?deadline:core);
 const result=api.getBudgetRates();for(let i=0;i<20;i++)await Promise.resolve();assert.equal(entered,true);assert.equal(budget,15000);trigger();await assert.rejects(result,/timed out/);assert.equal(signal.aborted,true);assert.equal(cleared,true);
 checks.push({phase,callerSettles:true,transportAborted:true,clockBudgetMs:budget,deadlineCleared:true});
}
const report={date:'2026-10-20',passed:true,scope:'Exact new client loader with installed FunctionsClient and controlled transport/clock; not connected to outfit generation or native UI yet',checks};await writeFile('docs/reports/readiness-evidence/budget-rates-sdk-deadline.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
