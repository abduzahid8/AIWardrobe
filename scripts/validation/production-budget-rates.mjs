/** Scoped production reference-rate/cache checks with one disposable free account. */
import 'dotenv/config';
import {randomUUID} from 'node:crypto';import {writeFile,readFile} from 'node:fs/promises';import ts from 'typescript';
const core={};Function('exports',ts.transpileModule(await readFile('backend/supabase/functions/_shared/budgetFx.ts','utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText)(core);
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,key=process.env.SUPABASE_SERVICE_KEY;if(!key)throw Error('Setup credential required');
const admin={apikey:key,Authorization:`Bearer ${key}`,'Content-Type':'application/json'},publicHeaders={apikey:anon,Authorization:`Bearer ${anon}`,'Content-Type':'application/json'};
const report={measuredAt:new Date().toISOString(),scope:'Dated reference-rate service and cache permissions; not outfit integration, checkout FX or 10000 DAU certification',checks:[]};let owner;
const call=async(path,headers,method='GET',body)=>{const start=performance.now(),r=await fetch(new URL(path,base),{method,headers,...(body===undefined?{}:{body:JSON.stringify(body)}),signal:AbortSignal.timeout(25000),redirect:'error'});return{status:r.status,ok:r.ok,data:await r.json().catch(()=>null),latencyMs:Math.round(performance.now()-start)};};
const check=(name,r,passed)=>report.checks.push({name,status:r.status,latencyMs:r.latencyMs,passed});
try{
 const suffix=randomUUID(),email=`fx-${suffix}@example.invalid`,password=randomUUID()+'Aa!9';
 const created=await call('/auth/v1/admin/users',admin,'POST',{email,password,email_confirm:true,app_metadata:{readiness_audit:suffix},user_metadata:{full_name:'FX Readiness Audit'}});if(!created.ok||!created.data?.id)throw Error('Fixture setup failed');owner=created.data.id;
 const session=await call('/auth/v1/token?grant_type=password',publicHeaders,'POST',{email,password});if(!session.data?.access_token)throw Error('Fixture login failed');const user={...publicHeaders,Authorization:`Bearer ${session.data.access_token}`};
 let r=await call('/functions/v1/budget-rates',publicHeaders);check('anonymous_cannot_use_rate_endpoint',r,r.status===401);
 r=await call('/functions/v1/budget-rates',user,'POST',{});let snapshot;
 try{snapshot=core.validateBudgetFxSnapshot(r.data?.snapshot);}catch{}
 check('free_authenticated_account_receives_valid_reference_snapshot',r,r.ok&&!!snapshot);
 if(!snapshot)throw Error('Hosted snapshot unavailable');report.snapshot=snapshot;
 r=await call('/functions/v1/budget-rates',user,'POST',{});check('fresh_cache_snapshot_reused',r,r.ok&&r.data?.snapshot?.id===snapshot.id&&r.data?.snapshot?.fetchedAt===snapshot.fetchedAt);
 r=await call('/rest/v1/budget_exchange_rates?select=id&limit=1',publicHeaders);check('anonymous_table_read_denied',r,[401,403].includes(r.status));
 r=await call(`/rest/v1/budget_exchange_rates?id=eq.${encodeURIComponent(snapshot.id)}`,user,'PATCH',{snapshot:{source:'forged'}});check('client_cannot_replace_reference_values',r,r.status===403&&r.data?.code==='42501');
 r=await call('/rest/v1/budget_exchange_rates',user,'POST',{id:'forged',as_of:snapshot.asOf,fetched_at:snapshot.fetchedAt,snapshot});check('client_cannot_insert_reference_values',r,r.status===403&&r.data?.code==='42501');
 r=await call('/functions/v1/budget-rates',user,'DELETE');check('endpoint_has_no_client_write_method',r,r.status===405);
 report.referenceExample={EUR100InUSD:core.referenceAmount(100,'EUR','USD',snapshot),asOf:snapshot.asOf};
}catch(error){report.failure=error.message;process.exitCode=1;}finally{
 if(owner){const r=await call(`/auth/v1/admin/users/${owner}`,admin,'DELETE');report.cleanup={temporaryUserDeleted:r.ok,status:r.status};if(!r.ok)process.exitCode=1;}
 await writeFile('docs/reports/readiness-evidence/production-budget-rates.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}if(!report.checks.length||report.checks.some(x=>!x.passed))process.exitCode=1;
