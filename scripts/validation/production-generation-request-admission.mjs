/** Small live admission requests only. Never sends oversized streams or invokes a provider intentionally. */
import 'dotenv/config';
import {randomUUID} from 'node:crypto';
import {writeFile} from 'node:fs/promises';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,service=process.env.SUPABASE_SERVICE_KEY;
if(!base||!anon||!service)throw Error('Fixture setup configuration required');
const admin={apikey:service,Authorization:`Bearer ${service}`,'Content-Type':'application/json'};
const report={measuredAt:new Date().toISOString(),scope:'Small production request-admission checks; no oversized production bodies, not generation/capacity certification',checks:[]};
const call=async(path,headers,body,method='POST')=>{const started=performance.now();const response=await fetch(new URL(path,base),{method,headers,...body===undefined?{}:{body:typeof body==='string'?body:JSON.stringify(body)},signal:AbortSignal.timeout(30000)});let data;try{data=await response.json();}catch{}return{status:response.status,data,latencyMs:Math.round(performance.now()-started)};};
let owner,access;
try{
 const suffix=randomUUID(),email=`readiness-${suffix}@aiwardrobe.club`,password=`${randomUUID()}Aa!9`;
 const created=await call('/auth/v1/admin/users',admin,{email,password,email_confirm:true,app_metadata:{readiness_audit:suffix},user_metadata:{username:`readiness_${suffix.replaceAll('-','').slice(0,20)}`,readiness_test:true}});
 if(created.status!==200||!created.data?.id)throw Error('Fixture creation failed');owner=created.data.id;
 const login=await call('/auth/v1/token?grant_type=password',{apikey:anon,'Content-Type':'application/json'},{email,password});access=login.data?.access_token;if(!access)throw Error('Fixture login failed');
 const headers={apikey:anon,Authorization:`Bearer ${access}`,'Content-Type':'application/json'};
 const trial=await call(`/rest/v1/profiles?id=eq.${owner}`,admin,{trial_started_at:new Date().toISOString()},'PATCH');
 if(![200,204].includes(trial.status))throw Error('Trial fixture initialization failed');
 const valid={prompt:'Create one complete casual outfit.',stylePreferences:'casual',occasion:'Everyday',weather:{temp:20,condition:'Clear'},limit:1,preferences:{maxShopBudget:{amount:1,currency:'USD'}}};
 for(const [name,body,status,code] of [['outfit_count_exceeds_product_max',{...valid,limit:16},400,'invalid_generation_request'],['non_object_root',[],400,'invalid_generation_request'],['malformed_json','{',400,'invalid_generation_request'],['valid_request_reaches_budget_admission',valid,422,'no_items_within_budget']]){
  const result=await call('/functions/v1/generate-outfits',headers,body);report.checks.push({name,status:result.status,code:result.data?.code||result.data?.error,latencyMs:result.latencyMs,passed:result.status===status&&(result.data?.code||result.data?.error)===code});
 }
}catch(error){report.failure=error.message==='Fixture creation failed'||error.message==='Fixture login failed'?error.message:error.name;process.exitCode=1;}
finally{
 report.cleanup={};
 if(access){try{const revoked=await call('/auth/v1/logout?scope=global',{apikey:anon,Authorization:`Bearer ${access}`},undefined);report.cleanup.refreshSessionsRevoked=revoked.status===204;report.cleanup.revokeStatus=revoked.status;}catch{report.cleanup.refreshSessionsRevoked=false;}}
 if(owner){try{const removed=await call(`/auth/v1/admin/users/${owner}`,admin,undefined,'DELETE');report.cleanup.userDeleted=removed.status===200;}catch{report.cleanup.userDeleted=false;}}
 if(owner&&!report.cleanup.userDeleted)process.exitCode=1;
 await writeFile(new URL('../../docs/reports/readiness-evidence/v105-generation-request-admission-live.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}
if(report.checks.length!==4||report.checks.some(check=>!check.passed))process.exitCode=1;
