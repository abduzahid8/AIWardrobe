/** Bounded production quota checks on one tagged disposable account; no AI calls. */
import 'dotenv/config';
import {randomUUID} from 'node:crypto';
import {writeFile} from 'node:fs/promises';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,service=process.env.SUPABASE_SERVICE_KEY;
if(!service)throw new Error('Setup credential required');
const admin={apikey:service,Authorization:`Bearer ${service}`,'Content-Type':'application/json'};
async function call(path,headers,body,method='POST'){
 const response=await fetch(new URL(path,base),{method,headers,...(body===undefined?{}:{body:JSON.stringify(body)}),signal:AbortSignal.timeout(15000),redirect:'error'});
 let data;try{data=await response.json();}catch{data=null;}return {ok:response.ok,status:response.status,data};
}
const report={measuredAt:new Date().toISOString(),scope:'Bounded production generation rate gate; invalid budgets prevent all AI calls',checks:[]};
const check=(name,passed,details={})=>report.checks.push({name,passed,...details});
const owners=[];
async function fixture(){
 const suffix=randomUUID(),password=`${randomUUID()}Aa!9`,email=`readiness-${suffix}@aiwardrobe.club`;
 const created=await call('/auth/v1/admin/users',admin,{email,password,email_confirm:true,app_metadata:{readiness_audit:suffix}});
 if(!created.ok||!created.data?.id)throw new Error('Fixture creation failed');owners.push(created.data.id);
 const login=await call('/auth/v1/token?grant_type=password',{apikey:anon,'Content-Type':'application/json'},{email,password});if(!login.data?.access_token)throw new Error('Fixture login failed');
 return {id:created.data.id,headers:{apikey:anon,Authorization:`Bearer ${login.data.access_token}`,'Content-Type':'application/json'}};
}
try{
 const owner=await fixture(),other=await fixture();
 const payload={preferences:{maxShopBudget:{amount:-1,currency:'USD'}},useProvidedWardrobeOnly:true,wardrobeItems:[]};
 for(const account of [owner,other]){
  const started=await call(`/rest/v1/profiles?id=eq.${account.id}`,account.headers,{trial_started_at:new Date().toISOString()},'PATCH');if(!started.ok)throw new Error('Trial fixture start failed');
 }
 const endpoint='/functions/v1/generate-outfits';
 const preflight=await call(endpoint,{apikey:anon},undefined,'OPTIONS');check('cors_preflight_not_metered',preflight.ok,{status:preflight.status});
 const invalidAuth=await call(endpoint,{apikey:anon,Authorization:`Bearer ${anon}`,'Content-Type':'application/json'},payload);check('anon_not_authenticated',invalidAuth.status===401,{status:invalidAuth.status});
 const results=await Promise.all(Array.from({length:12},()=>call(endpoint,owner.headers,payload)));
 const statuses=results.map(r=>r.status);const allowed=results.filter(r=>r.status===400&&r.data?.error==='invalid_outfit_budget').length,denied=results.filter(r=>r.status===429&&r.data?.error==='rate_limited'&&Number.isInteger(r.data?.retry_after_sec)&&r.data.retry_after_sec>0&&r.data.retry_after_sec<=60).length;
 check('concurrent_requests_capped_at_ten_per_window',allowed===10&&denied===2,{statuses,allowed,denied});
 const independent=await call(endpoint,other.headers,payload);check('second_user_has_independent_allowance',independent.status===400&&independent.data?.error==='invalid_outfit_budget',{status:independent.status});
 const counters=await call(`/rest/v1/usage_counters?user_id=eq.${owner.id}&select=feature,used`,admin,undefined,'GET');
 check('only_rate_admitted_attempts_reach_paid_meter',counters.ok&&Array.isArray(counters.data)&&counters.data.find(r=>r.feature==='edge:aiOutfits')?.used===10,{counters:counters.data});
}catch(error){report.failure=error.name||'Error';process.exitCode=1;}
finally{
 report.cleanup=[];
 for(const owner of owners){const removed=await call(`/auth/v1/admin/users/${owner}`,admin,undefined,'DELETE');report.cleanup.push({temporaryUserDeleted:removed.ok,status:removed.status});if(!removed.ok)process.exitCode=1;}
 await writeFile(new URL('../../docs/reports/readiness-evidence/production-generation-rate-admission.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}
if(report.checks.length!==5||report.checks.some(c=>!c.passed))process.exitCode=1;
