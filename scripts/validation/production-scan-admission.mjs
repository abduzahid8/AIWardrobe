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
const report={measuredAt:new Date().toISOString(),scope:'Exhausted disposable scan quotas and invalid inputs only; no provider calls',checks:[]};
const check=(name,passed,details={})=>report.checks.push({name,passed,...details});let owner;
try{
 const suffix=randomUUID(),password=`${randomUUID()}Aa!9`,email=`readiness-${suffix}@aiwardrobe.club`;
 const created=await call('/auth/v1/admin/users',admin,{email,password,email_confirm:true,app_metadata:{readiness_audit:suffix}});if(!created.ok||!created.data?.id)throw new Error('Fixture creation failed');owner=created.data.id;
 const login=await call('/auth/v1/token?grant_type=password',{apikey:anon,'Content-Type':'application/json'},{email,password});if(!login.data?.access_token)throw new Error('Fixture login failed');
 const user={apikey:anon,Authorization:`Bearer ${login.data.access_token}`,'Content-Type':'application/json'};
 const seeded=await call('/rest/v1/usage_counters',admin,{user_id:owner,feature:'edge:wardrobeScans',day:new Date().toISOString().slice(0,10),used:5});if(!seeded.ok)throw new Error('Quota fixture failed');
 const endpoint='/functions/v1/ai-process',payload={image:'AA==',operation:'classify'};
 const missing=await call(endpoint,user,{});check('missing_image_rejected_without_quota',missing.status===400,{status:missing.status});
 const invalid=await call(endpoint,user,{image:12,operation:'classify'});check('invalid_image_type_rejected',invalid.status===400,{status:invalid.status});
 const unknown=await call(endpoint,user,{image:'AA==',operation:'invented'});check('unknown_operation_rejected',unknown.status===400,{status:unknown.status});
 const denied=await call(endpoint,user,{...payload,tier:'vip',p_limit:-1});check('free_five_scan_limit_enforced_despite_client_claims',denied.status===403&&denied.data?.error==='feature_limit_reached',{status:denied.status});
 const alternate=await call(endpoint,user,{...payload,operation:'remove_bg'});check('alternate_operation_cannot_bypass_scan_quota',alternate.status===403,{status:alternate.status});
 const publicRequest=await call(endpoint,{apikey:anon,Authorization:`Bearer ${anon}`,'Content-Type':'application/json'},payload);check('public_anon_cannot_scan',publicRequest.status===401,{status:publicRequest.status});
 const counts=await call(`/rest/v1/usage_counters?user_id=eq.${owner}&feature=eq.edge:wardrobeScans&select=used`,admin,undefined,'GET');check('invalid_and_denied_attempts_do_not_increase_daily_quota',counts.ok&&counts.data?.[0]?.used===5,{used:counts.data?.[0]?.used});
 const paid=await call('/rest/v1/subscriptions',admin,{user_id:owner,tier:'lite',status:'active',end_date:new Date(Date.now()+86400000).toISOString()});if(!paid.ok)throw new Error('Lite fixture failed');
 const full=await call(`/rest/v1/usage_counters?user_id=eq.${owner}&feature=eq.edge:wardrobeScans`,admin,{used:30},'PATCH');if(!full.ok)throw new Error('Lite quota fixture failed');
 const lite=await call(endpoint,user,payload);check('lite_thirty_scan_limit_enforced',lite.status===403&&lite.data?.error==='feature_limit_reached',{status:lite.status});
}catch(error){report.failure=error.name||'Error';process.exitCode=1;}
finally{
 if(owner){const removed=await call(`/auth/v1/admin/users/${owner}`,admin,undefined,'DELETE');report.cleanup={temporaryUserDeleted:removed.ok,status:removed.status};if(!removed.ok)process.exitCode=1;}
 await writeFile(new URL('../../docs/reports/readiness-evidence/production-scan-admission.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}
if(report.checks.length!==8||report.checks.some(c=>!c.passed))process.exitCode=1;
