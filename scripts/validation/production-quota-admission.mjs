/** Bounded production quota checks on one tagged disposable account; no AI calls. */
import 'dotenv/config';
import {randomUUID} from 'node:crypto';
import {writeFile} from 'node:fs/promises';
const reportName=process.env.QUOTA_REPORT;
if(reportName&&!/^[a-z0-9-]+\.json$/.test(reportName))throw new Error('Invalid report name');
const mode=process.argv[2];if(!['before','after'].includes(mode))throw new Error('Choose before or after');
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,service=process.env.SUPABASE_SERVICE_KEY;
if(!service)throw new Error('Setup credential required');
const admin={apikey:service,Authorization:`Bearer ${service}`,'Content-Type':'application/json'};
async function call(path,headers,body,method='POST'){
 const response=await fetch(new URL(path,base),{method,headers,...(body===undefined?{}:{body:JSON.stringify(body)}),signal:AbortSignal.timeout(15000),redirect:'error'});
 let data;try{data=await response.json();}catch{data=null;}return {ok:response.ok,status:response.status,data};
}
const report={measuredAt:new Date().toISOString(),scope:'Disposable-user quota RPC and role checks; no AI/provider calls or customer data changes',mode,checks:[]};
const check=(name,passed,details={})=>report.checks.push({name,passed,...details});
let owner;
try{
 const suffix=randomUUID(),password=`${randomUUID()}Aa!9`,email=`readiness-${suffix}@aiwardrobe.club`;
 const created=await call('/auth/v1/admin/users',admin,{email,password,email_confirm:true,app_metadata:{readiness_audit:suffix}});
 if(!created.ok||!created.data?.id)throw new Error('Fixture creation failed');owner=created.data.id;
 const login=await call('/auth/v1/token?grant_type=password',{apikey:anon,'Content-Type':'application/json'},{email,password});if(!login.data?.access_token)throw new Error('Fixture login failed');
 const user={apikey:anon,Authorization:`Bearer ${login.data.access_token}`,'Content-Type':'application/json'};
 const rpc=(name,body,headers=user)=>call(`/rest/v1/rpc/${name}`,headers,body);
 if(mode==='before'){
  const result=await rpc('check_and_increment_usage',{p_feature:'homeSwipes',p_limit:-1,p_amount:10});
  report.observed=result.data;report.status=result.status;
 }else{
  const free=await rpc('consume_edge_quota',{p_user:owner,p_endpoint:'generate-outfits'},admin);check('free_paid_work_denied',free.ok&&free.data===false,{status:free.status});
  const outcomes=await Promise.all(Array.from({length:20},()=>rpc('check_and_increment_usage',{p_feature:'homeSwipes',p_limit:-1,p_amount:1})));
  const allowed=outcomes.filter(r=>r.data?.[0]?.allowed===true).length;
  check('concurrent_free_swipes_capped_despite_forged_unlimited',outcomes.every(r=>r.ok)&&allowed===3,{requests:20,allowed});
  const scanReservations=await Promise.all(Array.from({length:20},()=>rpc('check_and_increment_usage',{p_feature:'wardrobeScans',p_limit:999999,p_amount:1})));
  check('concurrent_free_scan_reservations_capped',scanReservations.every(r=>r.ok)&&scanReservations.filter(r=>r.data?.[0]?.allowed===true).length===5,{requests:20,allowed:scanReservations.filter(r=>r.data?.[0]?.allowed===true).length});
  const scanWork=await Promise.all(Array.from({length:20},()=>rpc('consume_edge_quota',{p_user:owner,p_endpoint:'ai-process'},admin)));
  check('concurrent_free_actual_scan_work_capped',scanWork.every(r=>r.ok)&&scanWork.filter(r=>r.data===true).length===5,{requests:20,allowed:scanWork.filter(r=>r.data===true).length});
  const inflated=await rpc('check_and_increment_usage',{p_feature:'wardrobeScans',p_limit:999999,p_amount:100});
  check('inflated_bulk_amount_does_not_raise_allowance',inflated.ok&&inflated.data?.[0]?.allowed===false&&inflated.data?.[0]?.used===5);
  const invalid=await rpc('check_and_increment_usage',{p_feature:'homeSwipes',p_limit:999999,p_amount:0});check('zero_amount_rejected',invalid.status===400,{status:invalid.status});
  const unknown=await rpc('check_and_increment_usage',{p_feature:'invented_feature',p_limit:-1,p_amount:1});check('unknown_feature_rejected',unknown.status===400,{status:unknown.status});
  const blocked=await rpc('consume_edge_quota',{p_user:owner,p_endpoint:'generate-outfits'});check('client_cannot_call_server_meter',!blocked.ok,{status:blocked.status});
  const tierDenied=await rpc('effective_subscription_tier',{p_user:owner});check('client_cannot_query_arbitrary_user_tier',!tierDenied.ok,{status:tierDenied.status});
  const started=await call(`/rest/v1/profiles?id=eq.${owner}`,{...user,Prefer:'return=representation'},{trial_started_at:'2099-01-01T00:00:00Z'},'PATCH');
  const trial=started.data?.[0]?.trial_started_at;
  check('first_trial_uses_server_time',started.ok&&!!trial&&Math.abs(Date.parse(trial)-Date.now())<60000,{status:started.status});
  const tier=await rpc('effective_subscription_tier',{p_user:owner},admin);check('server_trial_receives_premium',tier.ok&&tier.data==='premium');
  const admitted=await rpc('consume_edge_quota',{p_user:owner,p_endpoint:'generate-outfits'},admin);check('valid_trial_paid_work_admitted',admitted.ok&&admitted.data===true);
  const expired=await call(`/rest/v1/profiles?id=eq.${owner}`,admin,{trial_started_at:new Date(Date.now()-8*86400000).toISOString()},'PATCH');if(!expired.ok)throw new Error('Fixture trial transition failed');
  const denied=await rpc('consume_edge_quota',{p_user:owner,p_endpoint:'generate-outfits'},admin);check('expired_trial_paid_work_denied',denied.ok&&denied.data===false);
  const renewed=await call(`/rest/v1/profiles?id=eq.${owner}`,user,{trial_started_at:'2099-01-01T00:00:00Z'},'PATCH');check('client_cannot_restart_trial',!renewed.ok,{status:renewed.status});
  const counters=await call(`/rest/v1/usage_counters?user_id=eq.${owner}&select=feature,used`,admin,undefined,'GET');
  check('exact_persisted_counters',counters.ok&&counters.data?.find(row=>row.feature==='homeSwipes')?.used===3&&counters.data?.find(row=>row.feature==='edge:aiOutfits')?.used===1&&counters.data?.find(row=>row.feature==='wardrobeScans')?.used===5&&counters.data?.find(row=>row.feature==='edge:wardrobeScans')?.used===5,{counters:counters.data});
 }
}catch(error){report.failure=error.name||'Error';process.exitCode=1;}
finally{
 if(owner){const removed=await call(`/auth/v1/admin/users/${owner}`,admin,undefined,'DELETE');report.cleanup={temporaryUserDeleted:removed.ok,status:removed.status};if(!removed.ok)process.exitCode=1;}
 await writeFile(new URL(`../../docs/reports/readiness-evidence/${reportName||`production-quota-${mode}.json`}`,import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}
if(mode==='after'&&(!report.checks.length||report.checks.some(c=>!c.passed)))process.exitCode=1;
