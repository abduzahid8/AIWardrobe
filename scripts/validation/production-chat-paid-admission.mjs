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
const report={measuredAt:new Date().toISOString(),scope:'Paid chat admission on one disposable account; empty query prevents all provider calls',checks:[]};
const check=(name,passed,details={})=>report.checks.push({name,passed,...details});
let owner;
try{
 const suffix=randomUUID(),password=`${randomUUID()}Aa!9`,email=`readiness-${suffix}@aiwardrobe.club`;
 const created=await call('/auth/v1/admin/users',admin,{email,password,email_confirm:true,app_metadata:{readiness_audit:suffix}});
 if(!created.ok||!created.data?.id)throw new Error('Fixture creation failed');owner=created.data.id;
 const login=await call('/auth/v1/token?grant_type=password',{apikey:anon,'Content-Type':'application/json'},{email,password});if(!login.data?.access_token)throw new Error('Fixture login failed');
 const user={apikey:anon,Authorization:`Bearer ${login.data.access_token}`,'Content-Type':'application/json'};
 const payload={query:''};
 const attempt=()=>call('/functions/v1/ai-chat',user,payload);
 const denied=await attempt();check('free_user_denied_before_chat',denied.status===403&&denied.data?.error==='feature_limit_reached',{status:denied.status});
 const started=await call(`/rest/v1/profiles?id=eq.${owner}`,user,{trial_started_at:new Date().toISOString()},'PATCH');if(!started.ok)throw new Error('Trial fixture start failed');
 const trial=await attempt();check('valid_trial_reaches_input_validation',trial.status===400&&trial.data?.error==='Validation failed',{status:trial.status});
 const expired=await call(`/rest/v1/profiles?id=eq.${owner}`,admin,{trial_started_at:new Date(Date.now()-8*86400000).toISOString()},'PATCH');if(!expired.ok)throw new Error('Trial fixture expiry failed');
 const deniedAgain=await attempt();check('expired_trial_denied',deniedAgain.status===403&&deniedAgain.data?.error==='feature_limit_reached',{status:deniedAgain.status});
 for(const tier of ['lite','premium','vip']){
  const insert=await call('/rest/v1/subscriptions',admin,{user_id:owner,tier,status:'active',end_date:new Date(Date.now()+86400000).toISOString()});if(!insert.ok)throw new Error('Paid fixture setup failed');
  const result=await attempt();check(`${tier}_reaches_input_validation`,result.status===400&&result.data?.error==='Validation failed',{status:result.status});
  const remove=await call(`/rest/v1/subscriptions?user_id=eq.${owner}`,admin,undefined,'DELETE');if(!remove.ok)throw new Error('Paid fixture transition failed');
 }
 const swipes=await call('/rest/v1/rpc/check_and_increment_usage',user,{p_feature:'homeSwipes',p_limit:-1,p_amount:3});check('free_three_swipes_preserved',swipes.ok&&swipes.data?.[0]?.allowed===true,{status:swipes.status});
 const scan=await call('/rest/v1/rpc/check_and_increment_usage',user,{p_feature:'wardrobeScans',p_limit:-1,p_amount:5});check('free_five_scans_preserved',scan.ok&&scan.data?.[0]?.allowed===true,{status:scan.status});
}catch(error){report.failure=error.name||'Error';process.exitCode=1;}
finally{
 if(owner){const removed=await call(`/auth/v1/admin/users/${owner}`,admin,undefined,'DELETE');report.cleanup={temporaryUserDeleted:removed.ok,status:removed.status};if(!removed.ok)process.exitCode=1;}
 await writeFile(new URL('../../docs/reports/readiness-evidence/production-chat-paid-admission.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}
if(report.checks.length!==8||report.checks.some(c=>!c.passed))process.exitCode=1;
