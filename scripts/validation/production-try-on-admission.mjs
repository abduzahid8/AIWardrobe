/** Disposable user, denied/invalid requests only; no image generation requested. */
import 'dotenv/config';
import {randomUUID} from 'node:crypto';
import {writeFile} from 'node:fs/promises';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,service=process.env.SUPABASE_SERVICE_KEY;
if(!service||new URL(base).hostname!=='fyqpifmrsftsfqibhwhy.supabase.co')throw Error('Expected setup');
const admin={apikey:service,Authorization:`Bearer ${service}`,'Content-Type':'application/json'};
async function call(path,headers,body,method='POST',raw=false){const r=await fetch(new URL(path,base),{method,headers,...(body===undefined?{}:{body:raw?body:JSON.stringify(body)}),signal:AbortSignal.timeout(20000),redirect:'error'});let data;try{data=await r.json();}catch{data=null;}return{status:r.status,ok:r.ok,data};}
const report={measuredAt:new Date().toISOString(),runId:randomUUID(),scope:'Auth/entitlement/rate and invalid-input checks; no valid paid rendering input sent',checks:[],cleanup:[]};let owner;
const check=(name,passed,details={})=>report.checks.push({name,passed,...details});
try{
 const password=`${randomUUID()}Aa!9`,email=`readiness-${randomUUID()}@aiwardrobe.club`;
 const made=await call('/auth/v1/admin/users',admin,{email,password,email_confirm:true,app_metadata:{readiness_audit:report.runId}});if(!made.ok||!made.data?.id)throw Error('Fixture creation failed');owner=made.data.id;report.fixtureOwner=owner;
 const login=await call('/auth/v1/token?grant_type=password',{apikey:anon,'Content-Type':'application/json'},{email,password});if(!login.data?.access_token)throw Error('Login failed');const headers={apikey:anon,Authorization:`Bearer ${login.data.access_token}`,'Content-Type':'application/json'};
 const input={person_image:'AA==',garment_image:'AA==',garment_type:'upper_body'};
 const publicJwt=await call('/functions/v1/try-on',{apikey:anon,Authorization:`Bearer ${anon}`,'Content-Type':'application/json'},input);check('public_anon_denied',publicJwt.status===401,{status:publicJwt.status});
 const missing=await call('/functions/v1/try-on',{apikey:anon,'Content-Type':'application/json'},{});check('missing_auth_denied',missing.status===401,{status:missing.status});
 const free=await call('/functions/v1/try-on',headers,input);check('free_try_on_denied',free.status===403&&free.data?.error==='feature_limit_reached',{status:free.status});
 const trial=await call(`/rest/v1/profiles?id=eq.${owner}`,headers,{trial_started_at:new Date().toISOString()},'PATCH');if(!trial.ok)throw Error('Trial fixture failed');
 for(const [name,body,raw] of [['trial_invalid_image',{person_image:{},garment_image:'AA=='},false],['trial_invalid_type',{...input,garment_type:'unsupported'},false],['trial_malformed_json','{',true]]){const r=await call('/functions/v1/try-on',headers,body,'POST',raw);check(name,r.status===400&&r.data?.success===false,{status:r.status});}
 const bucket=Math.floor(Date.now()/60000)*60000;let count=0;
 for(let i=0;i<5&&count<5;i++){const seeded=await call('/rest/v1/rpc/increment_rate_bucket',admin,{p_user:owner,p_feature:`rl:try-on:${bucket}`});if(!seeded.ok||!Number.isSafeInteger(seeded.data))throw Error('Rate seed failed');count=seeded.data;}
 const rate=await call('/functions/v1/try-on',headers,{});check('five_per_minute_enforced',Math.floor(Date.now()/60000)*60000===bucket&&rate.status===429&&rate.data?.error==='rate_limited',{status:rate.status,seededBucketCount:count,sameFixedWindow:Math.floor(Date.now()/60000)*60000===bucket});
}catch(error){report.failure=error.name;process.exitCode=1;}
finally{if(owner){const removed=await call(`/auth/v1/admin/users/${owner}`,admin,undefined,'DELETE');report.cleanup.push({deleted:removed.ok,status:removed.status});if(!removed.ok)process.exitCode=1;}await writeFile(new URL('../../docs/reports/readiness-evidence/production-try-on-admission.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));}
if(report.checks.length!==7||report.checks.some(c=>!c.passed))process.exitCode=1;
