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
const report={measuredAt:new Date().toISOString(),scope:'Scheduled cleanup of one disposable-user fixture only',checks:[]};
let owner;
try{
 const suffix=randomUUID();
 const created=await call('/auth/v1/admin/users',admin,{email:`readiness-${suffix}@aiwardrobe.club`,password:`${randomUUID()}Aa!9`,email_confirm:true,app_metadata:{readiness_audit:suffix}});
 if(!created.ok||!created.data?.id)throw new Error('Fixture creation failed');owner=created.data.id;
 const today=new Date().toISOString().slice(0,10);
 const day=offset=>new Date(Date.parse(`${today}T00:00:00Z`)-offset*86400000).toISOString().slice(0,10);
 const rows=[0,1,2,3].map(offset=>({user_id:owner,feature:`__retention_live:${offset}`,day:day(offset),used:3}));
 const seeded=await call('/rest/v1/usage_counters',admin,rows);if(!seeded.ok)throw new Error('Fixture seed failed');
 const started=Date.now();let observed;
 for(let attempt=0;attempt<11;attempt++){
  const result=await call(`/rest/v1/usage_counters?user_id=eq.${owner}&select=feature,day,used`,admin,undefined,'GET');
  if(!result.ok||!Array.isArray(result.data))throw new Error('Fixture read failed');observed=result.data;
  if(!observed.some(r=>r.feature==='__retention_live:3'))break;
  if(attempt<10)await new Promise(resolve=>setTimeout(resolve,8000));
 }
 report.elapsedMs=Date.now()-started;
 report.checks.push({name:'scheduled_cleanup_removed_expired_fixture',passed:!observed.some(r=>r.feature==='__retention_live:3')});
 report.checks.push({name:'current_and_retention_boundary_counters_preserved',passed:observed.length===3&&[0,1,2].every(offset=>observed.some(r=>r.feature===`__retention_live:${offset}`&&r.used===3&&r.day===day(offset)))});
 report.observed=observed;
}catch(error){report.failure=error.name||'Error';process.exitCode=1;}
finally{
 if(owner){const removed=await call(`/auth/v1/admin/users/${owner}`,admin,undefined,'DELETE');report.cleanup={temporaryUserDeleted:removed.ok,status:removed.status};if(!removed.ok)process.exitCode=1;}
 await writeFile(new URL('../../docs/reports/readiness-evidence/production-usage-retention.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}
if(report.checks.length!==2||report.checks.some(c=>!c.passed))process.exitCode=1;
