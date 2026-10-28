/** Service-only coordination checks. Never publishes a rate snapshot. */
import 'dotenv/config';import {randomUUID} from 'node:crypto';import {writeFile} from 'node:fs/promises';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,service=process.env.SUPABASE_SERVICE_KEY,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;if(!service)throw Error('Setup credential unavailable');
const call=async(name,key,body)=>{const response=await fetch(new URL('/rest/v1/rpc/'+name,base),{method:'POST',headers:{apikey:key,Authorization:'Bearer '+key,'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(10000)});return{status:response.status,data:await response.json()}};
const report={measuredAt:new Date().toISOString(),scope:'Live service-only lease RPC concurrency and fencing; no rate snapshot publication or CBU call.',checks:[]};
const check=(name,passed)=>{report.checks.push({name,passed:!!passed});if(!passed)throw Error('Lease audit assertion failed')};
try{
 const denied=await call('claim_budget_rate_refresh',anon,{p_token:randomUUID()});check('anonymous_claim_denied',denied.status===401||denied.status===403);
 const tokens=Array.from({length:20},()=>randomUUID());const claims=await Promise.all(tokens.map(p_token=>call('claim_budget_rate_refresh',service,{p_token})));
 report.claimStatuses=claims.map(r=>r.status);report.claimResults=claims.map(r=>r.data);
 const owners=claims.map((r,i)=>r.status===200&&r.data==='acquired'?i:-1).filter(i=>i>=0);
 check('one_owner_across_twenty_requests',owners.length===1&&claims.filter(r=>r.data==='busy').length===19);
 const rejected=await call('finish_budget_rate_refresh',service,{p_token:randomUUID(),p_snapshot:{}});check('wrong_owner_cannot_publish',rejected.status===200&&rejected.data===false);
 const winner=tokens[owners[0]];const released=await call('finish_budget_rate_refresh',service,{p_token:winner,p_snapshot:null});check('owner_can_record_failed_refresh',released.status===200&&released.data===true);
 const late=await call('finish_budget_rate_refresh',service,{p_token:winner,p_snapshot:{}});check('released_owner_cannot_publish_late',late.status===200&&late.data===false);
 const cooldown=await call('claim_budget_rate_refresh',service,{p_token:randomUUID()});check('failure_cooldown_prevents_upstream_retry_storm',cooldown.status===200&&cooldown.data==='cooldown');
}catch(e){report.failure=e.name;process.exitCode=1}
finally{report.passed=report.checks.length===6&&report.checks.every(c=>c.passed)&&!report.failure;await writeFile('docs/reports/readiness-evidence/oct25-live-budget-refresh-lease.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));if(!report.passed)process.exitCode=1}
