/** Disposable free account only; no purchase, promotion, or customer-account access. */
import 'dotenv/config';
import {randomUUID} from 'node:crypto';
import {writeFile} from 'node:fs/promises';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,service=process.env.SUPABASE_SERVICE_KEY,rc=process.env.REVENUECAT_TEST_SERVER_KEY;
if(!service||!rc||new URL(base).hostname!=='fyqpifmrsftsfqibhwhy.supabase.co')throw Error('Expected authenticated setup');
const admin={apikey:service,Authorization:`Bearer ${service}`,'Content-Type':'application/json'};
async function call(url,headers,body,method='POST'){const r=await fetch(url,{method,headers,...(body===undefined?{}:{body:JSON.stringify(body)}),signal:AbortSignal.timeout(30000),redirect:'error'});let data;try{data=await r.json();}catch{data=null;}return {status:r.status,ok:r.ok,data};}
const api=(path,headers,body,method)=>call(new URL(path,base),headers,body,method);
const report={measuredAt:new Date().toISOString(),runId:randomUUID(),scope:'One disposable free account; no store purchase or paid access',checks:[],cleanup:[]};let owner;
const check=(name,passed,details={})=>report.checks.push({name,passed,...details});
try{
 const email=`readiness-${randomUUID()}@aiwardrobe.club`,password=`${randomUUID()}Aa!9`;
 const created=await api('/auth/v1/admin/users',admin,{email,password,email_confirm:true,app_metadata:{readiness_audit:report.runId}});if(!created.ok||!created.data?.id)throw Error('Fixture creation failed');owner=created.data.id;report.fixtureOwner=owner;
 const login=await api('/auth/v1/token?grant_type=password',{apikey:anon,'Content-Type':'application/json'},{email,password});if(!login.data?.access_token)throw Error('Fixture login failed');
 const headers={apikey:anon,Authorization:`Bearer ${login.data.access_token}`,'Content-Type':'application/json'};
 const anonymous=await api('/functions/v1/sync-subscription',{apikey:anon,Authorization:`Bearer ${anon}`,'Content-Type':'application/json'},{});check('anonymous_denied',anonymous.status===401,{status:anonymous.status});
 const direct=await call(`https://api.revenuecat.com/v1/subscribers/${owner}`,{Authorization:`Bearer ${rc}`},undefined,'GET');report.providerDiagnostic={status:direct.status,requestDateType:typeof direct.data?.request_date_ms,hasSubscriber:!!direct.data?.subscriber,entitlementsType:typeof direct.data?.subscriber?.entitlements};
 if(direct.ok&&Number.isSafeInteger(direct.data?.request_date_ms)&&direct.data?.subscriber?.entitlements&&Object.keys(direct.data.subscriber.entitlements).length===0){const rpc=await api('/rest/v1/rpc/apply_revenuecat_snapshots',admin,{p_event:{id:`diagnostic:${report.runId}`,type:'CUSTOMER_SYNC'},p_accounts:[{user_id:owner,request_date_ms:direct.data.request_date_ms,entitlements:[]}]});report.rpcDiagnostic={status:rpc.status,code:rpc.data?.code||null,message:rpc.data?.message||null};}
 const start=Date.now();const sync=await api('/functions/v1/sync-subscription',headers,{userId:randomUUID(),tier:'vip'});check('verified_free_cannot_self_grant',sync.ok&&sync.data?.verified===true&&sync.data?.tier==='free',{status:sync.status,latencyMs:Date.now()-start,verified:sync.data?.verified===true,tier:sync.data?.tier||null,error:sync.data?.error||null});
 const profile=await api(`/rest/v1/profiles?id=eq.${owner}&select=subscription_tier`,headers,undefined,'GET');check('profile_remains_free',profile.ok&&profile.data?.[0]?.subscription_tier==='free',{status:profile.status});
 const state=await api(`/rest/v1/revenuecat_sync_state?user_id=eq.${owner}&select=user_id,snapshot_ms`,admin,undefined,'GET');check('server_snapshot_records_only_owner',state.ok&&state.data?.length===1&&state.data[0].user_id===owner&&state.data[0].snapshot_ms>0,{status:state.status});
 const forbidden=await api('/rest/v1/rpc/apply_revenuecat_snapshots',headers,{p_event:{id:`forged:${report.runId}`},p_accounts:[]});check('client_cannot_call_authoritative_writer',[401,403].includes(forbidden.status),{status:forbidden.status});
 const offers=await call(`https://api.revenuecat.com/v1/subscribers/${owner}/offerings`,{Authorization:`Bearer ${process.env.EXPO_PUBLIC_REVENUECAT_API_KEY}`,'X-Platform':'ios'},undefined,'GET');
 const current=offers.data?.offerings?.find(o=>o.identifier===offers.data?.current_offering_id);const products=(current?.packages||[]).map(p=>p.platform_product_identifier).filter(Boolean).sort();
 const expected=['2.99','com.aiwardrobe.premium.monthly','com.aiwardrobe.premium.yearly'].sort();check('ios_offering_contains_three_approved_products',offers.ok&&JSON.stringify(products)===JSON.stringify(expected),{status:offers.status,products,currentOffering:offers.data?.current_offering_id||null});
}catch(error){report.failure=error.name;process.exitCode=1;}
finally{
 if(owner){
  const revenue=await call(`https://api.revenuecat.com/v1/subscribers/${owner}`,{Authorization:`Bearer ${rc}`},undefined,'DELETE');report.cleanup.push({system:'revenuecat',deletionRequested:[200,404].includes(revenue.status),status:revenue.status,note:'RevenueCat deletion is asynchronous; no GET used because it creates missing customers.'});
  const diagnosticLedger=await api(`/rest/v1/processed_webhooks?source=eq.revenuecat&event_id=eq.diagnostic:${report.runId}`,admin,undefined,'DELETE');report.cleanup.push({system:'diagnostic_ledger',deleted:diagnosticLedger.ok,status:diagnosticLedger.status});
  const ledger=await api(`/rest/v1/processed_webhooks?source=eq.revenuecat&event_id=like.customer-sync:${owner}:*`,admin,undefined,'DELETE');report.cleanup.push({system:'sync_ledger',deleted:ledger.ok,status:ledger.status});
  const user=await api(`/auth/v1/admin/users/${owner}`,admin,undefined,'DELETE');report.cleanup.push({system:'supabase_user',deleted:user.ok,status:user.status});
  if(![200,404].includes(revenue.status)||!ledger.ok||!user.ok)process.exitCode=1;
 }
 await writeFile(new URL('../../docs/reports/readiness-evidence/production-subscription-sync.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}
if(report.checks.length!==6||report.checks.some(c=>!c.passed))process.exitCode=1;
