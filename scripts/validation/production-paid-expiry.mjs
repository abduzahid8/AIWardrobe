/** One disposable user and synthetic database subscription; actual store + SDK. No store purchase. */
import 'dotenv/config';
import { randomUUID, createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { transpileModule, ModuleKind, ScriptTarget } from 'typescript';
import { create } from 'zustand';
import { createClient } from '@supabase/supabase-js';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,key=process.env.SUPABASE_SERVICE_KEY;
if(!base||!anon||!key)throw Error('Fixture configuration required');
const options={auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false}};
const admin=createClient(base,key,options);
let owner,client,patches=0;
const report={measuredAt:new Date().toISOString(),scope:'One disposable user and synthetic subscription row; actual owner SDK and store. Paid status and expiry reconciliation, not an App Store purchase, renewal or refund.',checks:[],sourceHashes:{}};
const check=(name,passed)=>{report.checks.push({name,passed:!!passed});if(!passed)throw Error('Fixture assertion failed');};
try {
 const suffix=randomUUID(),email=`readiness-${suffix}@aiwardrobe.club`,password=`${randomUUID()}Aa!9`;
 const created=await admin.auth.admin.createUser({email,password,email_confirm:true,app_metadata:{readiness_audit:suffix},user_metadata:{username:`readiness_${suffix.replaceAll('-','').slice(0,20)}`,readiness_test:true}});
 if(created.error||!created.data.user)throw Error('Fixture creation failed');owner=created.data.user.id;
 client=createClient(base,anon,{...options,global:{fetch:async(input,init)=>{
  const url=new URL(typeof input==='string'?input:input instanceof URL?input.href:input.url);
  if(url.pathname.endsWith('/profiles')&&init?.method==='PATCH')patches++;
  return fetch(input,{...init,signal:init?.signal??AbortSignal.timeout(30000)});
 }}});
 if((await client.auth.signInWithPassword({email,password})).error)throw Error('Fixture login failed');
 const modules={},cache=new Map();
 async function load(file){
  const source=await readFile(new URL(`../../${file}`,import.meta.url),'utf8');
  report.sourceHashes[file]=createHash('sha256').update(source).digest('hex');
  const api={};new Function('exports','require',transpileModule(source,{compilerOptions:{module:ModuleKind.CommonJS,target:ScriptTarget.ES2022}}).outputText)(api,name=>{
   if(name==='zustand')return{create};
   if(name.endsWith('/authEvents'))return modules.authEvents;
   if(name.endsWith('/logger'))return{__esModule:true,default:{log(){},warn(){},error(){}}};
   if(name.endsWith('lib/supabase'))return{supabase:client};
   if(name==='@react-native-async-storage/async-storage')return{__esModule:true,default:{getItem:async k=>cache.get(k)??null,setItem:async(k,v)=>{cache.set(k,v);},removeItem:async k=>{cache.delete(k);}}};
   throw Error('Unsupported trial fixture dependency');
  });return api;
 }
 modules.authEvents=await load('store/authEvents.ts');modules.authEvents.authEvents.emitLogin(owner);
 let store=(await load('store/subscriptionStore.ts')).default;
 const before=await client.from('profiles').select('trial_started_at').eq('id',owner).single();
 check('new_fixture_has_no_trial',!before.error&&before.data.trial_started_at===null);
 await store.getState().verifySubscriptionFromServer();
 const first=await client.from('profiles').select('trial_started_at').eq('id',owner).single();
 const date=first.data?.trial_started_at;
 check('owner_initializer_confirms_server_date_and_trial_access',!first.error&&date&&store.getState().trialStartedAt===date&&store.getState().isTrialActive&&store.getState().effectiveTier==='premium'&&patches===1&&cache.get('trial_started_at_v1')===date);
 await store.getState().initializeTrial(owner);
 store=(await load('store/subscriptionStore.ts')).default;
 await store.getState().verifySubscriptionFromServer();
 check('repeated_initialization_and_fresh_store_do_not_restart_trial',patches===1&&store.getState().trialStartedAt===date&&store.getState().isTrialActive);
 for(const requested of [null,new Date(Date.now()+30*86400000).toISOString()]){
  const attempt=await client.from('profiles').update({trial_started_at:requested}).eq('id',owner).select('trial_started_at').single();
  check(requested===null?'owner_cannot_clear_trial':'owner_cannot_extend_trial',attempt.error?.code==='42501');
 }
 const unchanged=await client.from('profiles').select('trial_started_at').eq('id',owner).single();
 check('rejected_resets_leave_original_date_unchanged',!unchanged.error&&unchanged.data.trial_started_at===date);
 // A synthetic subscription verifies database/store expiry, not payment processing.
 const subscriptionId=randomUUID();
 const paid=await admin.from('subscriptions').insert({id:subscriptionId,user_id:owner,tier:'premium',status:'active',end_date:new Date(Date.now()+86400000).toISOString(),platform:'ios',revenuecat_managed:false});
 if(paid.error)throw Error('Fixture paid setup failed');
 await store.getState().verifySubscriptionFromServer();
 check('future_paid_record_grants_verified_paid_access',store.getState().tier==='premium'&&store.getState().effectiveTier==='premium'&&store.getState().hasActiveSubscription&&!store.getState().isTrialActive);
 const expiry=await admin.from('subscriptions').update({end_date:new Date(Date.now()-60000).toISOString()}).eq('id',subscriptionId).eq('user_id',owner);
 if(expiry.error)throw Error('Fixture paid expiry setup failed');
 await store.getState().verifySubscriptionFromServer();
 check('expired_paid_record_stops_paid_status_and_preserves_original_trial',store.getState().tier==='free'&&store.getState().isTrialActive&&store.getState().trialStartedAt===date&&store.getState().effectiveTier==='premium');
 // Admin expiry is confined to the disposable fixture to exercise real server reconciliation.
 const expired=new Date(Date.now()-8*86400000).toISOString();
 const updated=await admin.from('profiles').update({trial_started_at:expired}).eq('id',owner);
 if(updated.error)throw Error('Fixture expiry setup failed');
 await store.getState().verifySubscriptionFromServer();
 check('expired_fixture_reconciles_to_free_without_restarting',store.getState().isTrialExpired&&!store.getState().isTrialActive&&store.getState().effectiveTier==='free'&&!store.getState().checkFeatureAccess('aiOutfits')&&patches===3);
 check('expired_free_allowances_are_preserved',store.getState().getTriesRemaining('homeSwipes',0)===3&&store.getState().getTriesRemaining('wardrobeScans',0)===5);
}catch(error){report.failure=/^Fixture .* failed$/.test(error.message)?error.message:error.name;process.exitCode=1;}
finally{
 report.cleanup={};
 if(client)try{report.cleanup.refreshSessionsRevoked=!(await client.auth.signOut({scope:'global'})).error;}catch{report.cleanup.refreshSessionsRevoked=false;}
 if(owner)try{report.cleanup.userDeleted=!(await admin.auth.admin.deleteUser(owner)).error;}catch{report.cleanup.userDeleted=false;}
 report.passed=report.checks.length===10&&report.checks.every(x=>x.passed)&&report.cleanup.refreshSessionsRevoked&&report.cleanup.userDeleted&&!report.failure;
 if(!report.passed)process.exitCode=1;
 await writeFile(new URL('../../docs/reports/readiness-evidence/production-paid-expiry.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}
