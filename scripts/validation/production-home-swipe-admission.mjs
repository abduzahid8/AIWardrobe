/** Actual quota store against live RPCs; controlled storage/plan, one disposable free account. */
import 'dotenv/config';import{createClient}from'@supabase/supabase-js';import{randomUUID,createHash}from'node:crypto';import{readFile,writeFile}from'node:fs/promises';import{createRequire}from'node:module';import ts from'typescript';
const require=createRequire(import.meta.url),base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,key=process.env.SUPABASE_SERVICE_KEY;if(!key)throw Error('Setup credential required');const options={auth:{persistSession:false,autoRefreshToken:false}},admin=createClient(base,key,options),client=createClient(base,anon,options);
const report={measuredAt:new Date().toISOString(),scope:'Twenty concurrent actual dailyUsageStore calls through live ordinary-account quota RPCs. AsyncStorage and free-plan selectors controlled; not native Home UI, provider work or10kDAUproof.',checks:[]};let owner;
const check=(name,passed)=>{report.checks.push({name,passed:!!passed});if(!passed)throw Error('Audit assertion failed')};
try{
 const suffix=randomUUID(),email=`readiness-${suffix}@aiwardrobe.club`,password=randomUUID()+'Aa!9';const created=await admin.auth.admin.createUser({email,password,email_confirm:true,app_metadata:{readiness_audit:suffix},user_metadata:{username:'readiness_'+suffix.replaceAll('-','').slice(0,20),readiness_test:true}});if(created.error)throw Error('Fixture creation failed');owner=created.data.user.id;report.fixtureUserId=owner;
 if((await client.auth.signInWithPassword({email,password})).error)throw Error('Fixture login failed');
 const profile=await admin.from('profiles').select('trial_started_at,subscription_tier').eq('id',owner).single();check('fixture_is_free_without_trial',!profile.error&&profile.data.subscription_tier==='free'&&profile.data.trial_started_at===null);
 const compile=source=>ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
 const deadlineSource=await readFile('src/lib/requestDeadline.ts','utf8'),deadline={};new Function('exports',compile(deadlineSource))(deadline);
 const source=await readFile('store/dailyUsageStore.ts','utf8'),api={},disk=new Map();report.sourceSha256=createHash('sha256').update(source).digest('hex');
 const fakeSubscription={refreshTimeState:()=>{},effectiveTier:'free'};
 new Function('require','exports',compile(source))(name=>{
  if(name==='zustand')return require(name);
  if(name.includes('async-storage'))return{__esModule:true,default:{setItem:async(k,v)=>disk.set(k,v),getItem:async k=>disk.get(k)??null}};
  if(name.includes('subscriptionStore'))return{__esModule:true,default:{getState:()=>fakeSubscription},TIER_FEATURES:{free:{homeSwipes:3}}};
  if(name.includes('lib/supabase'))return{supabase:client};
  if(name.includes('authEvents'))return{authEvents:{getUserId:()=>owner,getRevision:()=>1}};
  if(name.includes('requestDeadline'))return deadline;
  throw Error('Unexpected store dependency');
 },api);
 const store=api.default;let active=0,maxActive=0;const started=performance.now();
 const outcomes=await Promise.all(Array.from({length:20},async()=>{active++;maxActive=Math.max(maxActive,active);try{return{...await store.getState().consume('homeSwipes',1,{requireServer:true}),error:null}}catch(e){return{allowed:false,error:e.name}}finally{active--}}));
 report.requests={maxActive,elapsedMs:Math.round(performance.now()-started),allowed:outcomes.filter(o=>o.allowed).length,denied:outcomes.filter(o=>!o.allowed&&!o.error).length,errors:outcomes.filter(o=>o.error).length,errorTypes:[...new Set(outcomes.map(o=>o.error).filter(Boolean))]};
 check('exactly_three_of_twenty_reservations_allowed',report.requests.allowed===3&&report.requests.denied===17&&report.requests.errors===0);
 const authoritative=await client.rpc('get_usage_today',{p_feature:'homeSwipes'});report.serverUsed=authoritative.data;report.cachedUsed=store.getState().getUsed('homeSwipes');report.cachedRemaining=store.getState().getRemaining('homeSwipes');
 check('server_counter_is_three',!authoritative.error&&authoritative.data===3);check('cache_matches_server_after_out_of_order_responses',report.cachedUsed===3&&report.cachedRemaining===0);
}catch(e){report.failure=e.name;process.exitCode=1}
finally{report.cleanup={};try{report.cleanup.sessionsRevoked=!(await client.auth.signOut({scope:'global'})).error}catch{report.cleanup.sessionsRevoked=false}if(owner)try{report.cleanup.ownerDeleted=!(await admin.auth.admin.deleteUser(owner)).error}catch{report.cleanup.ownerDeleted=false}report.passed=report.checks.length===4&&report.checks.every(c=>c.passed)&&report.cleanup.sessionsRevoked&&report.cleanup.ownerDeleted&&!report.failure;await writeFile('docs/reports/readiness-evidence/oct25-home-swipe-admission.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));if(!report.passed)process.exitCode=1}
