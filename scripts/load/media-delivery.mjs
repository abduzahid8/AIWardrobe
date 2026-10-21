/** Bounded production diagnostic: disposable owners, private original/preview reads. */
import 'dotenv/config';
import {createClient} from '@supabase/supabase-js';
import {readFile,writeFile} from 'node:fs/promises';
import {randomUUID,createHash} from 'node:crypto';
import {performance} from 'node:perf_hooks';
import {loadAppMediaForNode} from '../validation/app-media-node.mjs';
import {transpileModule,ModuleKind} from 'typescript';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,key=process.env.SUPABASE_SERVICE_KEY;
if(!base||!anon||!key)throw new Error('Setup credential required');
const original=await readFile(new URL('../../assets/images/basic_outfit_1.png',import.meta.url));
const preview=await readFile('/tmp/aiwardrobe-native-thumbnail-artifacts/0-thumbnail.png');
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
const options={auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false}};
const admin=createClient(base,key,options),users=[],paths=[];
const report={startedAt:new Date().toISOString(),scope:'Bounded authenticated production Storage diagnostic, not complete 10000 DAU certification',originalBytes:original.length,previewBytes:preview.length,stages:[],checks:[],cleanup:[],fixtureUserIds:[],status:'running'};
const output=new URL(`../../docs/reports/readiness-evidence/production-media-delivery-${report.startedAt.replace(/[:.]/g,'-')}.json`,import.meta.url);
const save=()=>writeFile(output,JSON.stringify(report,null,2)+'\n');
const events={};
new Function('exports',transpileModule(await readFile(new URL('../../store/authEvents.ts',import.meta.url),'utf8'),{compilerOptions:{module:ModuleKind.CommonJS}}).outputText)(events);
const check=(name,passed)=>{report.checks.push({name,passed:Boolean(passed)});if(!passed)throw new Error('CheckFailed');};
const percentile=(values,p)=>values.length?[...values].sort((a,b)=>a-b)[Math.max(0,Math.ceil(values.length*p)-1)]:null;
async function request(user,variant){
 const started=performance.now();let headersMs=null,status=null,requestId=null;
 try{
  const response=await fetch(user[variant],{headers:{Authorization:`Bearer ${user.token}`},signal:AbortSignal.timeout(15000)});
  headersMs=performance.now()-started;status=response.status;requestId=response.headers.get('sb-request-id');
  const bytes=Buffer.from(await response.arrayBuffer());
  return {ok:response.ok&&digest(bytes)===digest(variant==='original'?original:preview),status,requestId,headersMs,bodyMs:performance.now()-started-headersMs,ms:performance.now()-started,bytes:bytes.length};
 }catch(error){return {ok:false,error:error.name,status,requestId,headersMs,ms:performance.now()-started,bytes:0};}
}
try{
 for(let index=0;index<2;index++){
  const email=`readiness-${randomUUID()}@aiwardrobe.club`,password=`${randomUUID()}Aa!9`;
  const created=await admin.auth.admin.createUser({email,password,email_confirm:true,user_metadata:{readiness_test:true}});
  if(created.error||!created.data.user)throw new Error('FixtureSetupFailed');
  const user={id:created.data.user.id,client:createClient(base,anon,options)};users.push(user);report.fixtureUserIds.push(user.id);await save();
  const login=await user.client.auth.signInWithPassword({email,password});
  if(login.error||!login.data.session)throw new Error('FixtureLoginFailed');
  user.token=login.data.session.access_token;
  const app=await loadAppMediaForNode(user.client,events,base);events.authEvents.emitLogin(user.id);
  const id=randomUUID();paths.push(`${user.id}/${id}/original.png`,`${user.id}/${id}/thumbnail.png`);
  user.original=await app.media.uploadWardrobeImage(user.id,id,`data:image/png;base64,${original.toString('base64')}`);
  user.preview=await app.media.uploadWardrobeImage(user.id,id,`data:image/png;base64,${preview.toString('base64')}`,undefined,'thumbnail');
 }
 const forbidden=await fetch(users[0].preview,{headers:{Authorization:`Bearer ${users[1].token}`},signal:AbortSignal.timeout(15000)});
 await forbidden.arrayBuffer();check('foreign_owner_read_rejected',!forbidden.ok);
 const anonymous=await fetch(users[0].preview,{signal:AbortSignal.timeout(15000)});await anonymous.arrayBuffer();check('anonymous_read_rejected',!anonymous.ok);
 const originals=[];for(let index=0;index<6;index++)originals.push(await request(users[index%2],'original'));
 report.originalReads={count:originals.length,failures:originals.filter(r=>!r.ok).length,p95Ms:percentile(originals.map(r=>r.ms),.95),samples:originals,bytes:originals.reduce((sum,r)=>sum+r.bytes,0)};
 check('all_original_bytes_match',originals.every(r=>r.ok));
 for(const rps of [2,5,10]){
  const seconds=10,planned=rps*seconds,results=[],active=new Set();let dropped=0;
  const start=performance.now();
  for(let index=0;index<planned;index++){
   const delay=start+index*1000/rps-performance.now();if(delay>0)await new Promise(resolve=>setTimeout(resolve,delay));
   if(active.size>=20){dropped++;continue;}
   const job=request(users[index%2],'preview').then(result=>results.push(result));active.add(job);job.finally(()=>active.delete(job));
  }
  await Promise.all(active);
  const elapsedMs=performance.now()-start;
  const stage={targetRps:rps,seconds,planned,completed:results.length,dropped,failures:results.filter(r=>!r.ok).length,elapsedMs,p95Ms:percentile(results.map(r=>r.ms),.95),p99Ms:percentile(results.map(r=>r.ms),.99),p95HeadersMs:percentile(results.map(r=>r.headersMs).filter(ms=>ms!==null),.95),p95BodyMs:percentile(results.filter(r=>r.ok).map(r=>r.bodyMs),.95),bytes:results.reduce((sum,r)=>sum+r.bytes,0),samples:results,errors:results.filter(r=>!r.ok)};
  report.stages.push(stage);await save();process.stdout.write(JSON.stringify(stage)+'\n');
  if(stage.failures||dropped)break;
 }
 check('all_planned_preview_stages_completed',report.stages.length===3&&report.stages.every(stage=>stage.completed===stage.planned&&!stage.failures&&!stage.dropped));
}catch(error){report.failure=error.name;process.exitCode=1;}
finally{
 if(paths.length){try{const result=await admin.storage.from('wardrobe-images').remove(paths);report.cleanup.push({objectsRemoved:!result.error});}catch{report.cleanup.push({objectsRemoved:false});}}
 for(const user of users){
  try{await user.client.auth.signOut();}catch{}
  try{const result=await admin.auth.admin.deleteUser(user.id);report.cleanup.push({userId:user.id,accountDeleted:!result.error});}catch{report.cleanup.push({userId:user.id,accountDeleted:false});}
 }
 report.finishedAt=new Date().toISOString();report.status='finished';
 report.passed=!report.failure&&report.checks.every(c=>c.passed)&&report.checks.length===4&&report.cleanup.length===3&&report.cleanup.every(c=>c.accountDeleted??c.objectsRemoved);
 if(!report.passed)process.exitCode=1;await save();process.stdout.write(JSON.stringify({passed:report.passed,cleanup:report.cleanup})+'\n');
}
