/** Fixture-only paired local/Edge transport diagnostic. Never prints credentials. */
import 'dotenv/config';
import {createClient} from '@supabase/supabase-js';
import {execFileSync} from 'node:child_process';
import {readFile,writeFile,unlink} from 'node:fs/promises';
import {randomUUID,createHash} from 'node:crypto';
import {loadAppMediaForNode} from '../validation/app-media-node.mjs';
import {transpileModule,ModuleKind} from 'typescript';
import {measureRemoteMedia} from './remote-media-probe.mjs';
const project='fyqpifmrsftsfqibhwhy',cli=process.env.SUPABASE_CLI_BIN;
const ledgerPath=new URL('../../docs/reports/readiness-evidence/remote-media-probe-lifecycle.json',import.meta.url);
const privatePath='/tmp/aiwardrobe-remote-media-session.json',previewPath='/tmp/aiwardrobe-remote-media-control.png';
const reportPathFor=runId=>{if(!/^[a-f0-9-]{36}$/.test(runId))throw new Error('Invalid run ID');return new URL(`../../docs/reports/readiness-evidence/paired-media-diagnostic-${runId}.json`,import.meta.url);};
const options={auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false}};
const base=new URL(process.env.EXPO_PUBLIC_SUPABASE_URL).origin,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
if(base!==`https://${project}.supabase.co`||!cli||!anon)throw new Error('Diagnostic configuration required');
async function ledger(){return JSON.parse(await readFile(ledgerPath,'utf8'));}
async function saveLedger(value){await writeFile(ledgerPath,JSON.stringify(value,null,2)+'\n');}
function command(args){try{return execFileSync(cli,args,{encoding:'utf8',stdio:['ignore','pipe','pipe'],timeout:30000});}catch{throw new Error('ManagementCommandFailed');}}
function adminClient(){
 let data;try{data=JSON.parse(command(['projects','api-keys','--project-ref',project,'--reveal','--output','json','--agent','yes']));}catch{throw new Error('FixtureCredentialUnavailable');}let key;
 function visit(value){if(!value||typeof value!=='object')return;if(typeof value.api_key==='string'){
  try{const claims=JSON.parse(Buffer.from(value.api_key.split('.')[1],'base64url').toString());if(claims.ref===project&&claims.role==='service_role')key=value.api_key;}catch{}
 }for(const child of Object.values(value))if(child&&typeof child==='object')visit(child);}
 visit(data);if(!key)throw new Error('FixtureCredentialUnavailable');return createClient(base,key,options);
}
async function savePrivate(value,initial=false){await writeFile(privatePath,JSON.stringify(value),{mode:0o600,flag:initial?'wx':'w'});}
async function cleanup(state,admin,deleteFunction){
 const checks=[];
 if(deleteFunction){try{command(['functions','delete',state.slug,'--project-ref',project,'--yes','--agent','yes']);checks.push({functionDeleteCommandSucceeded:true});}catch{checks.push({functionDeleteCommandSucceeded:false});}}
 try{
  const owner=state.users[0],found=await admin.auth.admin.getUserById(owner.id);
  if(!found.error&&found.data.user?.email===owner.email&&found.data.user?.app_metadata?.readiness_audit===state.runId){
   const result=await admin.storage.from('wardrobe-images').remove([`${owner.id}/${state.objectId}/thumbnail.png`]);checks.push({objectsRemoved:!result.error});
  }else if(found.error?.status===404){
   const remaining=await admin.storage.from('wardrobe-images').list(`${owner.id}/${state.objectId}`);checks.push({objectsRemoved:!remaining.error&&remaining.data.length===0});
  }else{checks.push({objectsRemoved:false,ownershipNotConfirmed:true});}
 }catch{checks.push({objectsRemoved:false});}
 for(const user of state.users){
  try{
   const found=await admin.auth.admin.getUserById(user.id);
   if(found.error?.status===404){checks.push({userId:user.id,accountDeleted:true});continue;}
   if(found.error||found.data.user?.email!==user.email||found.data.user?.app_metadata?.readiness_audit!==state.runId){checks.push({userId:user.id,accountDeleted:false,ownershipNotConfirmed:true});continue;}
   const deleted=await admin.auth.admin.deleteUser(user.id);checks.push({userId:user.id,accountDeleted:!deleted.error});
  }catch{checks.push({userId:user.id,accountDeleted:false});}
 }
 const passed=checks.every(check=>check.functionDeleteCommandSucceeded??check.objectsRemoved??check.accountDeleted);
 const log=await ledger();log.cleanup=checks;log.status=passed?'cleanup_commands_succeeded':'cleanup_requires_verification';await saveLedger(log);
 if(passed){await unlink(privatePath).catch(()=>undefined);await unlink(previewPath).catch(()=>undefined);}
 return {checks,passed};
}
const mode=process.argv[2];
if(mode==='prepare'){
 const log=await ledger();if(log.status!=='placeholder_absence_verified'||log.activeProbeDeploymentAttempted||!log.placeholderCleanupVerified||!/^readiness-media-probe-[a-z0-9]+-[a-z0-9]+$/.test(log.slug)||log.project!==project)throw new Error('Invalid fixture ledger');
 const state={slug:log.slug,project,runId:randomUUID(),objectId:randomUUID(),users:Array.from({length:2},()=>({id:randomUUID(),email:`readiness-${randomUUID()}@aiwardrobe.club`}))};
 await savePrivate(state,true);let admin;
 try{
  admin=adminClient();log.runId=state.runId;log.fixtureUserIds=state.users.map(user=>user.id);log.status='creating_fixtures';log.placeholderCleanupVerified=true;await saveLedger(log);
  const python=`from PIL import Image\nImage.open('assets/images/basic_outfit_1.png').convert('RGBA').resize((384,512),Image.Resampling.LANCZOS).save('${previewPath}',compress_level=0)\n`;
  execFileSync('python3',['-c',python],{stdio:['ignore','pipe','pipe']});const image=await readFile(previewPath);
  const clients=[];
  for(const user of state.users){
   const password=randomUUID()+'Aa!9';
   const result=await admin.auth.admin.createUser({id:user.id,email:user.email,password,email_confirm:true,user_metadata:{readiness_test:true,username:`readiness_${user.id}`},app_metadata:{readiness_audit:state.runId}});
   if(result.error||result.data.user?.id!==user.id)throw new Error('FixtureCreationFailed');
   const client=createClient(base,anon,options),login=await client.auth.signInWithPassword({email:user.email,password});
   if(login.error||!login.data.session)throw new Error('FixtureLoginFailed');user.token=login.data.session.access_token;clients.push(client);await savePrivate(state);
  }
  const events={};new Function('exports',transpileModule(await readFile(new URL('../../store/authEvents.ts',import.meta.url),'utf8'),{compilerOptions:{module:ModuleKind.CommonJS}}).outputText)(events);
  events.authEvents.emitLogin(state.users[0].id);const app=await loadAppMediaForNode(clients[0],events,base);
  await app.media.uploadWardrobeImage(state.users[0].id,state.objectId,`data:image/png;base64,${image.toString('base64')}`,'image/png','thumbnail');
  state.config={projectUrl:base,owner:state.users[0].id,objectId:state.objectId,bytes:image.length,sha256:createHash('sha256').update(image).digest('hex'),expiresAt:Date.now()+300000};
  await savePrivate(state);log.probeConfig=state.config;log.mediaJavaScriptHashes=app.hashes;log.fixtureDescription='Pillow RGBA 384x512 PNG, compression=0; matched network control, not native codec output';log.status='fixtures_ready';await saveLedger(log);
  console.log(JSON.stringify({prepared:true,fixtureBytes:image.length,fixtureUserIds:log.fixtureUserIds,slug:state.slug}));
 }catch(error){log.failure=error.name;await saveLedger(log);if(admin)await cleanup(state,admin,false);console.log(JSON.stringify({prepared:false,failure:error.name}));process.exitCode=1;}
}else if(mode==='measure'||mode==='cleanup'){
 const state=JSON.parse(await readFile(privatePath,'utf8'));if(state.project!==project||!/^readiness-media-probe-[a-z0-9]+-[a-z0-9]+$/.test(state.slug))throw new Error('Invalid fixture state');
 const admin=adminClient(),report={startedAt:new Date().toISOString(),scope:'Matched local-vs-same-project-Edge private image delivery; not complete capacity certification',fixtureBytes:state.config?.bytes,config:state.config,controls:[]};
 try{
  if(mode==='measure'){
   const endpoint=`${base}/functions/v1/${state.slug}`;
   const call=token=>fetch(endpoint,{method:'POST',headers:{apikey:anon,...(token?{Authorization:`Bearer ${token}`}:{})},signal:AbortSignal.timeout(120000)});
   for(const [name,token,expected] of [['anonymous_denied',null,401],['other_fixture_owner_denied',state.users[1].token,403]]){
    const response=await call(token);await response.arrayBuffer();const passed=response.status===expected;report.controls.push({name,status:response.status,passed});if(!passed)throw new Error('AuthorizationControlFailed');
   }
   const response=await call(state.users[0].token);report.remoteHttpStatus=response.status;
   if(!response.ok)throw new Error('RemoteProbeFailed');report.remote=await response.json();
   report.local=await measureRemoteMedia({...state.config,region:'local-test-machine'},`Bearer ${state.users[0].token}`);
  }
 }catch(error){report.failure=error.name;process.exitCode=1;}
 finally{report.cleanup=await cleanup(state,admin,true);report.finishedAt=new Date().toISOString();await writeFile(reportPathFor(state.runId),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({remote:report.remote,local:report.local,controls:report.controls,cleanup:report.cleanup,failure:report.failure}));if(!report.cleanup.passed)process.exitCode=1;}
}else{throw new Error('Use prepare, measure or cleanup');}
