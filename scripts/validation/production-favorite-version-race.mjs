/** Actual private fixture writes + held transport + injected failed readback; queue disk roundtrip. */
import 'dotenv/config';
import {randomUUID} from 'node:crypto';
import {mkdtemp,writeFile,readFile,unlink,rmdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {createClient} from '@supabase/supabase-js';
import {loadWardrobeSync} from './load-wardrobe-sync.mjs';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,service=process.env.SUPABASE_SERVICE_KEY;
if(!base||!anon||!service)throw Error('Fixture configuration required');
const admin={apikey:service,Authorization:`Bearer ${service}`,'Content-Type':'application/json'};
const report={measuredAt:new Date().toISOString(),scope:'One private fixture garment; real SQL conditional writes, locally delayed older PATCH and injected readback503; serialized queue replay, not native process-crash or capacity certification',checks:[]};
const directory=await mkdtemp(path.join(tmpdir(),'aiwardrobe-favorite-race-')),oldFile=path.join(directory,'old.json'),newFile=path.join(directory,'new.json');
const call=async(endpoint,body,method='POST')=>{const response=await fetch(new URL(endpoint,base),{method,headers:admin,...body===undefined?{}:{body:JSON.stringify(body)},signal:AbortSignal.timeout(30000)});let data;try{data=await response.json();}catch{}return{ok:response.ok,status:response.status,data};};
let owner,client,oldPending,releaseOld=()=>{},failReadback=false;
try{
 const suffix=randomUUID(),email=`readiness-${suffix}@aiwardrobe.club`,password=`${randomUUID()}Aa!9`;
 const created=await call('/auth/v1/admin/users',{email,password,email_confirm:true,app_metadata:{readiness_audit:suffix},user_metadata:{username:`readiness_${suffix.replaceAll('-','').slice(0,20)}`,readiness_test:true}});
 if(!created.ok||!created.data?.id)throw Error('Fixture creation failed');owner=created.data.id;
 let reached;const ready=new Promise(resolve=>{reached=resolve;}),held=new Promise(resolve=>{releaseOld=resolve;});
 let patchCount=0,injected=0,oldCheckpoints=0,newCheckpoints=0;const responses=[];
 client=createClient(base,anon,{auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},global:{fetch:async(input,init)=>{
  const url=new URL(typeof input==='string'?input:input instanceof URL?input.href:input.url),method=init?.method||'GET';
  if(url.pathname.endsWith('/clothing_items')&&method==='GET'&&failReadback){injected++;return new Response(JSON.stringify({code:'INJECTED_READBACK_FAILURE',message:'Fixture transport readback failure'}),{status:503,headers:{'content-type':'application/json'}});}
  let index;
  if(url.pathname.endsWith('/clothing_items')&&method==='PATCH'){
   index=++patchCount;
   if(index===1){const saved=JSON.parse(await readFile(oldFile,'utf8'));report.checks.push({name:'old_version_checkpoint_precedes_patch',passed:saved[0]?.payload?.favoriteVersion!==undefined});reached();await held;}
  }
  const response=await fetch(input,{...init,signal:init?.signal??AbortSignal.timeout(30000)});
  if(index){responses.push({patch:index,status:response.status});if(index===1&&response.status===406)failReadback=true;}
  return response;
 }}});
 const login=await client.auth.signInWithPassword({email,password});if(login.error)throw Error('Fixture login failed');
 const id=randomUUID(),inserted=await client.from('clothing_items').insert({id,user_id:owner,category:'top',name:'Readiness favorite race fixture',image_url:'https://example.invalid/favorite-fixture.jpg',is_favorite:false});if(inserted.error)throw Error('Fixture garment setup failed');
 const runtime=await loadWardrobeSync(client,base,owner);report.sourceHashes=runtime.hashes;
 const initial=(await runtime.fetchItemsFromServer())?.find(item=>item.id===id);if(!initial)throw Error('Fixture initial read failed');
 const action=(actionId,isFavorite)=>({id:actionId,type:'update_item',payload:{itemId:id,isFavorite},createdAt:new Date().toISOString()});
 const oldCheckpoint=async actions=>{await writeFile(oldFile,JSON.stringify(actions),{mode:0o600});oldCheckpoints++;};
 const newCheckpoint=async actions=>{await writeFile(newFile,JSON.stringify(actions),{mode:0o600});newCheckpoints++;};
 oldPending=runtime.processPendingActions([action('older',true)],[{...initial,isFavorite:true}],owner,oldCheckpoint);await ready;
 const newer=await runtime.processPendingActions([action('newer',false)],[initial],owner,newCheckpoint);
 report.checks.push({name:'newer_same_value_write_confirmed_and_version_advanced',passed:newer.processedIds.includes('newer')&&newer.updatedItems[0]?.isFavorite===false&&newer.updatedItems[0]?.updatedAt!==initial.updatedAt});
 releaseOld();const older=await oldPending;
 report.checks.push({name:'older_collision_retains_persisted_version_after_injected_readback_failure',passed:older.processedIds.length===0&&older.updatedActions?.[0]?.payload.favoriteVersion===initial.updatedAt&&injected>0});
 failReadback=false;const saved=JSON.parse(await readFile(oldFile,'utf8'));
 const retried=await runtime.processPendingActions(saved,[{...initial,isFavorite:true}],owner,oldCheckpoint);
 const final=await client.from('clothing_items').select('id,is_favorite,updated_at').eq('id',id).eq('user_id',owner).single();
 report.checks.push({name:'serialized_retry_adopts_server_without_another_patch',passed:retried.processedIds.includes('older')&&retried.updatedItems[0]?.isFavorite===false&&patchCount===2&&oldCheckpoints===1&&newCheckpoints===1});
 report.checks.push({name:'final_database_favorite_stays_newer_value',passed:!final.error&&final.data?.is_favorite===false});
 report.patchResponses=responses;report.injectedReadbackFailures=injected;
}catch(error){report.failure=/^Fixture .* failed$/.test(error.message)?error.message:error.name;process.exitCode=1;}
finally{
 failReadback=false;releaseOld();if(oldPending)await oldPending.catch(()=>undefined);report.cleanup={};
 if(client){try{report.cleanup.refreshSessionsRevoked=!(await client.auth.signOut({scope:'global'})).error;}catch{report.cleanup.refreshSessionsRevoked=false;}}
 if(owner){try{report.cleanup.userDeleted=(await call(`/auth/v1/admin/users/${owner}`,undefined,'DELETE')).ok;}catch{report.cleanup.userDeleted=false;}}
 for(const file of [oldFile,newFile])try{await unlink(file);}catch(error){if(error.code!=='ENOENT')report.cleanup.fileError=true;}
 try{await rmdir(directory);report.cleanup.queueFilesRemoved=true;}catch{report.cleanup.queueFilesRemoved=false;}
 report.passed=report.checks.length===5&&report.checks.every(check=>check.passed)&&report.cleanup.userDeleted===true&&report.cleanup.refreshSessionsRevoked===true&&report.cleanup.queueFilesRemoved===true&&!report.failure;
 if(!report.passed)process.exitCode=1;
 await writeFile(new URL('../../docs/reports/readiness-evidence/production-favorite-version-race.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}
