/** Bounded fixture-only comparison. Credentials remain in memory or curl stdin. */
import 'dotenv/config';
import {createClient} from '@supabase/supabase-js';
import {execFileSync,spawn} from 'node:child_process';
import {readFile,writeFile,mkdtemp,rm} from 'node:fs/promises';
import {randomUUID,createHash} from 'node:crypto';
import {performance} from 'node:perf_hooks';
const project='fyqpifmrsftsfqibhwhy',base=`https://${project}.supabase.co`,cli=process.env.SUPABASE_CLI_BIN;
const plan={scope:'Same Mac and object, Node fetch vs fresh-process curl HTTP/1.1 and HTTP/2; not capacity certification',rps:5,secondsPerClient:10,requestsPerClient:50,concurrencyCap:20,requestTimeoutMs:15000,clients:['node-fetch','curl-http1.1','curl-http2'],fixtureUsers:1,fixtureObjects:1};
if(process.argv[2]==='--plan'){console.log(JSON.stringify(plan));process.exit(0);}
if(!['--run','--cleanup'].includes(process.argv[2])||!cli||process.env.EXPO_PUBLIC_SUPABASE_URL?.replace(/\/$/,'')!==base)throw new Error('Explicit diagnostic configuration required');
function command(args){try{return execFileSync(cli,args,{encoding:'utf8',stdio:['ignore','pipe','pipe'],timeout:30000});}catch{throw new Error('ManagementCommandFailed');}}
let key;try{const data=JSON.parse(command(['projects','api-keys','--project-ref',project,'--reveal','--output','json','--agent','yes']));
 const visit=v=>{if(!v||typeof v!=='object')return;if(typeof v.api_key==='string'){try{const c=JSON.parse(Buffer.from(v.api_key.split('.')[1],'base64url'));if(c.ref===project&&c.role==='service_role')key=v.api_key;}catch{}}for(const x of Object.values(v))if(x&&typeof x==='object')visit(x);};visit(data);
}catch{throw new Error('FixtureCredentialUnavailable');}if(!key)throw new Error('FixtureCredentialUnavailable');
const options={auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},global:{fetch:(input,init)=>fetch(input,{...init,signal:AbortSignal.timeout(20000)})}};
const admin=createClient(base,key,options);key=undefined;
const sum=bytes=>createHash('sha256').update(bytes).digest('hex');
let report,output;
if(process.argv[2]==='--cleanup'){
 if(!/^[a-f0-9-]{36}$/.test(process.argv[3]||''))throw new Error('Run ID required');
 output=new URL(`../../docs/reports/readiness-evidence/media-transport-${process.argv[3]}.json`,import.meta.url);
 report=JSON.parse(await readFile(output,'utf8'));
 if(report.project!==project||report.runId!==process.argv[3])throw new Error('Invalid fixture ledger');
}else{
 const runId=randomUUID(),owner=randomUUID(),objectId=randomUUID();
 report={...plan,project,runId,owner,objectId,email:`readiness-${runId}@aiwardrobe.club`,startedAt:new Date().toISOString(),status:'preparing',stages:[]};
 output=new URL(`../../docs/reports/readiness-evidence/media-transport-${runId}.json`,import.meta.url);
 await writeFile(output,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
}
const save=()=>writeFile(output,JSON.stringify(report,null,2)+'\n');
const objectPath=`${report.owner}/${report.objectId}/thumbnail.png`;
async function cleanup(){
 const found=await admin.auth.admin.getUserById(report.owner);
 if(found.error?.status===404){report.cleanup={userAbsent:true,objectsAbsent:false};
  const list=await admin.storage.from('wardrobe-images').list(`${report.owner}/${report.objectId}`);report.cleanup.objectsAbsent=!list.error&&!list.data.length;
 }else if(found.error||found.data.user?.email!==report.email||found.data.user?.app_metadata?.readiness_audit!==report.runId){report.cleanup={ownershipConfirmed:false};}
 else{
  const removed=await admin.storage.from('wardrobe-images').remove([objectPath]);
  const objects=await admin.storage.from('wardrobe-images').list(`${report.owner}/${report.objectId}`);
  const deleted=await admin.auth.admin.deleteUser(report.owner);
  const absent=await admin.auth.admin.getUserById(report.owner);
  report.cleanup={ownershipConfirmed:true,removeSucceeded:!removed.error,objectsAbsent:!objects.error&&!objects.data.length,deleteSucceeded:!deleted.error,userAbsent:absent.error?.status===404};
 }
 report.cleanup.passed=Object.values(report.cleanup).every(Boolean);await save();
}
let temp;
try{
 if(process.argv[2]==='--run'){
  temp=await mkdtemp('/tmp/aiwardrobe-transport-');const imagePath=temp+'/fixture.png';
  execFileSync('python3',['-c',"from PIL import Image\nimport sys\nImage.open('assets/images/basic_outfit_1.png').convert('RGBA').resize((384,512),Image.Resampling.LANCZOS).save(sys.argv[1],compress_level=0)",imagePath],{stdio:['ignore','pipe','pipe']});
  const image=await readFile(imagePath);report.fixtureBytes=image.length;report.fixtureSha256=sum(image);report.nodeVersion=process.version;report.curlVersion=execFileSync('/usr/bin/curl',['--version'],{encoding:'utf8'}).split('\n')[0];await save();
  const password=randomUUID()+'Aa!9';
  const created=await admin.auth.admin.createUser({id:report.owner,email:report.email,password,email_confirm:true,user_metadata:{readiness_test:true},app_metadata:{readiness_audit:report.runId}});
  if(created.error||created.data.user?.id!==report.owner)throw new Error('FixtureCreateFailed');
  const client=createClient(base,process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,options),login=await client.auth.signInWithPassword({email:report.email,password});
  if(login.error||!login.data.session)throw new Error('FixtureLoginFailed');const token=login.data.session.access_token;
  const upload=await client.storage.from('wardrobe-images').upload(objectPath,image,{contentType:'image/png'});if(upload.error)throw new Error('FixtureUploadFailed');
  const url=`${base}/storage/v1/object/authenticated/wardrobe-images/${objectPath}`;report.status='measuring';await save();
  async function nodeRead(){const start=performance.now();let headersMs=null,status=null;try{
   const response=await fetch(url,{headers:{Authorization:`Bearer ${token}`},signal:AbortSignal.timeout(15000)});headersMs=performance.now()-start;status=response.status;
   const reader=response.body.getReader(),parts=[];let length=0;for(;;){const {done,value}=await reader.read();if(done)break;length+=value.length;if(length>image.length){await reader.cancel();throw new Error('ExcessBody');}parts.push(value);}
   const bytes=Buffer.concat(parts);return {ok:response.ok&&bytes.length===image.length&&sum(bytes)===report.fixtureSha256,status,headersMs,bodyMs:performance.now()-start-headersMs,ms:performance.now()-start,bytes:bytes.length};
  }catch(error){return {ok:false,status,headersMs,ms:performance.now()-start,error:error.name,cause:error.cause?.code||null};}}
  async function curlRead(kind,index){const start=performance.now(),path=`${temp}/${kind}-${index}.png`;
   const config=`url = "${url}"\nheader = "Authorization: Bearer ${token}"\n`;
   return new Promise(resolve=>{let stdout='';const child=spawn('/usr/bin/curl',['--config','-','--silent','--show-error',kind==='curl-http2'?'--http2':'--http1.1','--max-time','15','--max-filesize',String(image.length),'--output',path,'--write-out','%{json}'],{stdio:['pipe','pipe','pipe']});
    child.stdout.on('data',data=>{if(stdout.length<65536)stdout+=data;});child.stderr.resume();child.stdin.on('error',()=>{});child.stdin.end(config);
    child.on('error',()=>resolve({ok:false,error:'CurlSpawnFailed',ms:performance.now()-start}));
    child.on('close',async code=>{let metrics={};try{metrics=JSON.parse(stdout);}catch{}let bytes;try{bytes=await readFile(path);}catch{bytes=Buffer.alloc(0);}
     resolve({ok:code===0&&metrics.http_code===200&&bytes.length===image.length&&sum(bytes)===report.fixtureSha256,exitCode:code,status:metrics.http_code||null,protocol:metrics.http_version||null,headersMs:metrics.time_starttransfer*1000,bodyMs:(metrics.time_total-metrics.time_starttransfer)*1000,ms:performance.now()-start,transferMs:metrics.time_total*1000,bytes:bytes.length,error:code===28?'CurlTimeout':code?'CurlTransferFailed':null});});
   });
  }
  const percentile=values=>values.length?[...values].sort((a,b)=>a-b)[Math.ceil(values.length*.95)-1]:null;
  for(const kind of plan.clients){const results=[],active=new Set();let dropped=0;const start=performance.now();
   for(let index=0;index<50;index++){const delay=start+index*200-performance.now();if(delay>0)await new Promise(resolve=>setTimeout(resolve,delay));if(active.size>=20){dropped++;continue;}
    const job=(kind==='node-fetch'?nodeRead():curlRead(kind,index)).then(result=>results.push(result));active.add(job);job.finally(()=>active.delete(job));}
   await Promise.all(active);const errors={};for(const result of results)if(!result.ok){const name=result.error||'IntegrityOrHttpFailure';errors[name]=(errors[name]||0)+1;}
   const stage={client:kind,planned:50,completed:results.length,successful:results.filter(x=>x.ok).length,dropped,failures:results.filter(x=>!x.ok).length,errors,elapsedMs:performance.now()-start,p95Ms:percentile(results.map(x=>x.ms)),p95HeadersMs:percentile(results.map(x=>x.headersMs).filter(Number.isFinite)),p95SuccessfulBodyMs:percentile(results.filter(x=>x.ok).map(x=>x.bodyMs)),results};report.stages.push(stage);await save();console.log(JSON.stringify({...stage,results:undefined}));
  }
  report.status='measurements_complete';
 }
}catch(error){report.failure=error.name;report.status='failed';process.exitCode=1;}
finally{
 try{await cleanup();}catch{report.cleanup={passed:false,error:'CleanupFailed'};process.exitCode=1;}
 if(temp)await rm(temp,{recursive:true,force:true});report.finishedAt=new Date().toISOString();await save();
 console.log(JSON.stringify({runId:report.runId,status:report.status,cleanup:report.cleanup,failure:report.failure}));if(!report.cleanup.passed)process.exitCode=1;
}
