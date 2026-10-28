/** Bounded production scan concurrency stages with tagged disposable users. */
import 'dotenv/config';
import {randomUUID,createHash} from 'node:crypto';
import {readFile,writeFile} from 'node:fs/promises';
const stages=[2,5,10],maxUsers=10,maxRequests=stages.reduce((a,b)=>a+b,0);
if(process.argv.includes('--plan')){
 console.log(JSON.stringify({stages,maxUsers,maxRequests,escalation:'Stop on any response-contract or expected category/color failure',sameKnownPhoto:true,scope:'Short scan burst only; not 10000DAU certification'},null,2));process.exit(0);
}
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,service=process.env.SUPABASE_SERVICE_KEY;
if(new URL(base).hostname!=='fyqpifmrsftsfqibhwhy.supabase.co'||!service)throw new Error('Expected project setup required');
const admin={apikey:service,Authorization:`Bearer ${service}`,'Content-Type':'application/json'};
const cancellation=new AbortController();
process.once('SIGINT',()=>cancellation.abort());process.once('SIGTERM',()=>cancellation.abort());
const report={measuredAt:new Date().toISOString(),runId:randomUUID(),scope:'Bounded live scan concurrency; same known photo, ordinary free users, expected primary category/color only',plannedStages:stages,maxRequests,stages:[],cleanup:[]};
const filename=process.env.SCAN_CONCURRENCY_REPORT||'production-scan-concurrency.json';
if(!/^[a-z0-9-]+\.json$/.test(filename))throw Error('Invalid report filename');
const file=new URL('../../docs/reports/readiness-evidence/'+filename,import.meta.url);
const save=()=>writeFile(file,JSON.stringify(report,null,2)+'\n');
async function call(path,headers,body,method='POST',cleanup=false){
 const start=performance.now();
 try{
  const response=await fetch(new URL(path,base),{method,headers,...(body===undefined?{}:{body:JSON.stringify(body)}),signal:cleanup?AbortSignal.timeout(15000):AbortSignal.any([cancellation.signal,AbortSignal.timeout(65000)]),redirect:'error'});
  let data;try{data=await response.json();}catch{data=null;}
  return {requestId:response.headers.get('sb-request-id')||response.headers.get('x-request-id'),executionId:response.headers.get('x-deno-execution-id'),region:response.headers.get('x-sb-edge-region'),status:response.status,ok:response.ok,data,latencyMs:Math.round(performance.now()-start)};
 }catch(error){return {status:0,ok:false,error:error.name,latencyMs:Math.round(performance.now()-start)};}
}
const users=[];
try{
 const photo=await readFile('docs/reports/readiness-evidence/jpeg-thumbnail-native-ios/02-originalPath.jpg');
 report.fixture={bytes:photo.length,sha256:createHash('sha256').update(photo).digest('hex'),selection:'beige/cream torso shirt or jacket; exclude separate green draped layer'};
 const payload={image:`data:image/jpeg;base64,${photo.toString('base64')}`,operation:'classify'};
 for(let index=0;index<maxUsers;index++){
  if(cancellation.signal.aborted)throw new Error('Cancelled');
  const suffix=randomUUID(),password=`${randomUUID()}Aa!9`,email=`readiness-${suffix}@aiwardrobe.club`;
  const created=await call('/auth/v1/admin/users',admin,{email,password,email_confirm:true,app_metadata:{readiness_audit:report.runId}});
  if(!created.ok||!created.data?.id)throw new Error('Fixture creation failed');
  const owner={id:created.data.id};users.push(owner);
  const login=await call('/auth/v1/token?grant_type=password',{apikey:anon,'Content-Type':'application/json'},{email,password});
  if(!login.data?.access_token)throw new Error('Fixture login failed');
  owner.headers={apikey:anon,Authorization:`Bearer ${login.data.access_token}`,'Content-Type':'application/json'};
 }
 report.createdUsers=users.length;await save();
 for(const concurrency of stages){
  if(cancellation.signal.aborted)throw new Error('Cancelled');
  const start=performance.now();let active=0,maxActive=0;
  const outcomes=await Promise.all(users.slice(0,concurrency).map(async(owner,index)=>{
   active++;maxActive=Math.max(maxActive,active);
   const result=await call('/functions/v1/ai-process',owner.headers,payload);active--;
   const c=result.data?.classification;
   const valid=result.ok&&result.data?.success===true&&typeof c?.category==='string'&&typeof c?.attributes?.color==='string';
   const expected=valid&&['shirt','jacket'].includes(c.category)&&['beige','cream'].includes(c.attributes.color);
   return {userIndex:index,requestId:result.requestId||null,executionId:result.executionId||null,region:result.region||null,status:result.status,latencyMs:result.latencyMs,responseContractPassed:valid,expectedPrimaryCategoryColorPassed:expected,category:c?.category||null,color:c?.attributes?.color||null,error:result.data?.error||result.error||null};
  }));
  const ordered=outcomes.map(r=>r.latencyMs).sort((a,b)=>a-b);
  const stage={concurrency,maxActive,elapsedMs:Math.round(performance.now()-start),outcomes,responsePasses:outcomes.filter(r=>r.responseContractPassed).length,expectedCategoryColorPasses:outcomes.filter(r=>r.expectedPrimaryCategoryColorPassed).length,medianMs:ordered.length%2?ordered[Math.floor(ordered.length/2)]:(ordered[ordered.length/2-1]+ordered[ordered.length/2])/2,maxMs:ordered.at(-1)};
  stage.passed=stage.responsePasses===concurrency&&stage.expectedCategoryColorPasses===concurrency;
  report.stages.push(stage);await save();console.log(JSON.stringify({stage:concurrency,passed:stage.passed,responsePasses:stage.responsePasses,expectedCategoryColorPasses:stage.expectedCategoryColorPasses,maxMs:stage.maxMs}));
  if(!stage.passed){report.stoppedEscalation='A bounded stage failed; no higher stage started';process.exitCode=1;break;}
 }
}catch(error){report.failure=error.message==='Cancelled'?'Cancelled':error.name;process.exitCode=1;}
finally{
 for(const owner of users){const revoked=owner.headers?await call('/auth/v1/logout?scope=global',owner.headers,{},'POST',true):null;const removed=await call(`/auth/v1/admin/users/${owner.id}`,admin,undefined,'DELETE',true);report.cleanup.push({id:owner.id,sessionsRevoked:revoked?revoked.ok:null,sessionCreated:!!owner.headers,deleted:removed.ok,status:removed.status});if(!removed.ok||(owner.headers&&!revoked?.ok))process.exitCode=1;}
 report.completedAt=new Date().toISOString();await save();console.log(JSON.stringify({completedStages:report.stages.length,createdUsers:users.length,deletedUsers:report.cleanup.filter(r=>r.deleted).length,failed:!!process.exitCode}));
}
