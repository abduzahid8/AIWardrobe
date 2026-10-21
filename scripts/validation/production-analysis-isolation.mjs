/** Two disposable accounts; one bounded provider call after admission checks. */
import 'dotenv/config';
import {randomUUID} from 'node:crypto';
import {readFile,writeFile} from 'node:fs/promises';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,service=process.env.SUPABASE_SERVICE_KEY;
if(!service||new URL(base).hostname!=='fyqpifmrsftsfqibhwhy.supabase.co')throw new Error('Expected setup required');
const admin={apikey:service,Authorization:`Bearer ${service}`,'Content-Type':'application/json'};
async function call(path,headers,body,method='POST'){
 const response=await fetch(new URL(path,base),{method,headers,...(body===undefined?{}:{body:JSON.stringify(body)}),signal:AbortSignal.timeout(65000),redirect:'error'});
 let data;try{data=await response.json();}catch{data=null;}return {ok:response.ok,status:response.status,data,diagnostic:{patternForm:response.headers.get('x-analysis-pattern-form'),contract:response.headers.get('x-analysis-contract'),reason:response.headers.get('x-analysis-rejection'),fields:response.headers.get('x-analysis-fields'),providerStatus:response.headers.get('x-analysis-provider-status'),requestId:response.headers.get('sb-request-id')}};
}
const report={measuredAt:new Date().toISOString(),runId:randomUUID(),scope:`Caller/victim disposable fixtures only; ${process.env.ANALYSIS_VERIFY_SINGLE==='1'?'two':'one'} sequential real analysis request(s), no customer data`,checks:[],cleanup:[]};
const singleOnly=process.env.ANALYSIS_PRIMARY_MODE==='single';report.mode=singleOnly?'single':'outfit';
const users=[];const check=(name,passed,details={})=>report.checks.push({name,passed,...details});
try{
 for(let i=0;i<2;i++){
  const password=`${randomUUID()}Aa!9`,email=`readiness-${randomUUID()}@aiwardrobe.club`;
  const created=await call('/auth/v1/admin/users',admin,{email,password,email_confirm:true,app_metadata:{readiness_audit:report.runId}});if(!created.ok||!created.data?.id)throw new Error('Fixture creation failed');
  const owner={id:created.data.id,itemId:randomUUID()};users.push(owner);
  const login=await call('/auth/v1/token?grant_type=password',{apikey:anon,'Content-Type':'application/json'},{email,password});if(!login.data?.access_token)throw new Error('Fixture login failed');
  owner.headers={apikey:anon,Authorization:`Bearer ${login.data.access_token}`,'Content-Type':'application/json'};
  const garment=await call('/rest/v1/clothing_items',admin,{id:owner.itemId,user_id:owner.id,name:'Readiness beige jacket',category:'jacket',type:'jacket',primary_color:'beige',color:'beige',style:'casual',pattern:'solid',image_url:`https://example.invalid/readiness/${i}.jpg`});if(!garment.ok)throw new Error('Garment fixture failed');
 }
 const [caller,victim]=users;
 const anonymous=await call('/functions/v1/analyze-outfit',{apikey:anon,Authorization:`Bearer ${anon}`,'Content-Type':'application/json'},{image:'AA==',userId:victim.id});check('public_anon_denied',anonymous.status===401,{status:anonymous.status});
 const free=await call('/functions/v1/analyze-outfit',caller.headers,{image:'AA==',userId:victim.id});check('free_access_denied',free.status===403&&free.data?.error==='feature_limit_reached',{status:free.status});
 const started=await call(`/rest/v1/profiles?id=eq.${caller.id}`,caller.headers,{trial_started_at:new Date().toISOString()},'PATCH');if(!started.ok)throw new Error('Trial fixture failed');
 const invalid=await call('/functions/v1/analyze-outfit',caller.headers,{image:{},userId:victim.id});check('invalid_shape_denied_before_provider',invalid.status===400,{status:invalid.status});
 const rls=await call(`/rest/v1/clothing_items?id=eq.${victim.itemId}&select=id`,caller.headers,undefined,'GET');check('caller_cannot_read_victim_fixture_via_rls',rls.ok&&Array.isArray(rls.data)&&rls.data.length===0,{status:rls.status});
 const photo=await readFile('docs/reports/readiness-evidence/jpeg-thumbnail-native-ios/02-originalPath.jpg');
 const start=Date.now();const result=await call('/functions/v1/analyze-outfit',caller.headers,{image:`data:image/jpeg;base64,${photo.toString('base64')}`,mode:singleOnly?'single':'outfit',userId:victim.id});
 const recommended=(result.data?.recommendations||[]).flatMap(r=>r.similarFromWardrobe||[]);
 check(singleOnly?'single_mode_returns_exactly_one_private_response':'forged_owner_request_returns_only_callers_wardrobe',result.ok&&result.data?.success===true&&(singleOnly?result.data?.detectedItems?.length===1:recommended.some(r=>r.id===caller.itemId))&&!JSON.stringify(result.data).includes(victim.itemId),{status:result.status,diagnostic:result.diagnostic,latencyMs:Date.now()-start,success:result.data?.success||false,ownFixtureSeen:recommended.some(r=>r.id===caller.itemId),foreignFixtureSeen:JSON.stringify(result.data).includes(victim.itemId),detectedItemCount:result.data?.detectedItems?.length||0,wardrobeRecommendationCount:recommended.length,error:result.data?.error||null});
 const crossCategory=(result.data?.recommendations||[]).filter(r=>(r.similarFromWardrobe||[]).some(item=>item.id===caller.itemId)&&r.detectedItem?.category!=='jacket');check('jacket_fixture_never_recommended_for_other_categories',result.data?.success===true&&crossCategory.length===0,{crossCategoryCount:crossCategory.length});
 if(process.env.ANALYSIS_VERIFY_SINGLE==='1'){const singleStart=Date.now();const single=await call('/functions/v1/analyze-outfit',caller.headers,{image:`data:image/jpeg;base64,${photo.toString('base64')}`,mode:'single',userId:victim.id});check('single_mode_returns_exactly_one_item',single.ok&&single.data?.success===true&&single.data?.detectedItems?.length===1&&single.data?.itemCount===1&&!JSON.stringify(single.data).includes(victim.itemId),{status:single.status,latencyMs:Date.now()-singleStart,itemCount:single.data?.detectedItems?.length||0,error:single.data?.error||null});}
}catch(error){report.failure=error.name;process.exitCode=1;}
finally{
 for(const user of users){const removed=await call(`/auth/v1/admin/users/${user.id}`,admin,undefined,'DELETE');report.cleanup.push({deleted:removed.ok,status:removed.status});if(!removed.ok)process.exitCode=1;}
 await writeFile(new URL(`../../docs/reports/readiness-evidence/${singleOnly?'production-analysis-single':'production-analysis-isolation'}.json`,import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}
if(report.checks.length!==(process.env.ANALYSIS_VERIFY_SINGLE==='1'?7:6)||report.checks.some(c=>!c.passed))process.exitCode=1;
