/** Bounded live NVIDIA classification checks on one tagged disposable account. */
import 'dotenv/config';
import {randomUUID} from 'node:crypto';
import {writeFile,readFile} from 'node:fs/promises';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,anon=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,service=process.env.SUPABASE_SERVICE_KEY;
if(!service)throw new Error('Setup credential required');
const admin={apikey:service,Authorization:`Bearer ${service}`,'Content-Type':'application/json'};
async function call(path,headers,body,method='POST'){
 const response=await fetch(new URL(path,base),{method,headers,...(body===undefined?{}:{body:JSON.stringify(body)}),signal:AbortSignal.timeout(65000),redirect:'error'});
 let data;try{data=await response.json();}catch{data=null;}return {ok:response.ok,status:response.status,data};
}
const report={measuredAt:new Date().toISOString(),scope:'Two bounded live NVIDIA classifications; response contract, not full visual accuracy or capacity',checks:[]};
const repetitions=process.env.SCAN_CLASSIFICATION_REPETITIONS==='3'?3:1;
report.scope=`One invalid-image check plus ${repetitions} valid-photo classifications; expected primary category/color only, not exhaustive visual accuracy or capacity`;
let owner;
try{
 const suffix=randomUUID(),password=`${randomUUID()}Aa!9`,email=`readiness-${suffix}@aiwardrobe.club`;
 const created=await call('/auth/v1/admin/users',admin,{email,password,email_confirm:true,app_metadata:{readiness_audit:suffix}});if(!created.ok||!created.data?.id)throw new Error('Fixture creation failed');owner=created.data.id;
 const login=await call('/auth/v1/token?grant_type=password',{apikey:anon,'Content-Type':'application/json'},{email,password});if(!login.data?.access_token)throw new Error('Fixture login failed');
 const user={apikey:anon,Authorization:`Bearer ${login.data.access_token}`,'Content-Type':'application/json'};
 const before=Date.now();
 const invalid=await call('/functions/v1/ai-process',user,{image:'AA==',operation:'classify'});
 report.checks.push({name:'invalid_image_provider_failure_is_not_fabricated_success',passed:invalid.status===503&&invalid.data?.success===false&&invalid.data?.classification===null&&invalid.data?.error==='ai_classification_unavailable',status:invalid.status,latencyMs:Date.now()-before,error:invalid.data?.error});
 const photo=await readFile('docs/reports/readiness-evidence/jpeg-thumbnail-native-ios/02-originalPath.jpg');
 for(let iteration=1;iteration<=repetitions;iteration++){
 const start=Date.now();const result=await call('/functions/v1/ai-process',user,{image:`data:image/jpeg;base64,${photo.toString('base64')}`,operation:'classify'});
 const c=result.data?.classification;
 report.checks.push({name:`expected_primary_category_and_color_${iteration}`,passed:result.ok&&['shirt','jacket'].includes(c?.category)&&['beige','cream'].includes(c?.attributes?.color),expected:'Beige/cream torso shirt or jacket, excluding separate green draped layer'});
 report.checks.push({name:`known_clothing_photo_returns_validated_classification_${iteration}`,passed:result.ok&&result.data?.success===true&&typeof c?.category==='string'&&typeof c?.attributes?.color==='string'&&typeof result.data?.description==='string',status:result.status,latencyMs:Date.now()-start,classification:c||null,description:result.data?.description||null,error:result.data?.error||null});
 }
}catch(error){report.failure=error.name||'Error';process.exitCode=1;}
finally{
 if(owner){const removed=await call(`/auth/v1/admin/users/${owner}`,admin,undefined,'DELETE');report.cleanup={temporaryUserDeleted:removed.ok,status:removed.status};if(!removed.ok)process.exitCode=1;}
 await writeFile(new URL('../../docs/reports/readiness-evidence/production-scan-classification-fewshot.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}
if(report.checks.length!==(1+2*repetitions)||report.checks.some(c=>!c.passed))process.exitCode=1;
