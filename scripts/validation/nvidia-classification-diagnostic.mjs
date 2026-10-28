/** One direct provider diagnostic through the actual staged classifier, not hosted admission. */
import 'dotenv/config';
import {readFile,writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import ts from 'typescript';
import {createHash} from 'node:crypto';
const require=createRequire(import.meta.url);
const service=process.env.SUPABASE_SERVICE_KEY;
if(!service)throw new Error('Setup credential required');
const response=await fetch(new URL('/rest/v1/app_config?key=eq.nvidia_token&select=value',process.env.EXPO_PUBLIC_SUPABASE_URL),{headers:{apikey:service,Authorization:`Bearer ${service}`},signal:AbortSignal.timeout(15000)});
const configRows=await response.json();const token=configRows?.[0]?.value;
if(!response.ok||typeof token!=='string')throw new Error('Provider configuration unavailable');
const entry=process.env.CLASSIFICATION_DIAGNOSTIC_SOURCE||'/tmp/aiwardrobe-scan-deploy/supabase/functions/ai-process/index.ts';
const reportName=process.env.CLASSIFICATION_DIAGNOSTIC_REPORT;
if(reportName&&!/^[a-z0-9-]+\.json$/.test(reportName))throw new Error('Invalid diagnostic report name');
const source=await readFile(entry,'utf8');
const candidateModel=process.env.CLASSIFICATION_DIAGNOSTIC_MODEL;
if(candidateModel&&!/^[a-z0-9][a-z0-9./_-]{2,100}$/.test(candidateModel))throw Error('Invalid diagnostic model');
const report={measuredAt:new Date().toISOString(),sourceSha256:createHash('sha256').update(source).digest('hex'),scope:'One direct NVIDIA diagnostic using actual staged handler with controlled Auth/config; no production admission claim',providerResponses:[]};
let handler;
const config={select:()=>config,eq:()=>config,single:async()=>({data:{value:token},error:null})};
const trackedFetch=async(url,options)=>{
 if(url!=='https://integrate.api.nvidia.com/v1/chat/completions')throw new Error('Unexpected diagnostic provider target');
 let requestOptions=options;
 if(candidateModel){const body=JSON.parse(options.body);body.model=candidateModel;if(candidateModel==='nvidia/nemotron-3.5-super-vl-120b-a12b')body.chat_template_kwargs={enable_thinking:false};else delete body.chat_template_kwargs;requestOptions={...options,body:JSON.stringify(body)};report.candidateModel=candidateModel;}
 if(process.env.CLASSIFICATION_DIAGNOSTIC_FORMAT==='json_schema'){
  const body=JSON.parse(requestOptions.body);
  body.response_format={type:'json_schema',json_schema:{name:'GarmentClassification',schema:{type:'object',additionalProperties:false,properties:{
   category:{type:'string',enum:['t-shirt','shirt','blouse','sweater','hoodie','jacket','coat','cardigan','dress','skirt','pants','trousers','jeans','shorts','sneakers','boots','sandals','bag','hat','scarf','belt','clothing']},
   section:{type:'string',enum:['tops','bottoms','dresses','outerwear','shoes','accessories','other']},
   style:{type:'string',enum:['casual','formal','sport','semi_classic','elegant','minimalist','other']},
   color:{type:'string',enum:['black','white','grey','beige','cream','brown','red','pink','orange','yellow','green','teal','blue','navy','purple','multicolor']},
   material:{type:['string','null']},description:{type:'string',minLength:1}
  },required:['category','section','style','color','material','description']}}};
  requestOptions={...options,body:JSON.stringify(body)};report.requestFormat='json_schema';
 }
 const sent=JSON.parse(requestOptions.body);
 report.requestConfig=Object.fromEntries(['model','max_tokens','temperature','reasoning_effort','chat_template_kwargs','stream'].filter(k=>Object.hasOwn(sent,k)).map(k=>[k,sent[k]]));
 const result=await fetch(url,requestOptions);
 const observation={status:result.status};
 if(result.ok){try{
  const data=await result.clone().json();const content=data?.choices?.[0]?.message?.content;
  observation.finishReason=data?.choices?.[0]?.finish_reason;
  observation.tokenUsage=data?.usage?Object.fromEntries(['prompt_tokens','completion_tokens','total_tokens'].filter(k=>Number.isFinite(data.usage[k])).map(k=>[k,data.usage[k]])):null;
  observation.reasoningTokens=Number.isFinite(data?.usage?.completion_tokens_details?.reasoning_tokens)?data.usage.completion_tokens_details.reasoning_tokens:null;
  observation.contentLength=typeof content==='string'?content.length:null;
  if(process.env.CLASSIFICATION_DIAGNOSTIC_DETAILS==='1'&&typeof content==='string')observation.fixtureContentPreview=content.slice(0,1600);
  try{const match=typeof content==='string'?content.match(/\{[\s\S]*\}/):null;const parsed=match?JSON.parse(match[0]):null;
   observation.parsed=parsed?Object.fromEntries(['category','section','style','color','material','description','confidence','shoeType'].filter(k=>Object.hasOwn(parsed,k)).map(k=>[k,parsed[k]])):null;
  }catch{observation.parseable=false;}
 }catch{observation.envelopeReadable=false;}}
 report.providerResponses.push(observation);return result;
};
new Function('require','exports','Deno','console','fetch',ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText)(name=>{
 if(name.includes('server.ts'))return {serve:fn=>handler=fn};if(name.includes('rateLimit'))return {enforceRateLimit:async()=>null};if(name.includes('supabase-js'))return {createClient:()=>({from:()=>config})};if(name==='node:buffer')return require(name);return {};
},{},{env:{get:()=> 'controlled-config'}},{log:()=>{},warn:()=>{},error:()=>{}},trackedFetch);
const fixture=process.env.CLASSIFICATION_DIAGNOSTIC_FIXTURE||'docs/reports/readiness-evidence/jpeg-thumbnail-native-ios/02-originalPath.jpg';
if(!/^(?:assets\/images\/[a-z0-9_]+\.png|docs\/reports\/readiness-evidence\/jpeg-thumbnail-native-ios\/02-originalPath\.jpg)$/.test(fixture))throw Error('Invalid diagnostic fixture');
const photo=await readFile(fixture);report.fixture={path:fixture,bytes:photo.length,sha256:createHash('sha256').update(photo).digest('hex')};
const mime=fixture.endsWith('.png')?'image/png':'image/jpeg';
const start=Date.now();
const result=await handler(new Request('https://controlled.invalid',{method:'POST',body:JSON.stringify({image:`data:${mime};base64,${photo.toString('base64')}`,operation:'classify'})}));
report.handlerStatus=result.status;report.latencyMs=Date.now()-start;
const body=await result.json();report.acceptedClassification=body.classification||null;report.error=body.error||null;
await writeFile(new URL(`../../docs/reports/readiness-evidence/${reportName||('nvidia-classification-diagnostic'+(process.env.CLASSIFICATION_DIAGNOSTIC_DETAILS==='1'?'-content':process.env.CLASSIFICATION_DIAGNOSTIC_FORMAT==='json_schema'?'-schema':'')+'.json')}`,import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));

if(result.status!==200||!body.classification)process.exitCode=1;
