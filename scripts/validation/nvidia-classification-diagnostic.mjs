/** One direct provider diagnostic through the actual staged classifier, not hosted admission. */
import 'dotenv/config';
import {readFile,writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import ts from 'typescript';
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
const report={measuredAt:new Date().toISOString(),scope:'One direct NVIDIA diagnostic using actual staged handler with controlled Auth/config; no production admission claim',providerResponses:[]};
let handler;
const config={select:()=>config,eq:()=>config,single:async()=>({data:{value:token},error:null})};
const trackedFetch=async(url,options)=>{
 if(url!=='https://integrate.api.nvidia.com/v1/chat/completions')throw new Error('Unexpected diagnostic provider target');
 let requestOptions=options;
 if(process.env.CLASSIFICATION_DIAGNOSTIC_FORMAT==='json_schema'){
  const body=JSON.parse(options.body);
  body.response_format={type:'json_schema',json_schema:{name:'GarmentClassification',schema:{type:'object',additionalProperties:false,properties:{
   category:{type:'string',enum:['t-shirt','shirt','blouse','sweater','hoodie','jacket','coat','cardigan','dress','skirt','pants','trousers','jeans','shorts','sneakers','boots','sandals','bag','hat','scarf','belt','clothing']},
   section:{type:'string',enum:['tops','bottoms','dresses','outerwear','shoes','accessories','other']},
   style:{type:'string',enum:['casual','formal','sport','semi_classic','elegant','minimalist','other']},
   color:{type:'string',enum:['black','white','grey','beige','cream','brown','red','pink','orange','yellow','green','teal','blue','navy','purple','multicolor']},
   material:{type:['string','null']},description:{type:'string',minLength:1}
  },required:['category','section','style','color','material','description']}}};
  requestOptions={...options,body:JSON.stringify(body)};report.requestFormat='json_schema';
 }
 const result=await fetch(url,requestOptions);
 const observation={status:result.status};
 if(result.ok){try{
  const data=await result.clone().json();const content=data?.choices?.[0]?.message?.content;
  observation.finishReason=data?.choices?.[0]?.finish_reason;
  observation.contentLength=typeof content==='string'?content.length:null;
  if(process.env.CLASSIFICATION_DIAGNOSTIC_DETAILS==='1'&&typeof content==='string')observation.fixtureContentPreview=content.slice(0,1600);
  try{const match=typeof content==='string'?content.match(/\{[\s\S]*\}/):null;const parsed=match?JSON.parse(match[0]):null;
   observation.parsed=parsed?Object.fromEntries(['category','section','style','color','material','description','confidence'].filter(k=>Object.hasOwn(parsed,k)).map(k=>[k,parsed[k]])):null;
  }catch{observation.parseable=false;}
 }catch{observation.envelopeReadable=false;}}
 report.providerResponses.push(observation);return result;
};
new Function('require','exports','Deno','console','fetch',ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText)(name=>{
 if(name.includes('server.ts'))return {serve:fn=>handler=fn};if(name.includes('rateLimit'))return {enforceRateLimit:async()=>null};if(name.includes('supabase-js'))return {createClient:()=>({from:()=>config})};if(name==='node:buffer')return require(name);return {};
},{},{env:{get:()=> 'controlled-config'}},{log:()=>{},warn:()=>{},error:()=>{}},trackedFetch);
const photo=await readFile('docs/reports/readiness-evidence/jpeg-thumbnail-native-ios/02-originalPath.jpg');
const start=Date.now();
const result=await handler(new Request('https://controlled.invalid',{method:'POST',body:JSON.stringify({image:`data:image/jpeg;base64,${photo.toString('base64')}`,operation:'classify'})}));
report.handlerStatus=result.status;report.latencyMs=Date.now()-start;
const body=await result.json();report.acceptedClassification=body.classification||null;report.error=body.error||null;
await writeFile(new URL(`../../docs/reports/readiness-evidence/${reportName||('nvidia-classification-diagnostic'+(process.env.CLASSIFICATION_DIAGNOSTIC_DETAILS==='1'?'-content':process.env.CLASSIFICATION_DIAGNOSTIC_FORMAT==='json_schema'?'-schema':'')+'.json')}`,import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
