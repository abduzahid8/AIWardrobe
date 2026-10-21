/** One real provider request; uses actual source validators, stores no image or raw response. */
import 'dotenv/config';
import {readFile,writeFile} from 'node:fs/promises';
import ts from 'typescript';
const service=process.env.SUPABASE_SERVICE_KEY,base=process.env.EXPO_PUBLIC_SUPABASE_URL;
if(!service||new URL(base).hostname!=='fyqpifmrsftsfqibhwhy.supabase.co')throw Error('Expected authenticated configuration');
const config=await fetch(new URL('/rest/v1/app_config?key=eq.nvidia_token&select=value',base),{headers:{apikey:service,Authorization:`Bearer ${service}`},signal:AbortSignal.timeout(15000)});
const rows=await config.json(),token=rows?.[0]?.value;if(!config.ok||typeof token!=='string')throw Error('Provider configuration unavailable');
const entry=process.env.ANALYSIS_DIAGNOSTIC_SOURCE||'backend/supabase/functions/analyze-outfit/index.ts';
const source=await readFile(entry,'utf8');const exports={};const observation={};
const trackedFetch=async(url,options)=>{if(url!=='https://integrate.api.nvidia.com/v1/chat/completions')throw Error('Unexpected target');let request=options;if(process.env.ANALYSIS_DIAGNOSTIC_MODEL){if(process.env.ANALYSIS_DIAGNOSTIC_MODEL!=='nvidia/nemotron-nano-12b-v2-vl')throw Error('Unsupported diagnostic model');const body=JSON.parse(options.body);body.model=process.env.ANALYSIS_DIAGNOSTIC_MODEL;request={...options,body:JSON.stringify(body)};}const response=await fetch(url,request);observation.providerHttpStatus=response.status;return response;};
new Function('require','exports','console','fetch',ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText+'\nexports.probe={callNvidiaVision,buildOutfitAnalysisPrompt,parseNvidiaResponseText,extractContentFromResponse,parseJsonArrayFromContent,detectedItemIssues,normalizeDetectedItem};')(
 name=>name.includes('server.ts')?{serve:()=>{}}:{},exports,{log:()=>{},error:()=>{},warn:()=>{}},trackedFetch);
const report={providerModel:process.env.ANALYSIS_DIAGNOSTIC_MODEL||'meta/llama-3.2-11b-vision-instruct',mode:process.env.ANALYSIS_DIAGNOSTIC_MODE==='single'?'single':'outfit',measuredAt:new Date().toISOString(),scope:'One direct NVIDIA call through actual source request/parser/validator; no hosted admission or visual-accuracy claim',observation};
const photo=await readFile('docs/reports/readiness-evidence/jpeg-thumbnail-native-ios/02-originalPath.jpg');const start=Date.now();
try{
 const raw=await exports.probe.callNvidiaVision(token,`data:image/jpeg;base64,${photo.toString('base64')}`,exports.probe.buildOutfitAnalysisPrompt(process.env.ANALYSIS_DIAGNOSTIC_MODE==='single'?'single':'outfit'));
 const envelope=exports.probe.parseNvidiaResponseText(raw),content=exports.probe.extractContentFromResponse(raw),items=exports.probe.parseJsonArrayFromContent(content)?.map(exports.probe.normalizeDetectedItem);
 observation.finishReason=envelope.choices?.[0]?.finish_reason||null;observation.contentLength=typeof content==='string'?content.length:null;observation.arrayParsed=Array.isArray(items);
 observation.items=items?.map((item,index)=>({index,invalidPatternTerm:exports.probe.detectedItemIssues(item).includes('pattern')&&(item?.pattern===null||typeof item?.pattern==='string'&&/^[a-z _/-]{1,40}$/i.test(item.pattern))?item.pattern:undefined,taxonomy:{category:typeof item?.category==='string'&&/^[a-z -]{1,40}$/i.test(item.category)?item.category:'invalid',section:typeof item?.section==='string'&&/^[a-z -]{1,40}$/i.test(item.section)?item.section:'invalid'},invalidFields:exports.probe.detectedItemIssues(item),invalidFitTerm:exports.probe.detectedItemIssues(item).includes('fit')&&(item?.fit===null||typeof item?.fit==='string'&&/^[a-z _/-]{1,40}$/i.test(item.fit))?item.fit:undefined,missingFields:(!item||typeof item!=='object')?[]:['category','section','specificType','primaryColor','colorHex','style','material','pattern','fit','description'].filter(field=>!Object.hasOwn(item,field))}))||[];
 observation.expectedSingleJacket=Array.isArray(items)&&items.length===1&&items[0].category==='jacket'&&String(items[0].primaryColor||'').toLowerCase()==='beige';
 observation.schemaValid=!!items&&items.every(item=>exports.probe.detectedItemIssues(item).length===0)&&observation.finishReason!=='length';
}catch(error){report.failure=error.name;observation.providerFailure=true;}
report.latencyMs=Date.now()-start;
await writeFile(new URL('../../docs/reports/readiness-evidence/nvidia-analysis-diagnostic.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
