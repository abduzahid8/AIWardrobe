/** Hosted capability probe only; never installs a fallback or changes production configuration. */
import 'dotenv/config';
import {writeFile} from 'node:fs/promises';
const service=process.env.SUPABASE_SERVICE_KEY,base=process.env.EXPO_PUBLIC_SUPABASE_URL;
if(!service||!base)throw Error('Probe setup credential unavailable');
const report={measuredAt:new Date().toISOString(),scope:'Hosted NVIDIA structured-output capability; synthetic IDs, not outfit quality or capacity',checks:[]};
const modeOverride=process.env.NVIDIA_SCHEMA_PROBE_MODE;
if(modeOverride && modeOverride!=='guided_regex')throw Error('Invalid capability mode');
try{
 const config=await fetch(new URL('/rest/v1/app_config?select=key,value&key=in.(nvidia_token,gemma)',base),{headers:{apikey:service,Authorization:`Bearer ${service}`},signal:AbortSignal.timeout(15000)});
 if(!config.ok)throw Error('Provider configuration unavailable');
 const rows=await config.json();
 const key=rows.find(row=>row.key==='nvidia_token'&&row.value)?.value||rows.find(row=>row.key==='gemma'&&row.value)?.value;
 if(!key)throw Error('Provider configuration unavailable');
 const schema={type:'object',additionalProperties:false,required:['top','bottom','shoes'],properties:{top:{type:'string',enum:['probe-top']},bottom:{type:'string',enum:['probe-bottom']},shoes:{type:'string',enum:['probe-shoes']}}};
 for(const mode of modeOverride?[modeOverride]:['json_schema','json_object']){
  const started=performance.now();
  try{
   const constraint=mode==='guided_regex'?{nvext:{guided_regex:'\\{"top":"probe-top","bottom":"probe-bottom","shoes":"probe-shoes"\\}'}}:{response_format:mode==='json_schema'?{type:mode,json_schema:{name:'outfit_slots_probe',schema}}:{type:mode}};
   const response=await fetch('https://integrate.api.nvidia.com/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${key}`},body:JSON.stringify({model:'meta/llama-3.2-11b-vision-instruct',messages:[{role:'system',content:'Return a JSON object selecting clothing IDs. Available IDs: top=probe-top, bottom=probe-bottom, shoes=probe-shoes.'},{role:'user',content:'Return top and bottom only. Omit the shoes key.'}],temperature:0,max_tokens:120,...constraint}),signal:AbortSignal.timeout(45000)});
   const data=await response.json();
   const content=String(data.choices?.[0]?.message?.content||'');
   let value;try{value=JSON.parse(content.trim().replace(/^```(?:json)?\s*/i,'').replace(/\s*```$/,''));}catch{}
   report.checks.push({mode,status:response.status,latencyMs:Math.round(performance.now()-started),exactRequiredSlots:value?.top==='probe-top'&&value?.bottom==='probe-bottom'&&value?.shoes==='probe-shoes'&&Object.keys(value).length===3,returnedKeys:value&&typeof value==='object'?Object.keys(value).slice(0,10):[],syntheticContentPreview:content.slice(0,1000),finishReason:data.choices?.[0]?.finish_reason||null,errorCode:typeof data.error?.code==='string'?data.error.code.slice(0,80):null});
  }catch(error){report.checks.push({mode,latencyMs:Math.round(performance.now()-started),transportError:error.name});}
 }
}catch(error){report.failure=error.message==='Provider configuration unavailable'?error.message:error.name;process.exitCode=1;}
await writeFile(new URL(`../../docs/reports/readiness-evidence/${modeOverride?'nvidia-hosted-regex-capability':'nvidia-hosted-schema-capability'}.json`,import.meta.url),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
if(!report.checks.some(check=>check.mode!=='json_object'&&check.status===200&&check.exactRequiredSlots))process.exitCode=1;
