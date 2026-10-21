/** Read hosted model IDs using the existing backend credential; no configuration writes. */
import 'dotenv/config';
import {writeFile} from 'node:fs/promises';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL,service=process.env.SUPABASE_SERVICE_KEY;
if(!base||!service)throw Error('Probe configuration unavailable');
const report={measuredAt:new Date().toISOString(),scope:'Authenticated hosted model listing; listing does not prove chat access, quality, pricing or throughput'};
try{
 const config=await fetch(new URL('/rest/v1/app_config?select=key,value&key=in.(nvidia_token,gemma)',base),{headers:{apikey:service,Authorization:`Bearer ${service}`},signal:AbortSignal.timeout(15000)});
 if(!config.ok)throw Error('Provider configuration unavailable');const rows=await config.json();
 const key=rows.find(r=>r.key==='nvidia_token'&&r.value)?.value||rows.find(r=>r.key==='gemma'&&r.value)?.value;
 if(!key)throw Error('Provider configuration unavailable');
 const response=await fetch('https://integrate.api.nvidia.com/v1/models',{headers:{Authorization:`Bearer ${key}`},signal:AbortSignal.timeout(20000)});
 report.status=response.status;const body=await response.json();const ids=Array.isArray(body.data)?body.data.map(row=>row.id).filter(id=>typeof id==='string'):[];
 report.advertisedModelCount=ids.length;
 report.textCandidates=ids.filter(id=>/llama-3.3-70b|qwen2.5.*instruct|qwen3.*instruct|nemotron.*(?:super|lightning)/i.test(id));
 report.visionBaselineAdvertised=ids.includes('meta/llama-3.2-11b-vision-instruct');
 if(!response.ok||!ids.length)process.exitCode=1;
}catch(error){report.failure=error.message==='Provider configuration unavailable'?error.message:error.name;process.exitCode=1;}
await writeFile('docs/reports/readiness-evidence/nvidia-model-catalog-20261021.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
