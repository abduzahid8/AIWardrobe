/** Small public-data transport calibration; not a capacity test. */
import 'dotenv/config';
import {writeFile} from 'node:fs/promises';
import {performance} from 'node:perf_hooks';
import {responseDiagnostics,timingSummary} from '../load/http-diagnostics.mjs';
const base=process.env.EXPO_PUBLIC_SUPABASE_URL, key=process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
if(!base||!key)throw Error('Public project configuration missing');
const report={startedAt:new Date().toISOString(),scope:'120 public catalog reads at 4 RPS; transport calibration only, no fixtures/AI/writes or 10000-DAU claim',statuses:{},failures:[],timingMs:{}};
const all=[],headers=[],bodies=[],upstream=[],pending=new Set();
const started=performance.now();
async function request(index){
 const before=performance.now();let headersAt,metadata={};
 try{
  const r=await fetch(new URL('/rest/v1/shop_catalog?select=id,name,garment_type,price,image_url,currency,brand&limit=24',base),{headers:{apikey:key,Authorization:`Bearer ${key}`},redirect:'error',signal:AbortSignal.timeout(10000)});
  headersAt=performance.now();headers.push(headersAt-before);metadata=responseDiagnostics(r);if(metadata.upstreamMs!==undefined)upstream.push(metadata.upstreamMs);
  report.statuses[r.status]=(report.statuses[r.status]||0)+1;
  if(r.status!==200){report.failures.push({index,status:r.status,...metadata});await r.arrayBuffer();return;}
  const rows=await r.json();if(!Array.isArray(rows)||!rows.length||rows.some(row=>!row.id))report.failures.push({index,reason:'invalid_catalog_shape',...metadata});
 }catch(error){report.failures.push({index,reason:'transport_or_decode',...(typeof error?.cause?.code==='string'&&/^[A-Z0-9_]+$/.test(error.cause.code)?{causeCode:error.cause.code}:{}),...metadata});}
 finally{const now=performance.now();all.push(now-before);if(headersAt!==undefined)bodies.push(now-headersAt);}
}
for(let i=0;i<120;i++){
 const delay=started+i*250-performance.now();if(delay>0)await new Promise(resolve=>setTimeout(resolve,delay));
 const task=request(i).finally(()=>pending.delete(task));pending.add(task);
}
await Promise.all(pending);
report.completedAt=new Date().toISOString();report.requests=all.length;
report.timingMs={total:timingSummary(all),headers:timingSummary(headers),bodyIncludingValidation:timingSummary(bodies),reportedUpstream:timingSummary(upstream)};
await writeFile('docs/reports/readiness-evidence/catalog-transport-timing.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));if(report.failures.length)process.exitCode=1;
