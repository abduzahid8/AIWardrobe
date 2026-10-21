/** Full current handler through Node/real services; does not change hosted routing. */
import 'dotenv/config';
import ts from 'typescript';
import {readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {createClient} from '@supabase/supabase-js';
const root=path.resolve(process.argv[2]||'');if(!process.argv[2])throw Error('Pass the downloaded function directory');
const model=process.env.NVIDIA_CANDIDATE_MODEL||'nvidia/nemotron-3.5-lightning-30b-a3b';
if(!/^[a-z0-9][a-z0-9./_-]{2,100}$/.test(model))throw Error('Invalid candidate model');
const variables={SUPABASE_URL:process.env.EXPO_PUBLIC_SUPABASE_URL,SUPABASE_ANON_KEY:process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,SUPABASE_SERVICE_ROLE_KEY:process.env.SUPABASE_SERVICE_KEY,NVIDIA_TEXT_MODEL:model,NVIDIA_OUTFIT_SELECTION_MODE:process.env.NVIDIA_OUTFIT_SELECTION_MODE};
if(!variables.SUPABASE_SERVICE_ROLE_KEY)throw Error('Test setup credential missing');
const modules=new Map(),hashes={};let handler;
async function collect(file){
 const full=path.resolve(root,file);if(!full.startsWith(root+path.sep))throw Error('Module escapes downloaded package');
 if(modules.has(full))return;
 const source=await readFile(full,'utf8');hashes[path.relative(root,full)]=createHash('sha256').update(source).digest('hex');
 const ast=ts.createSourceFile(full,source,ts.ScriptTarget.Latest,true);
 modules.set(full,{source,api:{},loaded:false});
 for(const node of ast.statements)if(ts.isImportDeclaration(node)&&node.moduleSpecifier.text.startsWith('.'))await collect(path.relative(root,path.resolve(path.dirname(full),node.moduleSpecifier.text)));
}
function load(full){
 const item=modules.get(full);if(!item)throw Error('Uncollected dependency');if(item.loaded)return item.api;item.loaded=true;
 const code=ts.transpileModule(item.source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
 new Function('require','exports','Deno',code)(name=>{
  if(name.includes('supabase-js'))return{createClient};
  if(name==='https://deno.land/std@0.168.0/http/server.ts')return{serve:fn=>{handler=fn;}};
  if(name.startsWith('.'))return load(path.resolve(path.dirname(full),name));
  throw Error('Unsupported downloaded dependency');
 },item.api,{env:{get:name=>variables[name]}});
 return item.api;
}
await collect('generate-outfits/index.ts');load(path.join(root,'generate-outfits/index.ts'));
if(!handler)throw Error('Candidate handler not captured');
const originalFetch=globalThis.fetch;
const observations=[];
globalThis.fetch=async(input,init)=>{
 const url=new URL(typeof input==='string'||input instanceof URL?input:input.url);
 if(url.origin===new URL(variables.SUPABASE_URL).origin&&url.pathname==='/functions/v1/generate-outfits')return handler(new Request(input,init));
 if(url.origin==='https://integrate.api.nvidia.com'){
  const body=JSON.parse(init.body),started=performance.now();
  const observation={model:body.model,requestCharacters:body.messages.reduce((sum,m)=>sum+String(m.content).length,0),maxTokens:body.max_tokens};observations.push(observation);
  try{const response=await originalFetch(input,init);const data=await response.clone().json();Object.assign(observation,{status:response.status,latencyMs:Math.round(performance.now()-started),finishReason:data.choices?.[0]?.finish_reason??null,contentCharacters:String(data.choices?.[0]?.message?.content||'').length,usage:data.usage?{promptTokens:data.usage.prompt_tokens,completionTokens:data.usage.completion_tokens}:null});return response;}catch(error){Object.assign(observation,{latencyMs:Math.round(performance.now()-started),transportError:error.name});throw error;}
 }
 return originalFetch(input,init);
};
try{await import(process.env.OLD_MONEY_REPORT?'./production-old-money-work.mjs':'./production-outfit-quality.mjs');}
finally{
 globalThis.fetch=originalFetch;
 const filename=process.env.OLD_MONEY_REPORT||process.env.OUTFIT_QUALITY_REPORT;
 if(filename&&/^[a-z0-9-]+\.json$/.test(filename)){
  const reportPath=new URL('../../docs/reports/readiness-evidence/'+filename,import.meta.url);
  const report=JSON.parse(await readFile(reportPath,'utf8'));
  report.providerObservations=observations;
  report.execution={runtime:'Local Node with actual current handler modules; real Supabase Auth/catalog/rate guards and hosted NVIDIA',candidateModel:model,sourceHashes:hashes,productionConfigurationChanged:false,limits:'Not hosted Edge runtime or native UI; this comparison does not establish production model throughput.'};
  await writeFile(reportPath,JSON.stringify(report,null,2)+'\n');
 }
}
