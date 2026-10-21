/** Read-only public catalog diagnostic executing the supplied real function source. */
import 'dotenv/config';
import {readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import ts from 'typescript';
import {createClient} from '@supabase/supabase-js';
const root=path.resolve(process.argv[2]||'backend/supabase/functions'),cache=new Map(),hashes={};
async function prepare(file){
 const source=await readFile(file,'utf8');hashes[path.relative(root,file)]=createHash('sha256').update(source).digest('hex');
 const dependencies=[...source.matchAll(/from\s+['"](\.[^'"]+)['"]/g)].map(match=>path.resolve(path.dirname(file),match[1]));
 for(const dependency of dependencies)if(!cache.has(dependency)){cache.set(dependency,null);await prepare(dependency);}
 cache.set(file,{code:ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText});
}
function execute(file){
 const entry=cache.get(file);if(entry.api)return entry.api;entry.api={};
 const dependency=name=>name.includes('server.ts')?{serve:()=>{}}:name.includes('supabase-js')?{createClient}:execute(path.resolve(path.dirname(file),name));
 const exposed=file.endsWith('/generate-outfits/index.ts')?'\nexports.fetchShopItemsForAllSlotsEdge=fetchShopItemsForAllSlotsEdge;':'';
 new Function('require','exports','Deno',entry.code+exposed)(dependency,entry.api,{env:{get:()=>undefined}});return entry.api;
}
const file=path.join(root,'generate-outfits/index.ts');await prepare(file);const api={...execute(file),...execute(path.join(root,'_shared/occasionRules.ts'))};
const statuses=[];
const client=createClient(process.env.EXPO_PUBLIC_SUPABASE_URL,process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,{auth:{persistSession:false,autoRefreshToken:false},global:{fetch:async(...args)=>{const response=await fetch(...args);statuses.push(response.status);return response;}}});
const started=performance.now();const items=await api.fetchShopItemsForAllSlotsEdge(client,['top','bottom','shoes'],'sport',6);
const report={measuredAt:new Date().toISOString(),scope:'Read-only real candidate fetch and activity admission against public catalog; no provider call, not exhaustive fashion or capacity proof',latencyMs:Math.round(performance.now()-started),httpStatuses:statuses,candidateCounts:Object.fromEntries(['top','bottom','shoes'].map(slot=>[slot,items.filter(item=>item.macroCategory===slot).length])),invalidActivityCandidates:items.filter(item=>api.rejectForActivity(item,'Gym workout')).length,sourceHashes:hashes};
report.passed=statuses.length===3&&statuses.every(status=>status===200)&&report.invalidActivityCandidates===0;
await writeFile(new URL('../../docs/reports/readiness-evidence/production-activity-pool-query.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));if(!report.passed)process.exitCode=1;
