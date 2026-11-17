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
 const exposed=file.endsWith('/generate-outfits/index.ts')?'\nexports.fetchShopItemsForAllSlotsEdge=fetchShopItemsForAllSlotsEdge;exports.retainBudgetFloorCandidates=retainBudgetFloorCandidates;':'';
 new Function('require','exports','Deno',entry.code+exposed)(dependency,entry.api,{env:{get:()=>undefined}});return entry.api;
}
const file=path.join(root,'generate-outfits/index.ts');await prepare(file);const api={...execute(file),...execute(path.join(root,'_shared/occasionRules.ts')),...execute(path.resolve('src/domain/garmentContext.ts')),...execute(path.resolve('src/domain/stylingContext.ts'))};
const statuses=[];
const client=createClient(process.env.EXPO_PUBLIC_SUPABASE_URL,process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,{auth:{persistSession:false,autoRefreshToken:false},global:{fetch:async(...args)=>{const response=await fetch(...args);statuses.push(response.status);return response;}}});

const context=api.stylingContext({style:'classic',occasion:'work'});
const eligible=item=>!api.garmentContextReason(item,context)&&!api.rejectForStyleOccasion(item,'classic','work');
const measurements=[];
for(let run=0;run<2;run++) {
 statuses.length=0;const started=performance.now();
 const items=await api.fetchShopItemsForAllSlotsEdge(client,['top','outerwear','bottom','shoes'],'classic',12,new Set(),eligible,undefined,undefined,'work',3);
 const ranked=[...items].sort((a,b)=>api.scoreGarmentForContext(b,context)-api.scoreGarmentForContext(a,context));
 const trimmed=api.retainBudgetFloorCandidates(ranked,ranked,()=>0,eligible,undefined,10);
 measurements.push({run:run?'warm':'cold',latencyMs:Math.round(performance.now()-started),requests:statuses.length,httpStatuses:[...statuses],discovered:items.length,slots:Object.fromEntries(['top','outerwear','bottom','shoes'].map(slot=>[slot,trimmed.filter(item=>item.macroCategory===slot).length])),products:trimmed.map(item=>({id:item.id,name:item.name,slot:item.macroCategory}))});
}
const report={kind:'read-only execution of actual candidate retrieval source, not a deployed Edge or simulator test',measurements};
await writeFile('docs/reports/running-stylist-repair/retrieval.json',JSON.stringify(report,null,2));
console.log(JSON.stringify(measurements.map(({products,...measurement})=>measurement),null,2));
