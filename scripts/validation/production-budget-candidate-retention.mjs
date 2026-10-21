/** Public catalog + the exact deployed ranking helper, executed in Node; no writes or provider calls. */
import 'dotenv/config';
import {readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import ts from 'typescript';
import {createClient} from '@supabase/supabase-js';
const root=path.resolve(process.argv[2]||'');if(!process.argv[2])throw Error('Deployed function snapshot required');
const helpers={};
for(const name of ['occasionRules','outfitStructure'])new Function('exports',ts.transpileModule(await readFile(path.join(root,'_shared',name+'.ts'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText)(helpers);
const entry=await readFile(path.join(root,'generate-outfits/index.ts'),'utf8');
const strip=s=>s.replace(/^import .*\n/gm,'').replace(/\bexport\s+/g,'');
const compiled=ts.transpileModule(strip(await readFile(path.join(root,'_shared/modelOutfitResponse.ts'),'utf8'))+'\n'+strip(entry),{compilerOptions:{module:ts.ModuleKind.None,target:ts.ScriptTarget.ES2022}}).outputText;
const api=new Function('serve','createClient','Deno',...Object.keys(helpers),compiled+'\nreturn {retainBudgetFloorCandidates,rankItemsForStyleEdge,macroCategory,collapseBySimEdge,simKeyEdge};')(()=>{},createClient,{env:{get:()=>undefined}},...Object.values(helpers));
const client=createClient(process.env.EXPO_PUBLIC_SUPABASE_URL,process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,{auth:{persistSession:false,autoRefreshToken:false},global:{fetch:(input,init)=>fetch(input,{...init,signal:AbortSignal.timeout(15000)})}});
const report={measuredAt:new Date().toISOString(),scope:'Read-only real public catalog and exact deployed helper in Node. Not hosted handler execution, provider success or capacity evidence.',entrySha256:createHash('sha256').update(entry).digest('hex')};
const originalLog=console.log;console.log=()=>{};
try{
 const rows=[];
 for(let offset=0;;offset+=1000){const {data,error}=await client.from('shop_catalog').select('id,name,price,currency,category,garment_type,description,image_url').eq('is_active',true).order('id',{ascending:true}).range(offset,offset+999);if(error)throw Error('Catalog read failed');rows.push(...data);if(data.length<1000)break;}
 const items=rows.map(row=>({...row,id:'shop_'+row.id,type:row.garment_type,macroCategory:api.macroCategory(row.category,row.garment_type,row.name),isShopItem:true}));
 const eligible=item=>!!item.image_url&&!helpers.rejectForStyleOccasion(item,'casual','Everyday')&&!helpers.rejectForClimate(item,{temp:20})&&!helpers.rejectForActivity(item,'Everyday');
 const cost=item=>item.currency==='UZS'&&Number(item.price)>0?Number(item.price):Infinity;
 const valid=items.filter(eligible).filter(item=>Number.isFinite(cost(item)));
 const ranked=api.rankItemsForStyleEdge(valid,'casual',{minKeep:16,dropThreshold:-2,perCategoryFloor:4,eligibility:eligible});
 const before=api.collapseBySimEdge(ranked,item=>api.simKeyEdge(item.name,item.macroCategory)).slice(0,40);
 const after=api.retainBudgetFloorCandidates(valid,before,cost,eligible);
 report.catalogRows=rows.length;report.eligiblePricedRows=valid.length;
 report.slots=['top','bottom','shoes','outerwear','dress'].flatMap(slot=>{
 const inSlot=item=>(helpers.isDressGarment(item)?'dress':item.macroCategory)===slot;
 const source=valid.filter(inSlot);if(!source.length)return [];
 const minimum=Math.min(...source.map(cost));
 return [{slot,sourceMinimumUzs:minimum,beforeMinimumUzs:Math.min(...before.filter(inSlot).map(cost))||null,afterMinimumUzs:Math.min(...after.filter(inSlot).map(cost))||null,preserved:after.some(item=>inSlot(item)&&cost(item)===minimum)}];});
 report.checks={allKnownSlotMinimaRetained:report.slots.length>=3&&report.slots.every(slot=>slot.preserved),eligibleOnly:after.every(eligible),withinPoolCap:after.length<=40,uniqueIds:new Set(after.map(item=>item.id)).size===after.length};
 report.passed=Object.values(report.checks).every(Boolean);
}catch(error){report.failure=error.name;report.passed=false;}
finally{console.log=originalLog;await writeFile('docs/reports/readiness-evidence/budget-candidate-retention-public-v111.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));if(!report.passed)process.exitCode=1;}
