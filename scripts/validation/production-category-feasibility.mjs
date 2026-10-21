/** Public catalog diagnostic. Enumerates combinations; never generates replacement outfits. */
import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import ts from 'typescript';
import {createClient} from '@supabase/supabase-js';
const root=process.argv[2];
if(!root)throw new Error('Pass deployed function source directory');
const compile=source=>ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
const file=path.join(root,'generate-outfits/index.ts');
const requireModule=name=>{
 if(name.includes('server.ts'))return {serve:()=>{}};
 if(name.includes('supabase-js'))return {createClient};
 const result={};new Function('exports',compile(fs.readFileSync(path.resolve(path.dirname(file),name),'utf8')))(result);return result;
};
const api=new Function('require','exports','Deno',compile(fs.readFileSync(file,'utf8'))+'\nreturn {fetchShopItemsForAllSlotsEdge,filterValidOutfits,validateOutfitAgainstStyle,validateOutfitCompatibility,rankItemsForStyleEdge,collapseBySimEdge,simKeyEdge,outfitShopSpend};')(requireModule,{}, {env:{get:()=>undefined}});
const judge={};new Function('exports',compile(fs.readFileSync('backend/supabase/functions/_shared/occasionRules.ts','utf8')))(judge);
const client=createClient(process.env.EXPO_PUBLIC_SUPABASE_URL,process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,{auth:{persistSession:false,autoRefreshToken:false}});
const report={measuredAt:new Date().toISOString(),scope:'Public-catalog reads and deployed fetch/rank/full validation plus current shared style judge; no AI call or account writes. Empty wardrobe, no exclusions,20C,limit1. Not exact captured hosted candidate pools or exhaustive catalog feasibility.',checks:[]};
for(const [style,occasion] of [['casual','Everyday'],['sport','Gym workout']]){
 let candidates=await api.fetchShopItemsForAllSlotsEdge(client,['top','bottom','shoes'],style,6,new Set());
 candidates=candidates.filter(item=>!(Number(item.price)>3150000));
 candidates=api.rankItemsForStyleEdge(candidates,style,{minKeep:16,dropThreshold:-2,perCategoryFloor:4});
 candidates=api.collapseBySimEdge(candidates,item=>api.simKeyEdge(item.name||item.type||'',item.macroCategory||'')).slice(0,40);
 const pool=new Map(candidates.map(item=>[item.id,item]));const slot=name=>candidates.filter(item=>item.macroCategory===name);
 let tested=0,serverValid=0,strictValid=0,cheapest=null;
 const rejections={},serverRejections={};
 for(const top of slot('top'))for(const bottom of slot('bottom'))for(const shoes of slot('shoes')){
  const outfit={items:[top,bottom,shoes]};tested++;
  if(!api.filterValidOutfits([outfit],style,pool,false,{maxShopBudget:{amount:3150000,currency:'UZS'}},occasion).length){
   const structural=api.validateOutfitAgainstStyle(outfit,style,pool,false);
   const compatible=api.validateOutfitCompatibility(outfit,pool,style);
   const reason=!structural.ok?structural.reason:!compatible.ok?compatible.reason:api.outfitShopSpend(outfit.items,pool)>3150000?'over_budget':'other_server_rule';
   serverRejections[reason]=(serverRejections[reason]||0)+1;continue;
  }
  serverValid++;
  const reason=judge.styleJudgeReason(outfit.items,style,occasion,{temp:20,condition:'Clear'});
  if(reason){rejections[reason]=(rejections[reason]||0)+1;continue;}
  strictValid++;const spend=api.outfitShopSpend(outfit.items,pool);
  if(!cheapest||spend<cheapest.spendUzs)cheapest={spendUzs:spend,items:outfit.items.map(({id,name,price,currency})=>({id,name,price,currency}))};
 }
 report.checks.push({style,occasion,candidateCount:candidates.length,counts:{top:slot('top').length,bottom:slot('bottom').length,shoes:slot('shoes').length},combinationsTested:tested,serverValidWithinBudget:serverValid,strictValidWithinBudget:strictValid,strictRejections:rejections,serverRejections,cheapest,candidatePrices:candidates.map(({id,name,macroCategory,price,currency})=>({id,name,macroCategory,price,currency}))});
}
fs.writeFileSync('docs/reports/readiness-evidence/category-budget-feasibility.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
