import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import ts from 'typescript';
import {createClient} from '@supabase/supabase-js';
const root=process.argv[2];if(!root) throw new Error('Pass a downloaded deployed function source directory');
const file=path.join(root,'generate-outfits/index.ts');
const compile=source=>ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
const requireModule=name=>{
 if(name.includes('server.ts'))return {serve:()=>{}};
 if(name.includes('supabase-js'))return {createClient};
 const result={};new Function('exports',compile(fs.readFileSync(path.resolve(path.dirname(file),name),'utf8')))(result);return result;
};
const source=fs.readFileSync(file,'utf8');
const api=new Function('require','exports','Deno',compile(source)+'\nreturn {fetchShopItemsForAllSlotsEdge,validateOutfitAgainstStyle,outfitShopSpend};')(requireModule,{}, {env:{get:()=>undefined}});
const client=createClient(process.env.EXPO_PUBLIC_SUPABASE_URL,process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY,{auth:{persistSession:false,autoRefreshToken:false}});
const candidates=await api.fetchShopItemsForAllSlotsEdge(client,['top','bottom','shoes'],'old_money',Number(process.argv[3] || 6),new Set());
const pool=new Map(candidates.map(item=>[item.id,item]));const slot=name=>candidates.filter(item=>item.macroCategory===name);
let tested=0,valid=0,cheapest=null;
for(const top of slot('top'))for(const bottom of slot('bottom'))for(const shoes of slot('shoes')){
 const outfit={items:[top,bottom,shoes]};tested++;
 if(!api.validateOutfitAgainstStyle(outfit,'old_money',pool,false).ok)continue;
 valid++;const spend=api.outfitShopSpend(outfit.items,pool);
 if(!cheapest||spend<cheapest.spendUzs)cheapest={spendUzs:spend,items:outfit.items.map(({id,name,price,currency})=>({id,name,price,currency}))};
}
const report={measuredAt:new Date().toISOString(),scope:'Public-catalog read and deployed style validator; no model call or customer data writes',candidateCount:candidates.length,counts:{top:slot('top').length,bottom:slot('bottom').length,shoes:slot('shoes').length},combinationsTested:tested,validCombinations:valid,capUzs:3150000,withinBudgetCombinationExists:!!cheapest&&cheapest.spendUzs<=3150000,cheapest};
fs.writeFileSync('docs/reports/readiness-evidence/old-money-budget-feasibility.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
