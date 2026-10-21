/** Read and execute the actual pure monetary modules, with explicit reference data. */
import {readFile} from 'node:fs/promises';import ts from 'typescript';
const compile=source=>ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
export async function loadBudgetHelpers(){
 const core={},text={},client={};
 Function('exports',compile(await readFile('backend/supabase/functions/_shared/budgetFx.ts','utf8')))(core);
 Function('exports',compile(await readFile('backend/supabase/functions/_shared/outfitBudgetText.ts','utf8')))(text);
 Function('require','exports',compile(await readFile('src/utils/outfitBudget.ts','utf8')))(name=>{if(name.endsWith('/budgetFx'))return core;if(name.endsWith('/outfitBudgetText'))return text;throw Error('Unexpected monetary dependency');},client);
 const source=await readFile('backend/supabase/functions/generate-outfits/index.ts','utf8'),ast=ts.createSourceFile('edge.ts',source,ts.ScriptTarget.Latest,true);
 const names=['isShopPricedItem','convertShopAmountEdge','shopPriceOf','outfitShopSpend'];const selected=ast.statements.filter(n=>ts.isFunctionDeclaration(n)&&names.includes(n.name?.text));if(selected.length!==names.length)throw Error('Spend function extraction incomplete');
 const edge=Function('referenceAmount',compile(selected.map(n=>n.getText(ast)).join('\n'))+';return{shopPriceOf,outfitShopSpend};')(core.referenceAmount);
 return{core,client,edge};
}
export async function currentReference(core){
 const now=Date.now(),url=`https://cbu.uz/uz/arkhiv-kursov-valyut/json/all/${core.uzbekDateKey(now)}/`;const r=await fetch(url,{signal:AbortSignal.timeout(10000),redirect:'error'});if(!r.ok)throw Error('Official reference unavailable');const payload=await r.json(),snapshot=core.parseCbuBudgetRates(payload,now);return{now,url,payload,snapshot,fx:core.budgetFxContext(snapshot,now)};
}
