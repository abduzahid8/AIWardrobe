/** Apply only the verified budget changes to a freshly downloaded production function. */
import fs from 'node:fs';import ts from 'typescript';
const folder='/tmp/aiwardrobe-fx-generation/supabase/functions/',target=folder+'generate-outfits/index.ts';
let remote=fs.readFileSync(target,'utf8');const local=fs.readFileSync('backend/supabase/functions/generate-outfits/index.ts','utf8');
const ast=source=>ts.createSourceFile('x.ts',source,ts.ScriptTarget.Latest,true);
const fn=(source,name)=>{const file=ast(source);return file.statements.find(n=>ts.isFunctionDeclaration(n)&&n.name?.text===name)?.getText(file);};
for(const name of ['parseBudgetFromText','convertShopAmountEdge','shopPriceOf','outfitShopSpend','budgetInCurrencyEdge','fitOutfitToBudget']){const old=fn(remote,name),updated=fn(local,name);if(!old||!updated)throw Error('Missing function '+name);remote=remote.replace(old,updated);}
const file=ast(remote),constant=file.statements.find(n=>ts.isVariableStatement(n)&&n.declarationList.declarations.some(d=>d.name.getText(file)==='EDGE_UNITS_PER_USD'));if(!constant)throw Error('Static FX marker missing');remote=remote.replace(constant.getFullText(file),'\n');
remote="import { budgetCandidatesFromText as budgetCandidatesFromTextEdge } from '../_shared/outfitBudgetText.ts'\nimport { type BudgetFxContext, BudgetFxError, budgetFxContext, fitsBudgetAmount, referenceAmount } from '../_shared/budgetFx.ts'\nimport { currentBudgetRates, budgetRatesById } from '../_shared/budgetFxCache.ts'\n"+remote;
function block(source,start,end){const a=source.indexOf(start),b=source.indexOf(end,a);if(a<0||b<0)throw Error('Missing block '+start);return source.slice(a,b);}
remote=remote.replace(block(remote,'    let parsedBudget:','    const preferences = parsedBudget'),block(local,'    let parsedBudget:','    const preferences = parsedBudget'));
remote=remote.replace(block(remote,'        // Catalog rows are priced in UZS while','        // ── 3d.'),block(local,"        // Keep the user's original cap currency",'        // ── 3d.'));
const replacements=[
 ['shopPriceOf(i, preferences.maxShopBudget.currency)','shopPriceOf(i, preferences.maxShopBudget.currency, preferences.maxShopBudget.fx)'],
 ['outfitShopSpend(o.items || [], itemMap, budgetCap.currency)','outfitShopSpend(o.items || [], itemMap, budgetCap.currency, budgetCap.fx)'],
 ['spend > budgetCap.amount + 0.009','!fitsBudgetAmount(spend, budgetCap.amount)'],
 ['outfitShopSpend(parts, undefined, budgetCap.currency) > budgetCap.amount + 0.009','!fitsBudgetAmount(outfitShopSpend(parts, undefined, budgetCap.currency, budgetCap.fx), budgetCap.amount)'],
 ['shopPriceOf(item, parsedBudget!.currency)','shopPriceOf(item, parsedBudget!.currency, parsedBudget!.fx)'],
 ['minimumShopSpend > parsedBudget.amount + 0.009','!fitsBudgetAmount(minimumShopSpend, parsedBudget.amount)'],
 ['outfitBudgetFeedback(aiOutfits, itemMap, preferences?.maxShopBudget, convertShopAmountEdge)','outfitBudgetFeedback(aiOutfits, itemMap, preferences?.maxShopBudget, (amount, from, to) => convertShopAmountEdge(amount, from, to, preferences?.maxShopBudget?.fx))'],
 ['outfitShopSpend(outfit.items, itemMap, preferences.maxShopBudget.currency) <= preferences.maxShopBudget.amount + 0.009','fitsBudgetAmount(outfitShopSpend(outfit.items, itemMap, preferences.maxShopBudget.currency, preferences.maxShopBudget.fx), preferences.maxShopBudget.amount)'],
 ["source: 'ai', layered, provider: aiProviderUsed, providerAttempts: providerTrace", "source: 'ai', layered, provider: aiProviderUsed, providerAttempts: providerTrace, budgetReference: requestFx ? {id: requestFx.snapshot.id, asOf: requestFx.snapshot.asOf, source: 'CBU'} : undefined"],
 ["        console.error('generate-outfits error:', error)","        if (error instanceof BudgetFxError) return new Response(JSON.stringify({success:false,outfits:[],error:'budget_rates_unavailable'}), {status:503,headers:{...corsHeaders,'Content-Type':'application/json'}})\n        console.error('generate-outfits error:', error)"]
];
for(const [from,to] of replacements){if(!remote.includes(from))throw Error('Missing scoped edit '+from);remote=remote.replace(from,to);}
remote=remote.replaceAll('budget: parsedBudget','budget: parsedBudget ? {amount: parsedBudget.amount, currency: parsedBudget.currency} : null');
const marker='    const rateLimitResponse = await enforceRateLimit(req,';if(!remote.includes(marker))throw Error('Missing method guard anchor');
remote=remote.replace(marker,"    if (req.method !== 'POST') return new Response(JSON.stringify({success:false,outfits:[],error:'method_not_allowed'}), {status:405,headers:{...corsHeaders,'Content-Type':'application/json'}})\n\n"+marker);
fs.writeFileSync(target,remote);
for(const name of ['budgetFx.ts','budgetFxCache.ts','outfitBudgetText.ts'])fs.copyFileSync('backend/supabase/functions/_shared/'+name,folder+'_shared/'+name);
console.log('Staged reference snapshot, shared parser, budget comparisons and POST guard only');
