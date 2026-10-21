/** Execute actual monetary helpers against unknown shop prices. No provider calls. */
import {writeFile} from 'node:fs/promises';
import {loadBudgetHelpers} from './load-outfit-budget-helpers.mjs';
const {client,edge}=await loadBudgetHelpers();
const prices=[['missing',undefined],['null',null],['zero',0],['negative',-1],['NaN',NaN],['Infinity',Infinity]];
const checks=prices.map(([label,price])=>{
 const items=[{id:'shop_unknown',isShopItem:true,price,currency:'USD'},{id:'shop_known',isShopItem:true,price:20,currency:'USD'}];
 const c=client.outfitShopSpend(items,undefined,'USD'),e=edge.outfitShopSpend(items,undefined,'USD');
 return{label,client:{finite:Number.isFinite(c),passes100USD:c<=100},edge:{finite:Number.isFinite(e),passes100USD:e<=100},passed:c===Infinity&&e===Infinity};
});
const report={date:'2026-10-20',scope:'Actual client and Edge helpers with synthetic invalid or absent shop prices, not hosted generated output',checks,passed:checks.every(x=>x.passed)};
const filename='unpriced-shop-budget-after-fx.json';
await writeFile('docs/reports/readiness-evidence/'+filename,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));if(!report.passed)process.exitCode=1;
