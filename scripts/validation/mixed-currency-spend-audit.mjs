import {writeFile} from 'node:fs/promises';import {loadBudgetHelpers,currentReference} from './load-outfit-budget-helpers.mjs';
const {core,client,edge}=await loadBudgetHelpers(),{now,url,snapshot,fx}=await currentReference(core);
const items=[{id:'shop_local',price:100000,currency:'UZS',isShopItem:true},{id:'shop_import',price:20,currency:'USD',isShopItem:true}];
const expectedUZS=100000+20*snapshot.uzsPerUnit.USD,capUZS=200000;
const observed={client:client.outfitShopSpend(items,undefined,'UZS',fx),edge:edge.outfitShopSpend(items,new Map(items.map(x=>[x.id,x])),'UZS',fx)};
const passed=Math.abs(observed.client-expectedUZS)<1e-7&&Math.abs(observed.edge-expectedUZS)<1e-7;
const report={measuredAt:new Date(now).toISOString(),source:url,snapshot,items,capUZS,expectedUZS,observed,passed,scope:'Actual client/Edge helpers with pinned CBU reference, not checkout prices, hosted generated output or capacity'};
await writeFile('docs/reports/readiness-evidence/mixed-currency-spend-reference.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));if(!passed)process.exitCode=1;
