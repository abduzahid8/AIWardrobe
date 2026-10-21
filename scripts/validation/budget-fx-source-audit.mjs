import {writeFile} from 'node:fs/promises';import {loadBudgetHelpers,currentReference} from './load-outfit-budget-helpers.mjs';
const {core,client,edge}=await loadBudgetHelpers(),{now,url,payload,snapshot,fx}=await currentReference(core);
const quote=currency=>{const row=payload.find(row=>row.Ccy===currency);return Number(row.Rate)/Number(row.Nominal);};
const expected=100*quote('EUR')/quote('USD'),item={id:'shop_euro',price:100,currency:'EUR',isShopItem:true};
const clientCostUSD=client.itemShopPrice(item,'USD',fx),edgeCostUSD=edge.shopPriceOf(item,'USD',fx);
const passed=Math.abs(clientCostUSD-expected)<1e-8&&Math.abs(edgeCostUSD-expected)<1e-8;
const report={measuredAt:new Date(now).toISOString(),source:url,snapshot,example:{listedPrice:100,listedCurrency:'EUR',budgetUSD:100,referenceCostUSD:expected,clientCostUSD,edgeCostUSD,clientPassesBudget:core.fitsBudgetAmount(clientCostUSD,100),edgePassesBudget:core.fitsBudgetAmount(edgeCostUSD,100)},passed,scope:'Actual monetary helpers with a pinned published CBU reference; not hosted generation, retailer checkout/bank fees or capacity certification'};
await writeFile('docs/reports/readiness-evidence/budget-fx-reference-after.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));if(!passed)process.exitCode=1;
