/** @jest-environment node */
const {referenceFx}=require('../fixtures/budgetFx');
const {referenceAmount,fitsBudgetAmount}=require('../../backend/supabase/functions/_shared/budgetFx');
const fs = require('fs'), ts = require('typescript');
const source = fs.readFileSync('backend/supabase/functions/generate-outfits/index.ts', 'utf8');
const ast = ts.createSourceFile('edge.ts', source, ts.ScriptTarget.Latest, true);
const names = ['isShopPricedItem', 'convertShopAmountEdge', 'shopPriceOf', 'outfitShopSpend', 'fitOutfitToBudget'];
const selected = ast.statements.filter(node => ts.isFunctionDeclaration(node) && names.includes(node.name?.text) || ts.isVariableStatement(node) && node.declarationList.declarations.some(d => d.name.getText(ast) === 'EDGE_UNITS_PER_USD'));
const compiled = ts.transpileModule(selected.map(node => node.getText(ast)).join('\n'), {compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
const api = Function('Deno','referenceAmount','fitsBudgetAmount', compiled + ';return {outfitShopSpend,fitOutfitToBudget};')({env:{get:()=>undefined}},referenceAmount,fitsBudgetAmount);
const local = {id:'shop_local',price:100000,currency:'UZS',isShopItem:true,macroCategory:'top'};
const imported = {id:'shop_import',price:20,currency:'USD',isShopItem:true,macroCategory:'bottom'};
test('Edge sums mixed units in the cap currency', () => {
 expect(api.outfitShopSpend([local, imported],undefined,'UZS',referenceFx)).toBeCloseTo(340894.4);
 expect(api.outfitShopSpend([imported,local],undefined,'UZS',referenceFx)).toBeCloseTo(340894.4);
});
test('authoritative source currency, amount and ownership defeat forged model attributes', () => {
 const map=new Map([[imported.id,imported]]);
 expect(api.outfitShopSpend([{...imported,price:1,currency:'UZS',isShopItem:false,isWardrobe:true}],map,'UZS',referenceFx)).toBeCloseTo(240894.4);
});
test('unknown explicit currency cannot pass the cap', () => expect(api.outfitShopSpend([{...imported,currency:'ZZZ'}],undefined,'UZS',referenceFx)).toBe(Infinity));
test('substitution compares converted costs and keeps the original listed quote', () => {
 const cheaper={...imported,id:'shop_cheaper',price:5};
 const result=api.fitOutfitToBudget([local,imported],[local,imported,cheaper],{amount:200000,currency:'UZS',fx:referenceFx});
 expect(api.outfitShopSpend(result,undefined,'UZS',referenceFx)).toBeCloseTo(160223.6);
 expect(result[1]).toMatchObject({id:'shop_cheaper',price:5,currency:'USD'});
 expect(imported.price).toBe(20);
});

test.each([undefined,null,0,-1,NaN,Infinity])('Edge rejects unverified shop price %s under a cap but keeps owned pieces free',price=>{
 expect(api.outfitShopSpend([{...imported,price}],undefined,'USD')).toBe(Infinity);
 expect(api.outfitShopSpend([{...imported,price,isShopItem:false,isWardrobe:true}],undefined,'USD')).toBe(0);
});
