/** @jest-environment node */
const fs=require('fs'),path=require('path'),ts=require('typescript');const api={};
new Function('exports',ts.transpileModule(fs.readFileSync(path.resolve(__dirname,'../../backend/supabase/functions/_shared/outfitBudgetSelection.ts'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText)(api);
const {strictestOutfitBudget}=api;const usd=amount=>({amount,currency:'USD'});const same=budget=>budget;
test('prompt cannot raise the selected budget',()=>expect(strictestOutfitBudget(usd(250),usd(1000),same)).toEqual(usd(250)));
test('prompt can tighten the selected budget',()=>expect(strictestOutfitBudget(usd(250),usd(100),same)).toEqual(usd(100)));
test('cross-currency comparison uses the selected currency before choosing the stricter amount',()=>{
 const convert=jest.fn((budget,currency)=>({amount:budget.amount*12600,currency}));
 expect(strictestOutfitBudget({amount:3150000,currency:'UZS'},usd(1000),convert)).toEqual({amount:3150000,currency:'UZS'});
 expect(convert).toHaveBeenCalledWith(usd(1000),'UZS');
});
test('zero is an explicit no-shopping budget, not absence of a limit',()=>expect(strictestOutfitBudget(usd(0),usd(250),same)).toEqual(usd(0)));
test.each([{amount:-1,currency:'USD'},{amount:NaN,currency:'USD'},{amount:Infinity,currency:'USD'},{amount:100,currency:'UNKNOWN'},{amount:'100',currency:'USD'}])('invalid selected budget is rejected: %j',budget=>expect(()=>strictestOutfitBudget(budget,null,same)).toThrow('invalid_outfit_budget'));
test('unavailable currency conversion is rejected instead of comparing unlike amounts',()=>expect(()=>strictestOutfitBudget(usd(250),{amount:100,currency:'EUR'},()=>null)).toThrow('invalid_outfit_budget'));
test('either budget can be absent without mutating the supplied limit',()=>{const selected=usd(250);expect(strictestOutfitBudget(selected,null,same)).toEqual(selected);expect(strictestOutfitBudget(null,usd(100),same)).toEqual(usd(100));expect(strictestOutfitBudget(null,null,same)).toBeNull();expect(selected).toEqual(usd(250));});
