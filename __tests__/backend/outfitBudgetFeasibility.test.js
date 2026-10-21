/** @jest-environment node */
const fs=require('fs'),path=require('path'),ts=require('typescript'),api={};
new Function('exports',ts.transpileModule(fs.readFileSync(path.resolve(__dirname,'../../backend/supabase/functions/_shared/outfitBudgetFeasibility.ts'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText)(api);
const slots=['top','bottom','shoes'];const items=[{macroCategory:'top',cost:100},{macroCategory:'bottom',cost:100},{macroCategory:'shoes',cost:100}];
const cost=item=>item.cost;
test('lower bound adds the cheapest known candidate in each required slot',()=>expect(api.minimumSlotSpend([...items,{macroCategory:'top',cost:20}],slots,cost)).toBe(220));
test('owned zero-cost candidates reduce the lower bound',()=>expect(api.minimumSlotSpend([...items,{macroCategory:'shoes',cost:0}],slots,cost)).toBe(200));
test('missing required slot leaves the result undecided',()=>expect(api.minimumSlotSpend(items,['top','bottom','shoes','outerwear'],cost)).toBeNull());
test.each([null,NaN,Infinity,-1])('unknown or invalid cost %s cannot establish impossibility',costValue=>expect(api.minimumSlotSpend([...items,{macroCategory:'top',cost:costValue}],slots,cost)).toBeNull());
test('unrequired accessory prices do not affect the bound',()=>expect(api.minimumSlotSpend([...items,{macroCategory:'accessory',cost:null}],slots,cost)).toBe(300));
test('zero-cost complete pool is a known zero, not undecided',()=>expect(api.minimumSlotSpend(items.map(item=>({...item,cost:0})),slots,cost)).toBe(0));
