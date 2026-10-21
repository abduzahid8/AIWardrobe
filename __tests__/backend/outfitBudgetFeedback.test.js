/** @jest-environment node */
const fs=require('fs'), path=require('path'), ts=require('typescript');
const api={};new Function('exports',ts.transpileModule(fs.readFileSync(path.resolve(__dirname,'../../backend/supabase/functions/_shared/outfitBudgetFeedback.ts'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText)(api);
const {outfitBudgetFeedback}=api;
const rows=new Map([
 ['shop_top',{id:'shop_top',macroCategory:'top',price:120,currency:'USD',isShopItem:true}],
 ['shop_shoes',{id:'shop_shoes',macroCategory:'shoes',price:150,currency:'USD',isShopItem:true}],
 ['shop_cheap',{id:'shop_cheap',macroCategory:'shoes',price:40,currency:'USD',isShopItem:true}],
 ['owned',{id:'owned',macroCategory:'bottom',price:999,isWardrobe:true}],
]);
const outfit=ids=>[{items:ids.map(id=>({id}))}];
test('correction uses authoritative totals and keeps required slots and budget explicit',()=>{
 const output=outfitBudgetFeedback(outfit(['shop_top','shop_shoes','owned']),rows,{amount:250,currency:'USD'});
 expect(output).toContain('Previous catalog totals: [270]');expect(output).toContain('Hard limit 250 USD');expect(output).toContain('shop_cheap');expect(output).toContain('do not omit a required slot');
});
test('within-budget selections do not trigger correction and input is unchanged',()=>{
 const original=outfit(['shop_top','shop_cheap','owned']);const snapshot=JSON.stringify(original);
 expect(outfitBudgetFeedback(original,rows,{amount:250,currency:'USD'})).toBe('');expect(JSON.stringify(original)).toBe(snapshot);
});
test('unknown IDs and mismatched currency never produce an invented spending total',()=>{
 expect(outfitBudgetFeedback(outfit(['missing','shop_shoes']),rows,{amount:1,currency:'USD'})).toBe('');
 expect(outfitBudgetFeedback(outfit(['shop_top']),rows,{amount:1,currency:'EUR'})).toBe('');
});
test('guidance includes one complete costed combination, without selecting it for the model',()=>{
 const result=outfitBudgetFeedback(outfit(['shop_top','shop_shoes','owned']),rows,{amount:250,currency:'USD'});
 expect(result).toContain('total 160 USD');expect(result).toContain('only if it also meets every style and occasion rule');
 expect(result).toContain('"id":"owned","slot":"bottom","cost":0');
});
test('no feasible combination is claimed when all complete combinations exceed the cap',()=>{
 const result=outfitBudgetFeedback(outfit(['shop_top','shop_shoes','owned']),rows,{amount:100,currency:'USD'});
 expect(result).not.toContain('A complete candidate combination');expect(result).toContain('Hard limit 100 USD');
});

test('correction never recommends cheaper items rejected by the requested style',()=>{
 const pool=new Map([...rows,['bad_tee',{id:'bad_tee',macroCategory:'top',price:1,currency:'USD',isShopItem:true}]]);
 const output=outfitBudgetFeedback(outfit(['shop_top','shop_shoes','owned']),pool,{amount:250,currency:'USD'},undefined,item=>item.id!=='bad_tee');
 expect(output).not.toContain('bad_tee');expect(output).toContain('total 160 USD');
});
test('filtering all candidates for a mandatory slot never claims a complete combination',()=>{
 const output=outfitBudgetFeedback(outfit(['shop_top','shop_shoes','owned']),rows,{amount:250,currency:'USD'},undefined,item=>item.macroCategory!=='top');
 expect(output).not.toContain('A complete candidate combination');
 expect(output).not.toContain('"id":"shop_top"');
});
