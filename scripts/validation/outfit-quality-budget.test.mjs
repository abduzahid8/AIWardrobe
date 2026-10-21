import test from 'node:test';
import assert from 'node:assert/strict';
import {catalogOutfitSpends} from './outfit-quality-budget.mjs';
const rows=new Map([['a',{price:200,currency:'UZS'}],['b',{price:400,currency:'UZS'}]]);
test('two individually affordable outfits do not become one over-budget outfit',()=>{
 const spends=catalogOutfitSpends([{items:[{id:'shop_a'}]},{items:[{id:'shop_a'}]}],rows);
 assert.deepEqual(spends,[200,200]);assert.equal(spends.some(spend=>spend>300),false);
});
test('one genuinely excessive outfit is still detectable',()=>{
 assert.deepEqual(catalogOutfitSpends([{items:[{id:'shop_a'}]},{items:[{id:'shop_b'}]}],rows),[200,400]);
});
test('unknown or invalid price cannot be reported as zero spend',()=>{
 assert.deepEqual(catalogOutfitSpends([{items:[{id:'missing'}]},{items:[]}],rows),[null,null]);
});
