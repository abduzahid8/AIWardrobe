import test from 'node:test';
import assert from 'node:assert/strict';
import {sourceBoundCompositionPool,validateSkuCompositions,composeCatalogSkuLooks,
  sourceVerifyComposedOutfits,AI_SKU_COMPOSER_VERSION} from './sku_composer_ai.mjs';
const slots=['top','outerwear','bottom','shoes'];
const items=Array.from({length:6},(_,i)=>slots.map((slot,j)=>({
  id:'PRIVATE_SKU_'+i+'_'+j,macroCategory:slot,isShopItem:true,
  subtype:['dress-shirt','blazer','tailored-trouser','oxford'][j],
  name:'PUBLIC '+['shirt','blazer','trouser','oxford'][j]+' '+i,
  color:['white','navy','grey','brown','olive','cream'][i],
  price:35+i*4+j,currency:'USD',
  imageUrl:'https://images.zara.net/photo'+i+j+'.jpg',
  shopUrl:'https://www.zara.com/us/item'+i+j,
})));
const looks=items.map((x,i)=>({id:'look'+i,fashion:{score:.95-.015*i},
  items:x}));
const validAnswer=pool=>({outfits:Array.from({length:3},(_,i)=>Object.fromEntries(
  slots.map(slot=>[slot,pool.find(x=>x.slot===slot&&x.item===items[i][slots.indexOf(slot)]).token])
))});
const slotAnswer=pool=>{
  const {outfits}=validAnswer(pool);
  return {slotPicks:Object.fromEntries(slots.map(slot=>
    [slot,outfits.map(look=>look[slot])]))};
};
const result=answer=>Response.json({choices:[{
  finish_reason:'stop',message:{content:JSON.stringify(answer)},
}]});
test('SKU Composer sees multiple real merchandise menus, never identity or private image URLs',()=>{
  assert.match(AI_SKU_COMPOSER_VERSION,/sku-composer/);
  const p=sourceBoundCompositionPool(looks);
  assert.ok(p.length>=16);
  assert.ok(slots.every(s=>p.filter(x=>x.slot===s).length>=4));
  const anonymous=JSON.stringify(p.map(x=>x.public));
  assert.doesNotMatch(anonymous,/PRIVATE_SKU|images\.zara|www\.zara|shopUrl|imageUrl/);
  assert.match(anonymous,/PUBLIC shirt/);
});
test('AI outputs valid exactly source-bound full looks with no reused items or wrong slots',()=>{
  const p=sourceBoundCompositionPool(looks);
  const answer=validAnswer(p);
  const actual=validateSkuCompositions(answer,p,{layerRequired:true});
  assert.equal(actual.length,3);
  assert.ok(actual.flat().every(x=>x.id.startsWith('PRIVATE_SKU_')));
  assert.equal(new Set(actual.flat().map(x=>x.id)).size,12);
  assert.deepEqual(validateSkuCompositions({...answer,outfits:[
    {...answer.outfits[0],shoes:'P999'},...answer.outfits.slice(1)]},
    p,{layerRequired:true}),[]);
  assert.deepEqual(validateSkuCompositions({...answer,outfits:[
    {...answer.outfits[0],shoes:answer.outfits[0].top},...answer.outfits.slice(1)]},
    p,{layerRequired:true}),[]);
  assert.deepEqual(validateSkuCompositions({...answer,outfits:[
    answer.outfits[0],{...answer.outfits[1],top:answer.outfits[0].top},
    answer.outfits[2]]},p,{layerRequired:true}),[]);
  assert.deepEqual(validateSkuCompositions({...answer,outfits:[
    {...answer.outfits[0],outerwear:'none'},...answer.outfits.slice(1)]},
    p,{layerRequired:true}),[]);
});
test('model independently composes three new real SKU combinations, no product identities leaked',async()=>{
  const pool=sourceBoundCompositionPool(looks);
  let called=0;
  const r=await composeCatalogSkuLooks({
    pool,layerRequired:true,context:{style:'classic',occasion:'Work',brief:'Office',
      preferences:{fitPreference:'relaxed'}},apiKey:'nvapi-test-secret-123',
    fetchImpl:async(_,init)=>{
      called++;
      const body=JSON.parse(init.body);
      assert.equal(body.response_format.type,'json_schema');
      const slotPicks=body.response_format.json_schema.schema.properties.slotPicks;
      assert.equal(slotPicks.properties.top.maxItems,3);
      assert.equal(slotPicks.properties.top.uniqueItems,true);
      assert.equal(slotPicks.properties.bottom.uniqueItems,true);
      assert.equal(slotPicks.properties.shoes.uniqueItems,true);
      assert.doesNotMatch(init.body,/PRIVATE_SKU|images\.zara|www\.zara/);
      assert.match(body.messages[0].content,/CREATE THREE distinct beautiful complete outfits/);
      return result(slotAnswer(pool));
    },
  });
  assert.equal(called,1);
  assert.equal(r.status,'composed');
  assert.equal(r.looks.length,3);
  assert.equal(new Set(r.looks.flat().map(x=>x.id)).size,12);
});
test('composed choices rerun combination-level hard gate, budget and scorer; never invent approval',async()=>{
  const pool=sourceBoundCompositionPool(looks);
  const raw=validateSkuCompositions(validAnswer(pool),pool,{layerRequired:true});
  let gates=0;
  const request={budget:{amount:200,currency:'USD'}};
  const scorer={rankLook:()=>({score:.90,dimensions:{palette:.9}})};
  const accepted=await sourceVerifyComposedOutfits({
    looks:raw,request,requiredIds:[],scorer,hardGate:async()=>{
      gates++;return {ok:true};
    },
    priceToBudgetCurrency:x=>x.price,
  });
  assert.equal(accepted.length,3);
  assert.equal(gates,3);
  assert.ok(accepted.every(l=>l.id.startsWith('s6_ai_')&&l.briefId.startsWith('ai_composed_')));
  assert.equal(new Set(accepted.flatMap(l=>l.items.map(x=>x.id))).size,12);
  assert.equal((await sourceVerifyComposedOutfits({
    looks:raw,request:{budget:{amount:10,currency:'USD'}},
    scorer,hardGate:async()=>({ok:true}),priceToBudgetCurrency:x=>x.price,
  })).length,0,'a new combo cannot exceed exact user budget');
  assert.equal((await sourceVerifyComposedOutfits({
    looks:raw,request,scorer,hardGate:async()=>({ok:false}),
    priceToBudgetCurrency:x=>x.price,
  })).length,0,'source-item validity never bypasses full combination hard gate');
  const short=await sourceVerifyComposedOutfits({
    looks:raw,request,scorer,hardGate:async(x)=>{
      return {ok:x[0].id!==items[1][0].id};
    },priceToBudgetCurrency:x=>x.price,
  });
  assert.equal(short.length,2,'a bad AI outfit is declined, not converted to a synthetic success');
});
test('provider failure and missing inventory never become product claims',async()=>{
  const p=sourceBoundCompositionPool(looks);
  assert.equal((await composeCatalogSkuLooks({pool:p.slice(0,4),apiKey:'nvapi-test'}))
    .status,'insufficient_inventory');
  assert.equal((await composeCatalogSkuLooks({pool:p,apiKey:'bad'})).status,'not_configured');
  const offline=await composeCatalogSkuLooks({pool:p,apiKey:'nvapi-test',
    fetchImpl:async()=>Response.json({error:'bad'},{status:500})});
  assert.equal(offline.status,'provider_unavailable');
  assert.equal(offline.looks.length,0);
});

test('SKU Director sees real category alternatives outside old prebuilt ranker outfits',()=>{
  const additional=[
    {...items[0][0],id:'NEW-FINE-KNIT',name:'Fine Merino Crewneck',
      subtype:'fine-knit',color:'camel'},
    {...items[0][1],id:'NEW-CARDIGAN',name:'Merino Button Cardigan',
      subtype:'cardigan',color:'olive'},
    {...items[0][2],id:'NEW-CHINO',name:'Textured Cotton Chinos',
      subtype:'chino',color:'stone'},
    {...items[0][3],id:'NEW-LOAFER',name:'Suede Penny Loafers',
      subtype:'loafer',color:'burgundy'},
  ];
  const pool=sourceBoundCompositionPool(looks,{additionalItems:additional});
  for(const real of additional){
    assert.ok(pool.some(p=>p.item===real),
      'direct catalog garment '+real.id+' must reach AI even if NO legacy look used it');
  }
  assert.equal(pool.filter(p=>p.slot==='top').length,7);
  assert.ok(pool.every(p=>p.public.token===p.token));
  assert.doesNotMatch(JSON.stringify(pool.map(p=>p.public)),/NEW-FINE-KNIT|NEW-CARDIGAN/);
});


test('slot-column output assembles three user-facing real looks without repeating major items',()=>{
  const pool=sourceBoundCompositionPool(looks);
  const response=slotAnswer(pool);
  const actual=validateSkuCompositions(response,pool,{layerRequired:true});
  assert.equal(actual.length,3);
  assert.equal(new Set(actual.flat().map(i=>i.id)).size,12);
  assert.deepEqual(validateSkuCompositions({
    slotPicks:{...response.slotPicks,
      top:[response.slotPicks.top[0],response.slotPicks.top[0],response.slotPicks.top[2]]},
  },pool,{layerRequired:true}),[],'duplicate garment in a role never becomes a bought outfit');
  assert.deepEqual(validateSkuCompositions({
    slotPicks:{...response.slotPicks,
      outerwear:['none',...response.slotPicks.outerwear.slice(1)]},
  },pool,{layerRequired:true}),[],'required layer cannot be bypassed');
});
