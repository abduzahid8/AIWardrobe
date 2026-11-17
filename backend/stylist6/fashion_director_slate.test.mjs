import test from 'node:test';
import assert from 'node:assert/strict';
import {
  FASHION_SLATE_VERSION,anonymousFashionOptions,
  validateFashionChoice,selectFashionSlate,
} from './fashion_director_slate.mjs';
const items=Array.from({length:12},(_,n)=>({
  id:'PRIVATE-USER-SKU-'+n,direction:n<4?'ai':'house',
  items:[
    {slot:'top',subtype:'dress-shirt',name:'PRIVATE GARMENT',photo:{color:'cream'}},
    {slot:'outerwear',subtype:'blazer',photo:{texture:'tweed'}},
    {slot:'bottom',subtype:'tailored-trouser',photo:{color:'navy'}},
    {slot:'shoes',subtype:'loafer',photo:{footwearPurpose:'dress'}},
  ],
}));
const result=orderedLooks=>Response.json({choices:[{
  finish_reason:'stop',message:{content:JSON.stringify({orderedLooks})},
}]});
test('sanitized photo-informed candidates never disclose raw SKU, merchant or garment names',()=>{
  assert.equal(typeof FASHION_SLATE_VERSION,'string');
  const options=anonymousFashionOptions(items);
  assert.equal(options.length,12);
  assert.equal(options[0].token,'L00');
  assert.equal(options[0].items[0].photographed.color,'cream');
  assert.equal(JSON.stringify(options).includes('PRIVATE-USER-SKU'),false);
  assert.equal(JSON.stringify(options).includes('PRIVATE GARMENT'),false);
});
test('validation refuses invented, duplicate or incomplete selections',()=>{
  const options=anonymousFashionOptions(items);
  assert.deepEqual(validateFashionChoice({orderedLooks:['L00','L01','L02','L03','L04','L05']},options),
    ['L00','L01','L02','L03','L04','L05']);
  assert.deepEqual(validateFashionChoice({orderedLooks:['L00','L01','L02']},options),[]);
  assert.deepEqual(validateFashionChoice({orderedLooks:['L00','L01','L02','L03','L04','L99']},options),[]);
  assert.deepEqual(validateFashionChoice({orderedLooks:['L00','L01','L02','L03','L04','L04']},options),[]);
});
test('one bounded AI call chooses six actual complete-photo options',async()=>{
  const key='nvapi-test-key-0123456789ABC';
  let calls=0;
  const r=await selectFashionSlate({
    options:items,context:{style:'classic',occasion:'Work',brief:'Elegant day',
      preferences:{avoidColors:['pink']}},
    apiKey:key,fetchImpl:async(_,init)=>{
      calls++;
      const body=JSON.parse(init.body);
      assert.equal(body.response_format.type,'json_schema');
      assert.equal(body.response_format.json_schema.schema.properties.orderedLooks.maxItems,6);
      assert.equal(init.method,'POST');
      assert.ok(init.signal);
      assert.ok(!init.body.includes('PRIVATE-USER-SKU'));
      assert.ok(!init.body.includes('PRIVATE GARMENT'));
      assert.ok(!init.body.includes(key));
      return result(['L07','L04','L02','L01','L09','L05']);
    },
  });
  assert.equal(calls,1);
  assert.equal(r.status,'selected');
  assert.equal(r.providerCalls,1);
  assert.deepEqual(r.tokens,['L07','L04','L02','L01','L09','L05']);
});
test('fail closed on unsupported model, fabricated choice or provider failure',async()=>{
  const key='nvapi-test-key-0123456789ABC';
  assert.equal((await selectFashionSlate({options:items,apiKey:''})).status,'not_configured');
  assert.equal((await selectFashionSlate({options:items.slice(0,3),apiKey:key})).status,
    'insufficient_candidates');
  const forged=await selectFashionSlate({options:items,apiKey:key,
    fetchImpl:async()=>result(['L00','L01','L02','L03','L04','L99'])});
  assert.equal(forged.status,'invalid_selection');
  assert.deepEqual(forged.tokens,[]);
  const offline=await selectFashionSlate({options:items,apiKey:key,
    fetchImpl:async()=>Response.json({error:'down'},{status:503})});
  assert.equal(offline.status,'provider_unavailable');
});
