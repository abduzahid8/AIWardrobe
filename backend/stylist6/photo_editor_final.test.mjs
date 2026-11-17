import test from 'node:test';
import assert from 'node:assert/strict';
import {
  PHOTO_EDITOR_FINAL_VERSION,anonymousPhotoEditorialOptions,
  validatePhotoEditorialChoice,selectPhotoEditorialDeck,
} from './photo_editor_final.mjs';
const looks=Array.from({length:5},(_,i)=>({
  id:'SECRET-CLIENT-SKU-'+i,
  items:['top','outerwear','bottom','shoes'].map((slot,j)=>({
    id:'PRIVATE-'+i+'-'+j,name:'PRIVATE USER GARMENT '+i,
    macroCategory:slot,subtype:['dress-shirt','blazer','tailored-trouser','oxford'][j],
    imageUrl:'https://private.example/secret-abc.jpg',
  })),
}));
const reviews=looks.map((x,i)=>({
  candidateId:x.id,reject:false,
  slotColors:{top:'white',outerwear:['navy','grey','brown','olive','black'][i],bottom:'grey',shoes:'black'},
  slotObservations:{top:'Long-sleeve crisp shirt',outerwear:'Structured tailored wool',
    bottom:'Clean drape trousers',shoes:'Refined leather sole'},
  relationships:[{kind:'outfit',observation:'The shirt and wool blazer balance well'}],
  visualCohesion:.92,styleFit:.86,occasionFit:.9,formalityFit:.86,confidence:.95,
}));
const result=(orderedLooks)=>Response.json({
  choices:[{finish_reason:'stop',message:{content:JSON.stringify({orderedLooks})}}],
});
test('photo-grounded editorial options preserve evidence but never disclose private SKU/URLs',()=>{
  assert.match(PHOTO_EDITOR_FINAL_VERSION,/photo-grounded/);
  const x=anonymousPhotoEditorialOptions(looks,reviews);
  assert.equal(x.length,5);
  assert.equal(x[0].token,'L00');
  assert.equal(x[0].items[1].photoColor,'navy');
  assert.equal(x[0].items[1].photoObservation,'Structured tailored wool');
  assert.equal(x[0].photoAssessment.cohesion,.92);
  assert.doesNotMatch(JSON.stringify(x),/SECRET|PRIVATE|client|private\.example/);
});
test('only already-approved tokens, never hallucinated or duplicated, enter final deck',()=>{
  const a=anonymousPhotoEditorialOptions(looks,reviews);
  assert.deepEqual(validatePhotoEditorialChoice({orderedLooks:['L04','L00']},a),['L04','L00']);
  assert.deepEqual(validatePhotoEditorialChoice({orderedLooks:['L01','L01']},a),[]);
  assert.deepEqual(validatePhotoEditorialChoice({orderedLooks:['L99']},a),[]);
  assert.deepEqual(validatePhotoEditorialChoice({orderedLooks:[]},a),[]);
  assert.equal(anonymousPhotoEditorialOptions(looks,[{...reviews[0],reject:true}]).length,0);
});
test('the final director may choose two truly distinct photo-approved looks instead of padding three',async()=>{
  let calls=0;
  const r=await selectPhotoEditorialDeck({
    looks,critiques:reviews,context:{style:'classic',occasion:'Formal',brief:'Wedding'},
    apiKey:'nvapi-a-test-safe-key',fetchImpl:async(_,opts)=>{
      calls++;
      const body=JSON.parse(opts.body);
      assert.equal(body.response_format.type,'json_schema');
      assert.equal(body.response_format.json_schema.schema.properties.orderedLooks.maxItems,3);
      assert.match(body.messages[0].content,/Long-sleeve crisp shirt/);
      assert.doesNotMatch(opts.body,/SECRET|PRIVATE|private\.example/);
      return result(['L02','L00']);
    },
  });
  assert.equal(calls,1);
  assert.equal(r.status,'selected');
  assert.deepEqual(r.tokens,['L02','L00']);
});
test('provider failure, duplicate, invented SKU and missing approvals fail safely',async()=>{
  const key='nvapi-a-test-safe-key';
  assert.equal((await selectPhotoEditorialDeck({looks:looks.slice(0,2),
    critiques:reviews,apiKey:key})).status,'not_enough_approved');
  assert.equal((await selectPhotoEditorialDeck({looks,critiques:reviews,
    apiKey:'bad-key'})).status,'not_configured');
  const fake=await selectPhotoEditorialDeck({looks,critiques:reviews,apiKey:key,
    fetchImpl:async()=>result(['L99','L01'])});
  assert.equal(fake.status,'invalid_selection');
  assert.deepEqual(fake.tokens,[]);
  const down=await selectPhotoEditorialDeck({looks,critiques:reviews,apiKey:key,
    fetchImpl:async()=>Response.json({error:'fail'},{status:502})});
  assert.equal(down.status,'provider_unavailable');
});
