import test from 'node:test';
import assert from 'node:assert/strict';
import {
  PHOTO_EDITOR_FINAL_VERSION,anonymousPhotoEditorialOptions,
  anonymousPhotoEditorialPortfolios,validatePhotoEditorialPortfolioChoice,selectPhotoEditorialDeck,
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
const verifiedMenu=[
  {portfolioToken:'F00',orderedLooks:['L00','L01','L02']},
  {portfolioToken:'F01',orderedLooks:['L02','L00']},
  {portfolioToken:'F02',orderedLooks:['L04']},
];
const result=portfolio=>Response.json({
  choices:[{finish_reason:'stop',message:{content:JSON.stringify({portfolio})}}],
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
  const menu=anonymousPhotoEditorialPortfolios(verifiedMenu,a);
  assert.deepEqual(validatePhotoEditorialPortfolioChoice({portfolio:'F01'},menu),
    {token:'F01',looks:['L02','L00']});
  assert.equal(validatePhotoEditorialPortfolioChoice({portfolio:'F99'},menu),null);
  assert.equal(validatePhotoEditorialPortfolioChoice({portfolio:['F01']},menu),null);
  assert.equal(anonymousPhotoEditorialPortfolios([
    {portfolioToken:'F00',orderedLooks:['L00','L00']},
    {portfolioToken:'F01',orderedLooks:['L99']},
    ...verifiedMenu],a).length,3);
  assert.equal(anonymousPhotoEditorialOptions(looks,[{...reviews[0],reject:true}]).length,0);
});
test('the final director may choose two truly distinct photo-approved looks instead of padding three',async()=>{
  let calls=0;
  const r=await selectPhotoEditorialDeck({
    looks,critiques:reviews,portfolios:verifiedMenu,
    context:{style:'classic',occasion:'Formal',brief:'Wedding'},
    apiKey:'nvapi-a-test-safe-key',fetchImpl:async(_,opts)=>{
      calls++;
      const body=JSON.parse(opts.body);
      assert.equal(body.response_format.type,'json_schema');
      assert.deepEqual(body.response_format.json_schema.schema.properties.portfolio.enum,
        ['F00','F01','F02']);
      assert.match(body.messages[0].content,/Long-sleeve crisp shirt/);
      assert.doesNotMatch(opts.body,/SECRET|PRIVATE|private\.example/);
      return result('F01');
    },
  });
  assert.equal(calls,1);
  assert.equal(r.status,'selected');
  assert.deepEqual(r.tokens,['L02','L00']);
  assert.equal(r.portfolioToken,'F01');
});
test('provider failure, duplicate, invented SKU and missing approvals fail safely',async()=>{
  const key='nvapi-a-test-safe-key';
  assert.equal((await selectPhotoEditorialDeck({looks:looks.slice(0,2),
    critiques:reviews,apiKey:key})).status,'not_enough_approved');
  assert.equal((await selectPhotoEditorialDeck({looks,critiques:reviews,
    portfolios:verifiedMenu,apiKey:'bad-key'})).status,'not_configured');
  assert.equal((await selectPhotoEditorialDeck({looks,critiques:reviews,
    portfolios:[],apiKey:key})).status,'no_valid_portfolios');
  const fake=await selectPhotoEditorialDeck({looks,critiques:reviews,
    portfolios:verifiedMenu,apiKey:key,fetchImpl:async()=>result('F99')});
  assert.equal(fake.status,'invalid_selection');
  assert.deepEqual(fake.tokens,[]);
  const down=await selectPhotoEditorialDeck({looks,critiques:reviews,
    portfolios:verifiedMenu,apiKey:key,
    fetchImpl:async()=>Response.json({error:'fail'},{status:502})});
  assert.equal(down.status,'provider_unavailable');
});
