import sharp from 'sharp';
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  PHOTO_EDITOR_FINAL_VERSION,PHOTO_PORTFOLIO_VISION_MODEL,anonymousPhotoEditorialOptions,
  anonymousPhotoEditorialPortfolios,validatePhotoEditorialPortfolioChoice,selectPhotoEditorialDeck,
  safePhotoEditorialCreativeGap,
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


test('real original product-photo visual jury chooses only a fully verified whole deck',async()=>{
  const jpeg=await sharp({create:{width:110,height:130,channels:3,
    background:'#8c958d'}}).jpeg().toBuffer();
  let calls=0,photos=0;
  const selection=await selectPhotoEditorialDeck({
    looks,critiques:reviews,portfolios:verifiedMenu,
    context:{style:'classic',occasion:'Date night',
      preferences:{fitPreference:'relaxed'}},
    apiKey:'nvapi-test-visual-key',timeoutMs:8500,
    authorizePhoto:()=>true,
    resolvePhoto:async()=>{photos++;return {bytes:jpeg};},
    fetchImpl:async(_,opts)=>{
      calls++;
      const body=JSON.parse(opts.body);
      assert.equal(body.model,PHOTO_PORTFOLIO_VISION_MODEL);
      assert.equal(body.model,'nvidia/nemotron-3.5-super-vl-120b-a12b');
      assert.equal(body.chat_template_kwargs.enable_thinking,false);
      assert.equal(body.max_tokens,420);
      assert.equal(body.messages[0].content,'/no_think');
      assert.equal(body.messages[1].content.length,2);
      assert.equal(body.messages[1].content[1].type,'image_url');
      assert.match(body.messages[1].content[1].image_url.url,
        /^data:image\/jpeg;base64,/);
      assert.match(body.messages[1].content[0].text,/photo contact sheet/i);
      assert.match(body.messages[1].content[0].text,/luxury personal styling atelier/i);
      assert.doesNotMatch(body.messages[1].content[0].text,
        /PRIVATE|SECRET|client|private\.example|https:\/\//);
      return result('F01');
    },
  });
  assert.equal(calls,1,'one complete photographic portfolio decision, no additional LLM');
  assert.equal(photos,20,'real photo board includes original 5 x 4 SKU images');
  assert.equal(selection.status,'selected');
  assert.equal(selection.visualJury,'applied');
  assert.deepEqual(selection.tokens,['L02','L00']);
  assert.equal(selection.portfolioToken,'F01');
  assert.equal(selection.providerCalls,1);
});

test('visual provider HTTP failure keeps the proven source-safe text editor, never weakens gates',async()=>{
  const jpeg=await sharp({create:{width:70,height:75,channels:3,
    background:'#a8aec0'}}).jpeg().toBuffer();
  let calls=0;
  const selection=await selectPhotoEditorialDeck({
    looks,critiques:reviews,portfolios:verifiedMenu,
    context:{style:'classic',occasion:'Work'},timeoutMs:8500,
    apiKey:'nvapi-test-visual-key',
    authorizePhoto:()=>true,resolvePhoto:async()=>({bytes:jpeg}),
    fetchImpl:async(_url,opts)=>{
      calls++;
      const body=JSON.parse(opts.body);
      if(body.model===PHOTO_PORTFOLIO_VISION_MODEL)
        return Response.json({error:'visual model not configured'},{status:400});
      assert.equal(body.model,'nvidia/nemotron-3.5-lightning-30b-a3b');
      return result('F02');
    },
  });
  assert.equal(calls,2);
  assert.equal(selection.status,'selected');
  assert.equal(selection.visualJury,'unavailable');
  assert.deepEqual(selection.tokens,['L04']);
  assert.equal(selection.providerCalls,2);
});

test('missing authorized fashion photo degrades honestly to text, never invented visual inspection',async()=>{
  let calls=0;
  const selection=await selectPhotoEditorialDeck({
    looks,critiques:reviews,portfolios:verifiedMenu,
    timeoutMs:8500,apiKey:'nvapi-test-visual-key',
    authorizePhoto:()=>true,resolvePhoto:async()=>null,
    fetchImpl:async(_,opts)=>{
      calls++;
      assert.equal(JSON.parse(opts.body).model,
        'nvidia/nemotron-3.5-lightning-30b-a3b');
      return result('F02');
    },
  });
  assert.equal(calls,1);
  assert.equal(selection.visualJury,'image_unavailable');
  assert.equal(selection.status,'selected');
});

test('invalid photo-jury decision never unlocks an unverified portfolio token',async()=>{
  const jpeg=await sharp({create:{width:70,height:70,channels:3,
    background:'#c9c9c9'}}).jpeg().toBuffer();
  let calls=0;
  const selection=await selectPhotoEditorialDeck({
    looks,critiques:reviews,portfolios:verifiedMenu,
    timeoutMs:8500,apiKey:'nvapi-test-visual-key',
    authorizePhoto:()=>true,resolvePhoto:async()=>({bytes:jpeg}),
    fetchImpl:async(_url,opts)=>{
      calls++;
      return JSON.parse(opts.body).model===PHOTO_PORTFOLIO_VISION_MODEL
        ?result('F99'):result('F01');
    },
  });
  assert.equal(calls,2);
  assert.equal(selection.status,'selected');
  assert.equal(selection.visualJury,'invalid_selection');
  assert.deepEqual(selection.tokens,['L02','L00']);
});


test('genuine PHOTO fashion editor returns sanitized missing creative direction ONLY after a verified F-token selection',async()=>{
  const jpeg=await sharp({create:{width:110,height:130,channels:3,
    background:'#8d806f'}}).jpeg().toBuffer();
  const gap='A tailored suede overshirt with relaxed cream linen trousers adds a softer vertical silhouette instead of two identical cardigans';
  let calls=0;
  const edited=await selectPhotoEditorialDeck({
    looks,critiques:reviews,portfolios:verifiedMenu,
    context:{style:'classic',occasion:'Date night'},timeoutMs:9200,
    apiKey:'nvapi-test-good',authorizePhoto:()=>true,
    resolvePhoto:async()=>({bytes:jpeg}),
    fetchImpl:async(_url,req)=>{
      calls++;
      const body=JSON.parse(req.body);
      assert.equal(body.model,PHOTO_PORTFOLIO_VISION_MODEL);
      assert.match(body.messages[1].content[0].text,/missingFashionDirection/);
      return Response.json({choices:[{finish_reason:'stop',message:{
        content:JSON.stringify({portfolio:'F01',missingFashionDirection:gap}),
      }}]});
    },
  });
  assert.equal(calls,1);
  assert.equal(edited.visualJury,'applied');
  assert.equal(edited.creativeGap,gap);
  assert.deepEqual(edited.tokens,['L02','L00']);
  assert.equal(safePhotoEditorialCreativeGap(''),null);
  assert.equal(safePhotoEditorialCreativeGap('Short'),null);
  assert.match(safePhotoEditorialCreativeGap('A fresh https://evil.example P003 SKU991 jacket cut with a broader silhouette'),/\[redacted\]/);
  assert.doesNotMatch(safePhotoEditorialCreativeGap('A fresh https://evil.example P003 SKU991 jacket cut with a broader silhouette'),
    /https|P003|SKU991/);
  const invalid=await selectPhotoEditorialDeck({
    looks,critiques:reviews,portfolios:verifiedMenu,
    apiKey:'nvapi-test-good',timeoutMs:8500,authorizePhoto:()=>true,
    resolvePhoto:async()=>({bytes:jpeg}),
    fetchImpl:async()=>Response.json({choices:[{finish_reason:'stop',message:{
      content:JSON.stringify({portfolio:'F99',missingFashionDirection:gap}),
    }}]}),
  });
  assert.equal(invalid.status,'invalid_selection',
    'a fake portfolio can never use a gap to create source-unauthorized products');
});
