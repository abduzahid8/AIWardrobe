import sharp from 'sharp';
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  PHOTO_EDITOR_FINAL_VERSION,PHOTO_PORTFOLIO_VISION_MODEL,OPENROUTER_JURY_MODEL,GEMMA_JURY_MODEL,
  anonymousPhotoEditorialOptions,
  anonymousPhotoEditorialPortfolios,validatePhotoEditorialPortfolioChoice,selectPhotoEditorialDeck,
  safePhotoEditorialCreativeGap,
} from './photo_editor_final.mjs';
import { PROVIDER_ENDPOINTS } from './ai_provider_router.mjs';
import { createPhotoByteCache } from './editorial_board_renderer.mjs';
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
test('final VISUAL collection editor sees which real four-SKU looks realize actual trusted creative art, without trusting labels',async()=>{
  const plans=Array.from({length:3},(_,i)=>({
    id:'ai-director-'+(i+1),generatedBy:'s6-creative-director-v1',
    name:['Soft tactile date narrative','Fluid sculptural city energy',
      'Minimal sharp expression'][i],
    artDirection:[
      'Build a deliberate material dialogue between finely textured knit, a softly tailored layer, long gentle trouser proportions and grounded footwear for the personal evening.',
      'Create a self-possessed elegant silhouette through flowing outer construction, a clean shirt collar, lengthened trousers and delicate contrasted leather shoes.',
      'Compose a third artistic expression with understated soft structure, distinct jacket length, an open top line, graceful leg drape and characterful evening footwear.',
    ][i],
  }));
  const attributed=looks.map((look,i)=>({...look,
    briefId:i<3?'ai_director_'+(i+1):'ai_composed_'+(i+1)}));
  const options=anonymousPhotoEditorialOptions(attributed,reviews,{creativePlans:plans});
  assert.equal(options.length,5);
  assert.deepEqual(options.slice(0,3).map(x=>x.originalArtistConcept.planNumber),[1,2,3]);
  assert.equal(options[1].originalArtistConcept.artisticTitle,plans[1].name);
  assert.equal(options[2].originalArtistConcept.completeLookArtIntent,plans[2].artDirection);
  assert.ok(options.slice(3).every(x=>!Object.hasOwn(x,'originalArtistConcept')),
    'unscripted independently composed alternatives do not steal validated first-artist identity');
  const spoof=anonymousPhotoEditorialOptions(attributed,reviews,{creativePlans:[
    {...plans[0],generatedBy:'shopper-untrusted'},plans[1],
    {...plans[2],id:'ai-director-2'},
  ]});
  assert.equal(spoof[0].originalArtistConcept,undefined);
  assert.equal(spoof[1].originalArtistConcept.planNumber,2);
  assert.equal(spoof[2].originalArtistConcept,undefined);
  const scrubbed=anonymousPhotoEditorialOptions(attributed,reviews,{creativePlans:[
    {...plans[0],artDirection:plans[0].artDirection+
      ' https://private.example/secret/?token=abc and customer@example.com'},
    plans[1],plans[2],
  ]});
  assert.doesNotMatch(JSON.stringify(scrubbed),/private\.example|customer@example\.com|SECRET-CLIENT|PRIVATE-0-0/);
  let received='';
  const answer=await selectPhotoEditorialDeck({looks:attributed,
    critiques:reviews,portfolios:verifiedMenu,
    creativePlans:plans,context:{style:'classic',occasion:'Date night'},
    apiKey:'nvapi-concept-photos-trusted',fetchImpl:async(_url,init)=>{
      received=JSON.parse(init.body).messages[0].content;
      return result('F01');
    },
  });
  assert.equal(answer.status,'selected');
  assert.deepEqual(answer.tokens,['L02','L00'],
    'the editor STILL chooses a verified two-look fashion portfolio rather than forced 3');
  assert.match(received,/ORIGINAL ARTISTIC/);
  assert.match(received,/Soft tactile date narrative/);
  assert.match(received,/Fluid sculptural city energy/);
  assert.match(received,/Minimal sharp expression/);
  assert.doesNotMatch(received,/private\.example|SECRET-CLIENT-SKU|PRIVATE-2-3/);
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
  assert.equal((await selectPhotoEditorialDeck({looks:looks.slice(0,1),
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


test('final whole-portfolio AI judges the DATE human social purpose without claiming a reference photo',async()=>{
  let called=0;
  const selected=await selectPhotoEditorialDeck({
    looks,critiques:reviews,portfolios:verifiedMenu,
    context:{style:'classic',occasion:'Date night'},
    apiKey:'nvapi-test-date-fashion-final-editor',
    fetchImpl:async(_url,init)=>{
      called++;
      const payload=JSON.parse(init.body);
      const prompt=payload.messages[0].content;
      assert.match(prompt,/social DATE evening/);
      assert.match(prompt,/three compelling, genuinely different/);
      const context=JSON.parse(prompt.slice(prompt.lastIndexOf('\n')+1));
      assert.equal(context.user.humanFashionScene.scene,'personal_evening_date');
      assert.doesNotMatch(prompt,/PRIVATE-|SECRET-CLIENT-SKU|private\.example/);
      return result('F01');
    },
  });
  assert.equal(called,1);
  assert.equal(selected.status,'selected');
  assert.deepEqual(selected.tokens,['L02','L00']);
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


test('TWO independently photo-approved outfits receive real contact-sheet VLM critique and original source-safe F-token',async()=>{
  const photo=await sharp({create:{width:120,height:130,channels:3,
    background:'#596981'}}).jpeg().toBuffer();
  const gap='Missing relaxed suede tailoring and subtly flowing wide-leg trousers to break the repeated office suit uniform';
  let providerCalls=0,photoReads=0;
  const two=await selectPhotoEditorialDeck({
    looks:looks.slice(0,2),critiques:reviews,
    portfolios:[
      {portfolioToken:'F00',orderedLooks:['L00','L01']},
      {portfolioToken:'F01',orderedLooks:['L01']},
    ],
    context:{style:'classic',occasion:'Work'},
    apiKey:'nvapi-a-test-photo-editor-two',timeoutMs:9300,
    resolvePhoto:async()=>{photoReads++;return {bytes:photo};},
    authorizePhoto:()=>true,
    fetchImpl:async(_,opts)=>{
      providerCalls++;
      const body=JSON.parse(opts.body);
      assert.equal(body.model,PHOTO_PORTFOLIO_VISION_MODEL);
      assert.ok(body.messages[1].content[1].image_url.url.startsWith('data:image/jpeg;base64,'));
      assert.match(body.messages[1].content[0].text,/missingFashionDirection/);
      assert.match(body.messages[1].content[0].text,/F00/);
      assert.doesNotMatch(opts.body,/PRIVATE|SECRET|private\.example/);
      return Response.json({choices:[{finish_reason:'stop',
        message:{content:JSON.stringify({portfolio:'F00',missingFashionDirection:gap})}}]});
    },
  });
  assert.equal(photoReads,8,'two actual 4-item outfits must supply eight genuine product photos');
  assert.equal(providerCalls,1);
  assert.equal(two.status,'selected');
  assert.equal(two.visualJury,'applied');
  assert.deepEqual(two.tokens,['L00','L01']);
  assert.equal(two.creativeGap,gap);
  assert.equal(two.portfolioToken,'F00');
});

test('independent Kimi photographic fashion jury can VETO repetitive three-look collection but NEVER invent items',async()=>{
  const jpeg=await sharp({create:{width:115,height:155,channels:3,
    background:'#9b8777'}}).jpeg().toBuffer();
  let nvidia=0,kimi=0,photos=0;
  const selection=await selectPhotoEditorialDeck({
    looks,critiques:reviews,portfolios:verifiedMenu,
    context:{style:'classic',occasion:'Date night',
      preferences:{favoriteColors:['cream','navy']}},
    apiKey:'nvapi-test-independent-vision',
    timeoutMs:9400,independentVisionJury:true,
    authorizePhoto:()=>true,
    resolvePhoto:async()=>{photos++;return {bytes:jpeg};},
    fetchImpl:async(_,opts)=>{
      const body=JSON.parse(opts.body);
      assert.ok(body.messages.at(-1).content.at(-1).image_url.url
        .startsWith('data:image/jpeg;base64,'));
      assert.doesNotMatch(opts.body,/SECRET|PRIVATE|private\.example|https:\/\/private/);
      if(body.model===PHOTO_PORTFOLIO_VISION_MODEL){
        nvidia++;return result('F00'); // 3 complete looks
      }
      assert.equal(body.model,'moonshotai/kimi-k3.1-pro');
      assert.equal(Object.hasOwn(body,'top_p'),false,
        'real Kimi provider FIXES top_p=0.95: sending Nemotron 0.9 causes HTTP 400');
      assert.equal(body.chat_template_kwargs.enable_thinking,false);
      assert.match(body.messages[0].content[0].text,
        /SECOND, INDEPENDENT expert VISUAL FASHION EDITOR/);
      kimi++;
      return result('F01'); // TWO, exact verified SUBSET of Nemotron's 3
    },
  });
  assert.equal(nvidia,1);
  assert.equal(kimi,1,'two different model families see SAME licensed real JPEG board');
  assert.equal(photos,20,'real authorized product photos built once only');
  assert.equal(selection.status,'selected');
  assert.equal(selection.visualJury,'applied');
  assert.equal(selection.visualMediaStatus,'ready');
  assert.equal(selection.independentJury,'narrowed');
  assert.deepEqual(selection.tokens,['L02','L00']);
  assert.equal(selection.portfolioToken,'F01');
  assert.equal(selection.providerCalls,2);
});

test('Kimi second-lab original-photo verdict survives PRIMARY portfolio VLM outage, never using text-only guesses',async()=>{
  const jpeg=await sharp({create:{width:135,height:145,channels:3,
    background:'#78838b'}}).jpeg().toBuffer();
  const models=[];
  const gap='A more structured suede layer and contrasting trouser drape add an original romantic silhouette';
  const chosen=await selectPhotoEditorialDeck({
    looks,critiques:reviews,portfolios:verifiedMenu,
    context:{style:'classic',occasion:'Date night'},
    apiKey:'nvapi-test-original-source',timeoutMs:9400,
    independentVisionJury:true,
    authorizePhoto:()=>true,
    resolvePhoto:async()=>({bytes:jpeg}),
    fetchImpl:async(_url,req)=>{
      const body=JSON.parse(req.body);
      models.push(body.model);
      assert.ok(body.messages.at(-1).content.at(-1).image_url.url
        .startsWith('data:image/jpeg;base64,'));
      return body.model===PHOTO_PORTFOLIO_VISION_MODEL
        ?Response.json({error:'primary_VLM_unavailable'},{status:503})
        :Response.json({choices:[{finish_reason:'stop',message:{
          content:JSON.stringify({portfolio:'F01',
            missingFashionDirection:gap}),
        }}]});
    },
  });
  assert.deepEqual(new Set(models),new Set([
    PHOTO_PORTFOLIO_VISION_MODEL,'moonshotai/kimi-k3.1-pro']));
  assert.equal(chosen.status,'selected');
  assert.deepEqual(chosen.tokens,['L02','L00'],
    'a second lab can choose ONLY an already-photo-approved feasible F-token');
  assert.equal(chosen.portfolioToken,'F01');
  assert.equal(chosen.visualJury,'applied',
    'Kimi actually saw the same source JPEGs, not a text-only portfolio guess');
  assert.equal(chosen.independentJury,'standalone',
    'report honestly that Kimi alone supplied the collection-level visual choice');
  assert.equal(chosen.visualMediaStatus,'ready');
  assert.equal(chosen.providerCalls,2,
    'never need the third text-only provider call after verified independent photo choice');
  assert.equal(chosen.creativeGap,gap);
});

test('independent original-photo Kimi may finish AFTER primary VLM-specific deadline without exceeding outer budget',async()=>{
  const jpeg=await sharp({create:{width:110,height:145,channels:3,
    background:'#86856c'}}).jpeg().toBuffer();
  let primarySignal=null,kimiSignal=null;
  const models=[];
  const chosen=await selectPhotoEditorialDeck({
    looks,critiques:reviews,portfolios:verifiedMenu,
    apiKey:'nvapi-primary-timeout-independent-photo',
    timeoutMs:6000,independentVisionJury:true,
    authorizePhoto:()=>true,resolvePhoto:async()=>({bytes:jpeg}),
    fetchImpl:async(_url,req)=>{
      const body=JSON.parse(req.body);
      models.push(body.model);
      if(body.model===PHOTO_PORTFOLIO_VISION_MODEL){
        primarySignal=req.signal;
        return new Promise((_resolve,reject)=>{
          if(req.signal.aborted)return reject(Error('primary_vlm_timed_out'));
          req.signal.addEventListener('abort',()=>{
            reject(Error('primary_vlm_timed_out'));
          },{once:true});
        });
      }
      assert.equal(body.model,'moonshotai/kimi-k3.1-pro');
      kimiSignal=req.signal;
      return new Promise((resolve,reject)=>{
        if(req.signal.aborted)return reject(Error('kimi_pre_aborted'));
        req.signal.addEventListener('abort',()=>reject(Error('kimi_aborted')),
          {once:true});
        const respond=()=>setTimeout(()=>{
          if(req.signal.aborted)return reject(Error('kimi_aborted_on_primary'));
          resolve(result('F01'));
        },100);
        if(primarySignal?.aborted)respond();
        else primarySignal.addEventListener('abort',respond,{once:true});
      });
    },
  });
  assert.equal(primarySignal.aborted,true,
    'the real primary VLM-specific timeout must have fired');
  assert.equal(kimiSignal.aborted,false,
    'an independent completed photo verdict must NOT inherit primary VLM abort');
  assert.equal(chosen.status,'selected');
  assert.deepEqual(chosen.tokens,['L02','L00']);
  assert.equal(chosen.visualJury,'applied');
  assert.equal(chosen.independentJury,'standalone');
  assert.equal(chosen.providerCalls,2,
    'real Kimi photo verdict replaces extra text-only paid provider retry');
  assert.deepEqual(new Set(models),new Set([
    'moonshotai/kimi-k3.1-pro',PHOTO_PORTFOLIO_VISION_MODEL]));
});

test('TWO photo-approved looks still receive an independent original-photo Kimi decision when primary vision fails',async()=>{
  const jpeg=await sharp({create:{width:112,height:124,channels:3,
    background:'#938e80'}}).jpeg().toBuffer();
  const models=[];
  const verifiedPair=[{portfolioToken:'F00',orderedLooks:['L00','L01']}];
  const photo=await selectPhotoEditorialDeck({
    looks:looks.slice(0,2),critiques:reviews,portfolios:verifiedPair,
    context:{style:'classic',occasion:'Date night'},
    apiKey:'nvapi-real-photo-two-look-independent',timeoutMs:9400,
    independentVisionJury:true,
    authorizePhoto:()=>true,resolvePhoto:async()=>({bytes:jpeg}),
    fetchImpl:async(_url,req)=>{
      const payload=JSON.parse(req.body);
      models.push(payload.model);
      assert.ok(payload.messages.at(-1).content.at(-1).image_url.url
        .startsWith('data:image/jpeg;base64,'),
      'both independent judges must ACTUALLY receive the authorized original photo board');
      return payload.model===PHOTO_PORTFOLIO_VISION_MODEL
        ?Response.json({error:'visual_primary_outage'},{status:503})
        :result('F00');
    },
  });
  assert.deepEqual(new Set(models),new Set([
    PHOTO_PORTFOLIO_VISION_MODEL,'moonshotai/kimi-k3.1-pro']));
  assert.equal(photo.status,'selected');
  assert.deepEqual(photo.tokens,['L00','L01']);
  assert.equal(photo.visualJury,'applied');
  assert.equal(photo.visualMediaStatus,'ready');
  assert.equal(photo.independentJury,'standalone');
  assert.equal(photo.providerCalls,2,
    'independent real-photo approval avoids a third paid text-only editor');
});

test('cross-lab photo fallback cannot select a fabricated F-token after primary vision failure',async()=>{
  const jpeg=await sharp({create:{width:115,height:130,channels:3,
    background:'#b9a48c'}}).jpeg().toBuffer();
  const tried=[];
  const answer=await selectPhotoEditorialDeck({
    looks,critiques:reviews,portfolios:verifiedMenu,
    apiKey:'nvapi-test-no-hallucinations',timeoutMs:9300,
    independentVisionJury:true,
    authorizePhoto:()=>true,resolvePhoto:async()=>({bytes:jpeg}),
    fetchImpl:async(_url,req)=>{
      const body=JSON.parse(req.body);tried.push(body.model);
      return body.model==='moonshotai/kimi-k3.1-pro'
        ?result('F99') // invalid fabricated portfolio, MUST NEVER pass
        :body.model===PHOTO_PORTFOLIO_VISION_MODEL
          ?Response.json({error:'VLM temporarily unavailable'},{status:502})
          :result('F00'); // existing verified text-editor fallback
    },
  });
  assert.equal(answer.status,'selected');
  assert.equal(answer.visualJury,'unavailable',
    'invalid Kimi photographic hallucination never claims applied visual jury');
  assert.deepEqual(answer.tokens,['L00','L01','L02']);
  assert.equal(answer.providerCalls,3);
  assert.equal(tried.filter(x=>x==='moonshotai/kimi-k3.1-pro').length,1);
});

test('conflicting ORIGINAL-PHOTO editor votes trigger independent third fashion judgment ONLY between already verified photo portfolios',async()=>{
  const jpeg=await sharp({create:{width:100,height:130,channels:3,
    background:'#788790'}}).jpeg().toBuffer();
  const calls=[];
  let thirdPayload=null;
  const selected=await selectPhotoEditorialDeck({
    looks,critiques:reviews,portfolios:verifiedMenu,
    context:{style:'classic',occasion:'Date night'},
    apiKey:'nvapi-cross-lab-photo-disagreement',
    timeoutMs:9600,independentVisionJury:true,crossLabTieBreak:true,
    authorizePhoto:()=>true,resolvePhoto:async()=>({bytes:jpeg}),
    fetchImpl:async(_url,req)=>{
      const b=JSON.parse(req.body);
      calls.push(b);
      if(b.model==='moonshotai/kimi-k3.1-pro')return result('F00');
      if(b.model===PHOTO_PORTFOLIO_VISION_MODEL)return result('F01');
      thirdPayload=b;
      return result('F00');
    },
  });
  assert.deepEqual(new Set(calls.map(x=>x.model)),new Set([
    'moonshotai/kimi-k3.1-pro',PHOTO_PORTFOLIO_VISION_MODEL,
    'nvidia/nemotron-3.5-lightning-30b-a3b']));
  assert.equal(selected.providerCalls,3);
  assert.equal(thirdPayload.model,'nvidia/nemotron-3.5-lightning-30b-a3b');
  assert.equal(thirdPayload.response_format.type,'json_schema');
  assert.deepEqual(thirdPayload.response_format.json_schema.schema.properties.portfolio.enum,
    ['F01','F00'],'referee cannot select any third unspecified portfolio');
  assert.match(thirdPayload.messages[0].content,/INDEPENDENT FINAL FASHION ARBITER/);
  assert.match(thirdPayload.messages[0].content,/original-photo/i);
  assert.match(thirdPayload.messages[0].content,/NEVER suppress a\s+third excellent/i);
  assert.doesNotMatch(thirdPayload.messages[0].content,
    /SECRET-CLIENT|PRIVATE-|private\.example|https:\/\//i);
  assert.equal(selected.independentJury,'arbitrated_second');
  assert.equal(selected.independentVoteSize,3);
  assert.equal(selected.visualJury,'applied');
  assert.equal(selected.status,'selected');
  assert.equal(selected.portfolioToken,'F00');
  assert.deepEqual(selected.tokens,['L00','L01','L02'],
    'an actually photo-verified feasible triple is allowed when third arbiter decides it adds fashion value');
});

test('third fashion arbiter may prefer stronger TWO-look primary choice, never forces three cards',async()=>{
  const jpeg=await sharp({create:{width:105,height:115,channels:3,
    background:'#9b958d'}}).jpeg().toBuffer();
  const verdict=await selectPhotoEditorialDeck({
    looks,critiques:reviews,portfolios:verifiedMenu,
    apiKey:'nvapi-verify-beauty-not-count',timeoutMs:9300,
    independentVisionJury:true,crossLabTieBreak:true,
    authorizePhoto:()=>true,resolvePhoto:async()=>({bytes:jpeg}),
    fetchImpl:async(_url,req)=>{
      const model=JSON.parse(req.body).model;
      return result(model==='moonshotai/kimi-k3.1-pro'?'F00':'F01');
    },
  });
  assert.equal(verdict.providerCalls,3);
  assert.equal(verdict.independentJury,'arbitrated_primary');
  assert.equal(verdict.independentVoteSize,3);
  assert.equal(verdict.portfolioToken,'F01');
  assert.deepEqual(verdict.tokens,['L02','L00']);
});

test('unavailable or invented third vote never changes original source-and-photo-verified portfolio',async()=>{
  const jpeg=await sharp({create:{width:99,height:111,channels:3,
    background:'#847e78'}}).jpeg().toBuffer();
  for(const mode of ['unavailable','fabricated']){
    const decision=await selectPhotoEditorialDeck({
      looks,critiques:reviews,portfolios:verifiedMenu,
      apiKey:'nvapi-no-unverified-inclusion',timeoutMs:9500,
      independentVisionJury:true,crossLabTieBreak:true,
      authorizePhoto:()=>true,resolvePhoto:async()=>({bytes:jpeg}),
      fetchImpl:async(_url,req)=>{
        const model=JSON.parse(req.body).model;
        if(model==='moonshotai/kimi-k3.1-pro')return result('F00');
        if(model===PHOTO_PORTFOLIO_VISION_MODEL)return result('F01');
        return mode==='unavailable'?Response.json({error:'temporary'},{status:503})
          :result('F99');
      },
    });
    assert.equal(decision.providerCalls,3);
    assert.equal(decision.independentJury,'disagreed');
    assert.equal(decision.independentVoteSize,3);
    assert.deepEqual(decision.tokens,['L02','L00']);
    assert.equal(decision.portfolioToken,'F01');
  }
});

test('independent studio jury may NEVER add a nonprimary look, silently override NVIDIA, or weaken unavailable-model fallback',async()=>{
  const jpeg=await sharp({create:{width:110,height:135,channels:3,
    background:'#7c8580'}}).jpeg().toBuffer();
  const common={
    looks,critiques:reviews,portfolios:verifiedMenu,
    context:{style:'classic',occasion:'Date night'},
    apiKey:'nvapi-test-independent-vision',timeoutMs:9400,
    independentVisionJury:true,
    authorizePhoto:()=>true,resolvePhoto:async()=>({bytes:jpeg}),
  };
  const divergent=await selectPhotoEditorialDeck({
    ...common,fetchImpl:async(_,req)=>{
      const model=JSON.parse(req.body).model;
      return result(model==='moonshotai/kimi-k3.1-pro'
        ?'F02' // L04 was not in primary F00: cannot add it
        :'F00');
    },
  });
  assert.equal(divergent.independentJury,'disagreed');
  assert.deepEqual(divergent.tokens,['L00','L01','L02']);
  assert.equal(divergent.providerCalls,2);
  const unavailable=await selectPhotoEditorialDeck({
    ...common,fetchImpl:async(_,req)=>{
      const model=JSON.parse(req.body).model;
      return model==='moonshotai/kimi-k3.1-pro'
        ?Response.json({error:'not_available'},{status:410})
        :result('F00');
    },
  });
  assert.equal(unavailable.independentJury,'unavailable');
  assert.deepEqual(unavailable.tokens,['L00','L01','L02']);
  assert.equal(unavailable.visualJury,'applied');
  assert.equal(unavailable.providerCalls,2);
  const photoModels=[];
  const onlyTwo=await selectPhotoEditorialDeck({
    ...common,looks:looks.slice(0,2),
    portfolios:[{portfolioToken:'F00',orderedLooks:['L00','L01']}],
    fetchImpl:async(_,req)=>{
      const model=JSON.parse(req.body).model;
      photoModels.push(model);
      assert.ok([PHOTO_PORTFOLIO_VISION_MODEL,
        'moonshotai/kimi-k3.1-pro'].includes(model));
      return result('F00');
    },
  });
  assert.deepEqual(new Set(photoModels),new Set([
    PHOTO_PORTFOLIO_VISION_MODEL,'moonshotai/kimi-k3.1-pro']),
    'real photographs merit independent cross-lab editing even for TWO approved looks');
  assert.equal(onlyTwo.independentJury,'agreed');
  assert.equal(onlyTwo.providerCalls,2);
  assert.equal(onlyTwo.visualMediaStatus,'ready');
});

test('cross-lab portfolio jury fails over from Kimi 503 to OpenRouter Gemini 2.5 Flash and Gemma 4 instead of returning unavailable', async () => {
  const jpeg = await sharp({ create: { width: 140, height: 180, channels: 3, background: '#7f7264' } }).jpeg().toBuffer();

  // Case 1: Kimi 503 -> OpenRouter Gemini 2.5 Flash narrows repetitive 3-look F00 to verified 2-look F01
  const openrouterAttempts = [];
  const orSelection = await selectPhotoEditorialDeck({
    looks,
    critiques: reviews,
    portfolios: verifiedMenu,
    context: { style: 'classic', occasion: 'Date night' },
    apiKey: 'nvapi-test-crosslab-failover',
    openrouterApiKey: 'sk-or-v1-test-crosslab-key-0123456',
    timeoutMs: 9500,
    independentVisionJury: true,
    authorizePhoto: () => true,
    resolvePhoto: async () => ({ bytes: jpeg }),
    fetchImpl: async (url, init) => {
      const body = JSON.parse(init.body);
      openrouterAttempts.push({ url, model: body.model });
      if (body.model === PHOTO_PORTFOLIO_VISION_MODEL) return result('F00');
      if (body.model === 'moonshotai/kimi-k3.1-pro') return new Response('kimi outage', { status: 503 });
      assert.equal(url, PROVIDER_ENDPOINTS.openrouter);
      assert.equal(body.model, OPENROUTER_JURY_MODEL);
      assert.equal(Object.hasOwn(body, 'chat_template_kwargs'), false);
      assert.match(body.messages[0].content[0].text, /RUTHLESSLY PENALIZE VISUAL REDUNDANCY/);
      return result('F01');
    },
  });

  assert.equal(orSelection.status, 'selected');
  assert.equal(orSelection.visualJury, 'applied');
  assert.equal(orSelection.independentJury, 'narrowed');
  assert.equal(orSelection.independentJuryModel, OPENROUTER_JURY_MODEL);
  assert.equal(orSelection.independentJuryProvider, 'openrouter');
  assert.deepEqual(orSelection.tokens, ['L02', 'L00']);
  assert.deepEqual(orSelection.boardDimensions, { width: 1800, height: 1000 });
  assert.ok(openrouterAttempts.some((a) => a.model === OPENROUTER_JURY_MODEL));

  // Case 2: NVIDIA-only with enableCrossLabFailover: Kimi 410 -> Gemma 4 31B succeeds!
  const gemmaSelection = await selectPhotoEditorialDeck({
    looks,
    critiques: reviews,
    portfolios: verifiedMenu,
    context: { style: 'classic', occasion: 'Work' },
    apiKey: 'nvapi-test-gemma-failover',
    enableCrossLabFailover: true,
    timeoutMs: 9500,
    independentVisionJury: true,
    authorizePhoto: () => true,
    resolvePhoto: async () => ({ bytes: jpeg }),
    fetchImpl: async (_url, init) => {
      const body = JSON.parse(init.body);
      if (body.model === PHOTO_PORTFOLIO_VISION_MODEL) return result('F00');
      if (body.model === 'moonshotai/kimi-k3.1-pro') return new Response('gone', { status: 410 });
      assert.equal(body.model, GEMMA_JURY_MODEL);
      return result('F00');
    },
  });
  assert.equal(gemmaSelection.independentJury, 'agreed');
  assert.equal(gemmaSelection.independentJuryModel, GEMMA_JURY_MODEL);
});

test('high-res 1800x1000 comparison sheet verifies JPEG dimensions, uses photoCache, and handles 6 concurrent editors without load_shed', async () => {
  const jpeg = await sharp({ create: { width: 120, height: 150, channels: 3, background: '#687482' } }).jpeg().toBuffer();
  const photoCache = createPhotoByteCache({ maxEntries: 64 });
  let photoFetches = 0;
  let capturedBoardMeta = null;

  const first = await selectPhotoEditorialDeck({
    looks: looks.slice(0, 3),
    critiques: reviews,
    portfolios: [{ portfolioToken: 'F00', orderedLooks: ['L00', 'L01', 'L02'] }],
    context: { style: 'classic', occasion: 'Work' },
    apiKey: 'nvapi-test-highres-board',
    timeoutMs: 9200,
    photoCache,
    authorizePhoto: () => true,
    resolvePhoto: async () => { photoFetches++; return { bytes: jpeg }; },
    fetchImpl: async (_url, init) => {
      const body = JSON.parse(init.body);
      const dataUrl = body.messages[1].content[1].image_url.url;
      const b64 = dataUrl.replace(/^data:image\/jpeg;base64,/, '');
      capturedBoardMeta = await sharp(Buffer.from(b64, 'base64')).metadata();
      assert.match(body.messages[1].content[0].text, /RUTHLESSLY PENALIZE VISUAL REDUNDANCY/);
      return result('F00');
    },
  });

  assert.equal(first.status, 'selected');
  assert.equal(photoFetches, 12);
  assert.equal(capturedBoardMeta.format, 'jpeg');
  assert.equal(capturedBoardMeta.width, 1800);
  assert.equal(capturedBoardMeta.height, 1000);

  // 6 concurrent portfolio editor calls succeed with ZERO load_shed (legacy inFlight >= 2 wall removed!)
  const concurrent = await Promise.all(
    Array.from({ length: 6 }, () =>
      selectPhotoEditorialDeck({
        looks: looks.slice(0, 3),
        critiques: reviews,
        portfolios: [{ portfolioToken: 'F00', orderedLooks: ['L00', 'L01', 'L02'] }],
        context: { style: 'classic', occasion: 'Work' },
        apiKey: 'nvapi-test-concurrent-scale',
        timeoutMs: 4000,
        fetchImpl: async () => {
          await new Promise((r) => setTimeout(r, 20));
          return result('F00');
        },
      }),
    ),
  );

  assert.equal(concurrent.length, 6);
  assert.ok(concurrent.every((r) => r.status === 'selected'), 'all 6 concurrent requests must succeed without load_shed');
});

test('secondary independent portfolio jury fails over cleanly from Kimi timeout to OpenRouter Gemini 2.5 Flash', async () => {
  const jpeg = await sharp({ create: { width: 120, height: 150, channels: 3, background: '#5b6770' } }).jpeg().toBuffer();
  const attempted = [];
  const selection = await selectPhotoEditorialDeck({
    looks,
    critiques: reviews,
    portfolios: verifiedMenu,
    context: { style: 'classic', occasion: 'Work' },
    apiKey: 'nvapi-test-kimi-timeout-failover',
    openrouterApiKey: 'sk-or-v1-test-openrouter-failover-key-0123456789',
    enableCrossLabFailover: true,
    timeoutMs: 6500,
    independentVisionJury: true,
    authorizePhoto: () => true,
    resolvePhoto: async () => ({ bytes: jpeg }),
    fetchImpl: async (url, init) => {
      const body = JSON.parse(init.body);
      attempted.push({ url: String(url), model: body.model });
      if (body.model === PHOTO_PORTFOLIO_VISION_MODEL) {
        return result('F00');
      }
      if (body.model === 'moonshotai/kimi-k3.1-pro') {
        // Simulate Kimi hanging past its per-provider timeout so signal aborts
        return await new Promise((_, reject) => {
          const onAbort = () => {
            const err = new Error('The operation was aborted due to timeout');
            err.name = 'TimeoutError';
            reject(err);
          };
          if (init.signal?.aborted) return onAbort();
          init.signal?.addEventListener('abort', onAbort, { once: true });
        });
      }
      assert.equal(body.model, OPENROUTER_JURY_MODEL, 'must fail over to OpenRouter Gemini 2.5 Flash after Kimi timeout');
      assert.match(String(url), /openrouter\.ai/);
      return result('F00');
    },
  });

  assert.equal(selection.status, 'selected');
  assert.equal(selection.visualJury, 'applied');
  assert.equal(selection.independentJury, 'agreed');
  assert.equal(selection.independentJuryProvider, 'openrouter');
  assert.equal(selection.independentJuryModel, OPENROUTER_JURY_MODEL);
  assert.ok(attempted.some((a) => a.model === 'moonshotai/kimi-k3.1-pro'));
  assert.ok(attempted.some((a) => a.model === OPENROUTER_JURY_MODEL));
});
