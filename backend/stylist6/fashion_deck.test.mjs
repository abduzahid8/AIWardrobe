import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import { adjacentCreativeStyles, applyExclusionQualityFloor, buildPriceProfile, elevatedBasePolishPenaltyForLook, ELEVATED_BASE_POLISH_PENALTY, elevatedFootwearRegisterContext, fashionStylingContext, polishedEveningLayerRequired, footwearRegisterPenaltyForLook, FOOTWEAR_REGISTER_PENALTY, pricePenaltyForLook, unbudgetedPriceSeedAdjustment, UNBUDGETED_PRICE_PENALTY_MAX, explicitlyRequestsLuxury, priceFreshnessPenaltyForLook, PRICE_FRESHNESS_PENALTY_MAX, rankVisualChallengerPool, visualArchitectureCapacity, visualRescueTripleCount, visualProductUniqueTripleCount, visualPortfolioSingleVetoResilience, chooseIndependentPhotoReviewTrio, assembleIndependentVisualPortfolio, directorSlatePreview, workoutItemTruthReason, freezingCatalogVestReason, rejectUnrequestedOfficeRegisterPiece, recommendFashionDeck, rejectForUnsuitableDateNight, rejectUnpolishedDateNightLook, rejectQuietSmartLeisureLayer, rejectPolishedContextMismatch, singleCurrencyBonus, trainerFootwearDescriptor, refineQuietOfficeConcept, quietLuxuryHoodieReason, quietLuxuryBasicTeePenaltyForLook, namedCardiganBriefMismatch, rejectUnrequestedRivieraPiece, extremeUnrequestedCatalogPriceReason, observedPriceComparison, photographicPaletteEvidence } from './fashion_deck.mjs';
import { devLookFeasibility, sourceHash } from './catalog_adapter.mjs';
import { shortlistRetrievalDeck } from './prototype.mjs';
import { feedbackTasteModel } from './feedback_taste.mjs';

function garment(id, name, macroCategory, subtype, extra = {}) {
  return { id, name, macroCategory, subtype, isShopItem: true, currency: 'UZS', price: 250000,
    imageUrl: 'https://catalog.example/image/' + id + '.jpg', shopUrl: 'https://catalog.example/' + id, ...extra };
}
const products = [
  ...['white', 'navy', 'cream', 'grey', 'beige'].map((color) => garment('shirt-' + color, color + ' cotton shirt', 'top', 'shirt')),
  ...['grey', 'navy', 'black', 'brown', 'charcoal'].map((color) => garment('trouser-' + color, color + ' tailored suit trousers', 'bottom', 'tailored-trouser')),
  ...['brown', 'black', 'tan', 'burgundy', 'navy'].map((color) => garment('derby-' + color, color + ' leather derby shoes', 'shoes', 'derby')),
  ...['grey', 'navy', 'black', 'brown'].map((color) => garment('blazer-' + color, color + ' wool blazer', 'outerwear', 'blazer')),
  garment('snow-parka', 'Padded neon red snow parka', 'outerwear', 'jacket'),
  garment('training-shoe', 'Neon yellow technical running sneakers', 'shoes', 'sneaker'),
];
const refs = [{
  id: 'rt_001', referenceId: 'rt_001', name: 'Shirt and trousers',
  roles: { top: ['shirt'], outerwear: ['blazer'], bottom: ['tailored-trouser'], shoes: ['derby'] },
  layerPolicy: 'optional', palette: ['navy', 'white', 'brown', 'grey'],
  garmentNotes: { top: 'white shirt', bottom: 'grey trousers', shoes: 'brown leather derby' },
}];
const base = {
  catalog: products, referenceBriefs: refs, hardGate: async () => ({ ok: true }),
  priceToBudgetCurrency: (item, currency) => item.currency === currency ? item.price : NaN,
  request: { sourceMode: 'catalog', style: 'classic', occasion: 'Work', weather: { temp: 28 },
    optionalLayer: true, referenceId: 'rt_001', budget: { amount: 750000, currency: 'UZS' },
    preferences: { avoidColors: ['black'] }, requiredIds: ['shirt-white'] },
};

test('live Classic Work rejects a terry resort shirt and guru-collar statement jacket before NVIDIA', () => {
  const office = {sourceMode:'catalog',style:'classic',occasion:'Work',request:'polished office tailoring'};
  const resort = garment('yellow-terry','Yellow Terry Towelling Resort Shirt','top','shirt');
  const jacket = garment('guru-jacket','Icon guru-collar wool-blend jacket with off-centre fastening','outerwear','jacket');
  assert.equal(rejectUnrequestedOfficeRegisterPiece(resort,office),'unrequested_resort_top_in_polished_office');
  assert.equal(rejectUnrequestedOfficeRegisterPiece(jacket,office),'unrequested_statement_jacket_in_classic_office');
  // Preserve appropriate garments, weekend freedom and direct user intent.
  for (const item of [garment('blue-shirt','Blue Cutaway Cotton Shirt','top','dress-shirt'),
    garment('blue-blazer','Navy Single-Breasted Wool Blazer','outerwear','blazer'),
    garment('brown-knit','Fine Merino Wool Polo','top','polo')])
    assert.equal(rejectUnrequestedOfficeRegisterPiece(item,office),null);
  assert.equal(rejectUnrequestedOfficeRegisterPiece(resort,{...office,occasion:'Weekend'}),null);
  assert.equal(rejectUnrequestedOfficeRegisterPiece(jacket,{...office,style:'old_money',occasion:'Weekend'}),null);
  assert.equal(rejectUnrequestedOfficeRegisterPiece(resort,{...office,sourceMode:'wardrobe'}),null);
  assert.equal(rejectUnrequestedOfficeRegisterPiece(resort,{...office,request:'Please use my yellow terry resort shirt'}),null);
  assert.equal(rejectUnrequestedOfficeRegisterPiece(jacket,{...office,request:'Wear my asymmetric statement jacket'}),null);
  assert.equal(rejectUnrequestedOfficeRegisterPiece(garment('unknown','Item 123','top','shirt'),office),null);
});

test('workout catalog admission rejects real-world polo, jeans, and casual jackets before ranking',()=>{
  const context={occasion:'Workout',style:'sport'};
  const truth=(name,macroCategory)=>workoutItemTruthReason({id:'x',name,macroCategory},context);
  assert.equal(truth('Rust Linen Polo Shirt','top'),'non_workout_top');
  assert.equal(truth('regular fit knit polo in viscose mix','top'),'non_workout_top');
  assert.equal(truth('Sea Green Ribbed T-Shirt','top'),'unverified_workout_top');
  assert.equal(truth('The Athletic 4-Way Stretch Organic Jean | Uniform','bottom'),'non_workout_bottom');
  assert.equal(truth('The Performance Chino | Uniform','bottom'),'non_workout_bottom');
  assert.equal(truth('Full zip jacket with pockets','outerwear'),'unverified_workout_layer');
  assert.equal(truth('BASIC COTTON TRAINING T-SHIRT','top'),null);
  assert.equal(truth('TECHNICAL TRAINING T-SHIRT','top'),null);
  assert.equal(truth('Regular-Fit Tapered Sports Joggers with DryMove','bottom'),null);
  assert.equal(truth('SLIM FIT JOGGER WAIST TROUSERS','bottom'),null);
  assert.equal(truth('RUNNING TRAINERS','shoes'),null);
  assert.equal(truth('TRAINING TRAINERS','shoes'),null);
  assert.equal(truth('RETRO STYLE TRAINERS','shoes'),'unverified_workout_footwear');
  assert.equal(workoutItemTruthReason({name:'Rust Linen Polo Shirt',macroCategory:'top'},
    {occasion:'Casual',style:'streetwear'}),null,'do not alter non-workout styling');
});

test('Old Money / Casual avoids surprise resort shorts and terry shirts when weather is unknown',()=>{
  const context={sourceMode:'catalog',style:'old_money',occasion:'Casual',request:'Old Money Casual',weather:{}};
  const shorts=garment('e','The 7-inch Slim Fit Short','bottom','shorts');
  const terry=garment('t','Blue Turquoise Terry T-Shirt','top','t-shirt');
  const culotte=garment('zara-new-06091253800-f1',
    'Pleated Relaxed Fit Culotte Trousers','bottom','tailored-trouser');
  const chino=garment('b','Stone Regular Fit Chino','bottom','chino');
  const knit=garment('k','Merino Cotton Polo','top','polo');
  assert.equal(rejectUnrequestedRivieraPiece({slot:'bottom',subtype:'shorts'},context,shorts),'unknown_weather_unrequested_short_bottom');
  assert.equal(rejectUnrequestedRivieraPiece({slot:'top',subtype:'t-shirt'},context,terry),'unknown_weather_resort_terry_top');
  assert.equal(rejectUnrequestedRivieraPiece({slot:'bottom',subtype:'tailored-trouser'},context,culotte),
    'unknown_weather_unrequested_short_bottom',
    'a Zara culotte was shown as shorts on actual Old Money / Casual iOS');
  assert.equal(rejectUnrequestedRivieraPiece({slot:'bottom',subtype:'tailored-trouser'},
    {...context,request:'summer culottes'},culotte),null);
  assert.equal(rejectUnrequestedRivieraPiece({slot:'bottom',subtype:'chino'},context,chino),null);
  assert.equal(rejectUnrequestedRivieraPiece({slot:'top',subtype:'polo'},context,knit),null);
  assert.equal(rejectUnrequestedRivieraPiece({slot:'bottom',subtype:'shorts'},{...context,weather:{temp:28}},shorts),null);
  assert.equal(rejectUnrequestedRivieraPiece({slot:'bottom',subtype:'shorts'},{...context,request:'Old Money summer shorts'},shorts),null);
  assert.equal(rejectUnrequestedRivieraPiece({slot:'bottom',subtype:'shorts'},{...context,sourceMode:'wardrobe'},shorts),null);
  assert.equal(rejectUnrequestedRivieraPiece({slot:'bottom',subtype:'shorts'},{...context,style:'casual'},shorts),null);
});

test('Old Money catalog excludes generic hoodies while respecting the owner or an explicit hoodie request', () => {
  const hoodie=garment('zara-hoodie','Classic Wool-Blend Knit Hoodie','outerwear','hoodie');
  const cardigan=garment('genuine-cardigan','Merino Wool Button Cardigan','outerwear','cardigan');
  const scuba=garment('hm-1304459001','Loose-Fit Scuba Hooded Jacket','outerwear','jacket');
  const woolJacket=garment('normal-tailoring','Soft Wool Unstructured Jacket','outerwear','jacket');
  const context={sourceMode:'catalog',style:'old_money',occasion:'Casual',request:''};
  assert.equal(quietLuxuryHoodieReason(hoodie,{subtype:'hoodie'},context),'quiet_luxury_hoodie_mismatch');
  assert.equal(quietLuxuryHoodieReason(cardigan,{subtype:'cardigan'},context),null);
  assert.equal(quietLuxuryHoodieReason(scuba,{subtype:'jacket'},context),'quiet_luxury_hoodie_mismatch',
    'actual H&M Scuba Hooded Jacket must not pass Old Money / Casual via a jacket subtype');
  assert.equal(quietLuxuryHoodieReason(woolJacket,{subtype:'jacket'},context),null);
  assert.equal(quietLuxuryHoodieReason(scuba,{subtype:'jacket'}, {...context,request:'I specifically want a scuba jacket'}),null);
  assert.equal(quietLuxuryHoodieReason(scuba,{subtype:'jacket'}, {...context,request:'I want a hooded jacket'}),null);
  assert.equal(quietLuxuryHoodieReason(scuba,{subtype:'jacket'}, {...context,sourceMode:'wardrobe'}),null);
  assert.equal(quietLuxuryHoodieReason(scuba,{subtype:'jacket'}, {...context,style:'casual'}),null);
  assert.equal(quietLuxuryHoodieReason(hoodie,{subtype:'hoodie'}, {...context,request:'I want a hoodie'}),null);
  assert.equal(quietLuxuryHoodieReason(hoodie,{subtype:'hoodie'}, {...context,sourceMode:'wardrobe'}),null);
  assert.equal(quietLuxuryHoodieReason(hoodie,{subtype:'hoodie'}, {...context,style:'streetwear'}),null);
});
test('real H&M hooded jacket is never admitted to Old Money catalog finalists',async()=>{
  const scuba=garment('hm-1304459001','Loose-Fit Scuba Hooded Jacket','outerwear','jacket',
    {currency:'USD',price:49.99,brand:'H&M'});
  const deck=await recommendFashionDeck({
    ...base,catalog:[...products,scuba],referenceBriefs:[],
    request:{sourceMode:'catalog',style:'old_money',occasion:'Casual',
      weather:{},optionalLayer:true,preferences:{},requiredIds:[]},
  });
  assert.ok(deck.outfits.length>0,'other tailored catalog garments remain eligible');
  assert.ok(deck.outfits.every(look=>look.items.every(item=>item.id!==scuba.id)),
    'no ranked look or editorial variant can reintroduce the hooded scuba product');
});

test('Old Money / Casual gently favors refined tops over generic tees without excluding them',()=>{
  const context={sourceMode:'catalog',style:'old_money',occasion:'Casual',request:'Old Money Casual'};
  const baseTee=garment('md-uz-61870411','100% cotton short sleeve T-shirt','top','t-shirt');
  const knit=garment('md-uz-61735938','100% cotton short sleeve knit sweater','top','sweater');
  const blazer=garment('tailored','Soft Wool Blazer','outerwear','blazer');
  assert.equal(quietLuxuryBasicTeePenaltyForLook([baseTee],context),0.065);
  assert.equal(quietLuxuryBasicTeePenaltyForLook([baseTee,blazer],context),0.025);
  assert.equal(quietLuxuryBasicTeePenaltyForLook([knit],context),0);
  assert.equal(quietLuxuryBasicTeePenaltyForLook([baseTee],{...context,request:'Old Money cotton T-shirt'}),0);
  assert.equal(quietLuxuryBasicTeePenaltyForLook([baseTee],{...context,sourceMode:'wardrobe'}),0);
  assert.equal(quietLuxuryBasicTeePenaltyForLook([baseTee],{...context,style:'casual'}),0);
  assert.equal(quietLuxuryBasicTeePenaltyForLook([baseTee],{...context,occasion:'Work',request:'Office'}),0);
});

test('named Cardigan direction cannot silently become a hooded knit outfit', () => {
  const context={sourceMode:'catalog',style:'old_money',occasion:'Casual'};
  const brief={name:'Burgundy preppy cardigan',roles:{outerwear:['cardigan']}};
  assert.equal(namedCardiganBriefMismatch([garment('h','Classic Wool-Blend Knit Hoodie','outerwear','hoodie')],brief,context),true);
  assert.equal(namedCardiganBriefMismatch([garment('c','Merino Button Cardigan','outerwear','cardigan')],brief,context),false);
  assert.equal(namedCardiganBriefMismatch([],brief,context),true);
  assert.equal(namedCardiganBriefMismatch([],brief,context,['pinned-owned']),false);
  assert.equal(namedCardiganBriefMismatch([],brief,{...context,sourceMode:'wardrobe'}),false);
});

test('subzero catalog outfits cannot treat a down vest as a protective coat', () => {
  const vest = garment('winter-vest', 'Water Repellent Down Vest', 'outerwear', 'jacket');
  const cold = { sourceMode: 'catalog', weather: { temp: -5 } };
  assert.equal(freezingCatalogVestReason(vest, cold), 'freezing_requires_sleeved_outerwear');
  assert.equal(freezingCatalogVestReason(garment('gilet', 'Quilted Gilet', 'outerwear', 'jacket'), cold),
    'freezing_requires_sleeved_outerwear');
  assert.equal(freezingCatalogVestReason(garment('parka', 'Insulated Winter Parka', 'outerwear', 'jacket'), cold), null);
  assert.equal(freezingCatalogVestReason(vest, { ...cold, weather: { temp: 16 } }), null);
  assert.equal(freezingCatalogVestReason(vest, { ...cold, sourceMode: 'wardrobe' }), null);
  assert.equal(freezingCatalogVestReason(vest, { ...cold, sourceMode: 'mixed' }), null);
});

test('cold-weather classic office cannot serve a blazer plus loafers as winter protection',async()=>{
  const request={sourceMode:'catalog',style:'classic',occasion:'Work',weather:{temp:-2,condition:'snow'},optionalLayer:true};
  const invalidCatalog=[
    ...products.filter(p=>['top','bottom','shoes','outerwear'].includes(p.macroCategory)),
  ];
  const r=await recommendFashionDeck({ ...base, request, catalog:invalidCatalog,
    referenceBriefs:[],hardGate:async()=>({ok:true}) });
  assert.ok((r.outfits||[]).every(o=>
    o.items.some(i=>/\b(?:puffer|parka|insulated|overcoat|topcoat|quilted)\b/i.test(i.name)) &&
    o.items.some(i=>/\b(?:boot|chelsea|winter)\b/i.test(i.name))),
    'no suit blazer and lightweight loafer can be a claimed -2°C winter outfit');
  assert.equal(r.outfits.length,0,'the mock inventory lacks coats/boots and must fail honestly');
});

test('Old Money Work retrieves office-clean roles before the hosted critic, without weakening casual use', () => {
  const ivy = {
    id:'refined-ivy', slots:[['dress-shirt','shirt'],['blazer','cardigan'],['chino','tailored-trouser'],['loafer','derby']],
  };
  const work = refineQuietOfficeConcept(ivy,{sourceMode:'catalog',style:'old_money',occasion:'Work'});
  assert.deepEqual(work.slots[0], ['dress-shirt']);
  assert.deepEqual(work.slots[3], ['loafer','derby']);
  assert.equal(ivy.slots[0].length,2, 'shared original concept is immutable');
  assert.equal(refineQuietOfficeConcept(ivy,{sourceMode:'catalog',style:'old_money',occasion:'Weekend'}),ivy);
  assert.equal(refineQuietOfficeConcept(ivy,{sourceMode:'catalog',style:'classic',occasion:'Work'}),ivy);
  assert.equal(refineQuietOfficeConcept(ivy,{sourceMode:'wardrobe',style:'old_money',occasion:'Work'}),ivy);
  assert.equal(refineQuietOfficeConcept(ivy,{sourceMode:'mixed',style:'old_money',occasion:'Work'}),ivy);
  assert.deepEqual(refineQuietOfficeConcept({id:'smart-knit-layering'},
    {sourceMode:'catalog',style:'old_money',occasion:'Business meeting'}).slots[1],['cardigan']);
  assert.deepEqual(refineQuietOfficeConcept({id:'textured-tailoring'},
    {sourceMode:'catalog',style:'old_money',occasion:'Работа'}).slots[0],['dress-shirt','fine-knit']);
});

test('fourth visual candidate unlocks genuinely distinct three-look directions over a close score clone', () => {
  const slots = ['top', 'outerwear', 'bottom', 'shoes'];
  const make = (id, colors, topSubtype = 'shirt') => ({ id,
    items: slots.map((slot, i) => ({ id:`${id}-${slot}`, macroCategory:slot,
      color:colors[i], subtype: i === 0 ? topSubtype : slot === 'outerwear' ? 'blazer' : slot,
      texture: i === 0 && topSubtype === 'fine-knit' ? 'knit' : 'cotton' })),
    fashion: { score:0.89 },
  });
  const current = [
    make('hero',['navy','navy','navy','navy']),
    make('cool',['grey','grey','navy','navy']),
    make('warm',['navy','navy','brown','brown']),
  ];
  const clone = make('high-score-duplicate',['navy','grey','navy','navy']);
  clone.fashion.score=0.92;
  const architectural = make('clean-new-direction',['navy','navy','navy','navy'],'fine-knit');
  architectural.fashion.score=0.88;
  const baseline = visualArchitectureCapacity(current);
  assert.equal(baseline,2);
  assert.equal(visualArchitectureCapacity([...current,clone]),2);
  assert.equal(visualArchitectureCapacity([...current,architectural]),3);
  const candidates = [
    {look:clone,capacityGain:0,difference:0.4},
    {look:architectural,capacityGain:1,difference:0.25},
  ];
  assert.equal(rankVisualChallengerPool(candidates)[0].id,architectural.id);
});

test('visual challenger rank prefers a genuinely unseen main garment after rescue capacity', () => {
  const look=(id,score)=>({id,fashion:{score}});
  const duplicate={look:look('same-top',.94),rescueGain:1,capacityGain:0,unseenTop:0,difference:.4};
  const newTop={look:look('new-top',.89),rescueGain:1,capacityGain:0,unseenTop:1,difference:.2};
  assert.equal(rankVisualChallengerPool([duplicate,newTop])[0],newTop.look);
  const essential={look:look('structural-break',.85),rescueGain:2,capacityGain:0,unseenTop:0,difference:.2};
  assert.equal(rankVisualChallengerPool([newTop,essential])[0],essential.look,
    'ability to make three independent outfits outranks top novelty');
});

test('fourth visual challenger favors safe whole-look quality over extreme novelty', () => {
  const options=[
    {look:{id:'eccentric',fashion:{score:0.74}},difference:0.92},
    {look:{id:'coherent',fashion:{score:0.84}},difference:0.26},
    {look:{id:'balanced',fashion:{score:0.81}},difference:0.51},
    {look:{id:'tie-less-novel',fashion:{score:0.80}},difference:0.15},
    {look:{id:'tie-more-novel',fashion:{score:0.80}},difference:0.38},
  ];
  assert.deepEqual(rankVisualChallengerPool(options).map(x=>x.id),
    ['coherent','balanced','tie-more-novel','tie-less-novel','eccentric']);
  assert.deepEqual(options.map(x=>x.look.id),
    ['eccentric','coherent','balanced','tie-less-novel','tie-more-novel'],
    'sorting must not mutate source shortlist');
});

test('warm work outfits keep the exact anchor, avoid excluded colors and omit the unnecessary layer', async () => {
  const result = await recommendFashionDeck(base);
  assert.equal(result.status, 'complete');
  assert.equal(result.diagnostics.visualVerdict, 'NOT_RUN');
  for (const look of result.outfits) {
    assert.equal(look.items.length, 3);
    assert.ok(look.items.some(({ id }) => id === 'shirt-white'));
    assert.ok(look.items.every(({ id }) => !id.includes('black') && id !== 'training-shoe'));
    assert.ok(look.items.reduce((sum, { price }) => sum + price, 0) <= 750000);
    assert.ok(Number.isFinite(look.fashion.referenceMetadata.slotColorFit));
  }
  assert.ok(new Set(result.outfits.flatMap(({ items }) => items.map(({ id }) => id))).size >= 7);
});

test('summer 32°C work avoids an unnecessary blazer when the weather was explicit',async()=>{
  const request={...base.request,brief:'Three summer old money work outfits without a jacket',weather:{temp:32,condition:'clear'}};
  assert.equal(fashionStylingContext(request).weather.temp,32);
  const result=await recommendFashionDeck({...base,request});
  assert.equal(result.status,'complete');
  assert.equal(result.outfits.length,3);
  assert.ok(result.outfits.every(({items})=>items.length===3 && !items.some(item=>item.macroCategory==='outerwear')),
    'summer season word must not erase the explicit 32°C temperature');
});

test('cold looks retain layers and an infeasible budget never gains a fallback outfit', async () => {
  const request = { ...base.request, weather: { temp: 9 }, budget: { amount: 1000000, currency: 'UZS' } };
  const cold = await recommendFashionDeck({ ...base, request });
  assert.equal(cold.status, 'complete');
  assert.ok(cold.outfits.every(({ items }) => items.length === 4 && items.some(({ id }) => id === 'shirt-white')));
  const impossible = await recommendFashionDeck({ ...base, request: { ...request, budget: { amount: 997500, currency: 'UZS' } } });
  assert.deepEqual(impossible.outfits, []);
});

test('current cached color can exclude a garment and a stale profile cannot', async () => {
  const item = garment('anchor', 'Cotton shirt', 'top', 'shirt');
  const sourceHash = 'a'.repeat(32);
  const visualFashion = {
    version: 1, imageUrl: item.imageUrl, imageSha256: 'b'.repeat(64), sourceHash, model: 'fixture', revision: 'one',
    attributes: { color: { value: 'black', confidence: .95, margin: .4, evidence: 'image-model' } },
  };
  const request = { ...base.request, requiredIds: ['anchor'] };
  const run = (visualSourceHash) => recommendFashionDeck({
    ...base, request, catalog: [...products, { ...item, visualFashion, visualSourceHash }],
  });
  assert.deepEqual((await run(sourceHash)).outfits, []);
  const stale = await run('c'.repeat(32));
  assert.ok(stale.outfits.length > 0);
  assert.ok(stale.outfits.every(({ items, fashion }) => items.some(({ id }) => id === 'anchor') && fashion.imageSupportedItems === 0));
});

test('denied image review leaves the catalog-ranked deck intact with zero critic calls', async () => {
  const baseline = await recommendFashionDeck(base);
  const result = await recommendFashionDeck({
    ...base, canJudgeOutfit: async () => { throw new Error('grant unavailable'); },
    visualCritic: () => assert.fail('uncleared photos cannot be sent'),
  });
  assert.deepEqual(result.outfits, baseline.outfits);
  assert.equal(result.diagnostics.visualVerdict, 'NOT_RUN');
});


const admittedVisualVerdict = (candidate, overrides = {}) => ({
  candidateId: candidate.id, visualCohesion: 0.9, referenceFidelity: 0.9,
  // Every real NVIDIA verdict now includes a bounded observed hue per image.
  // Derive mock hues from the fixture's explicit garment title rather than
  // allowing missing-photo-color evidence to mask tests about deck selection.
  slotColors:Object.fromEntries(candidate.products.map(product=>[
    product.slot,
    (/\b(white|navy|cream|grey|beige|brown|burgundy|tan|black|charcoal|olive|blue)\b/i
      .exec(String(product.name||''))?.[1]||'cream').toLowerCase(),
  ])),
  styleFit: 0.9, occasionFit: 0.9, formalityFit: 0.9, confidence: 0.9, reject: false,
  evidence: ['shirt and trousers proportion', 'layer/palette balance', 'footwear register'],
  ...overrides,
});

test('real planned AI architectures reach visual evaluation before house challengers fill the eight-call limit', async () => {
  const aiBrief={id:'ai-director-1',name:'Cream Merino Contemporary',
    generatedBy:'s6-creative-director-v1',palette:['cream','brown','beige'],silhouette:'balanced',
    layerPolicy:'required',roles:{top:['fine-knit'],outerwear:['coat'],bottom:['chino'],shoes:['loafer']}};
  const extra=[
    garment('ai-knit','Cream Fine Merino Knit','top','fine-knit',{color:'cream'}),
    garment('ai-coat','Camel Wool Overcoat','outerwear','coat',{color:'brown'}),
    garment('ai-chino','Beige Tailored Chino','bottom','chino',{color:'beige'}),
    garment('ai-loafer','Brown Suede Penny Loafers','shoes','loafer',{color:'brown'}),
  ];
  let judged=[];
  const result=await recommendFashionDeck({
    ...base, catalog:[...products,...extra], aiCreativeBriefs:[aiBrief],referenceBriefs:[],
    request:{sourceMode:'catalog',style:'old_money',occasion:'Weekend',
      weather:{temp:17},optionalLayer:true,budget:{amount:1500000,currency:'UZS'},
      preferences:{},requiredIds:[]},
    visualDeckReview:true,canJudgeOutfit:async()=>true,
    visualCritic:async candidates=>{judged=candidates;return candidates.map(x=>admittedVisualVerdict(x));},
  });
  assert.equal(result.diagnostics.aiCreativeDirectorBriefs,1);
  assert.ok(result.diagnostics.aiCreativeDirectorEligible>0,'real role match creates a feasible AI outfit');
  // The hosted critic only receives product photos/names, not trusted
  // internal plan provenance, which is recorded in server diagnostics.
  assert.ok(result.diagnostics.distinctLayerArchitectures >= 1);
  assert.ok(result.diagnostics.visualReviewCoverage.aiCreativeDirectorJudged>0,
    'validated AI plans must not disappear before whole-look visual review');
  assert.ok(judged.some(look=>look.products.some(it=>it.id==='ai-knit')),
    'actual AI-plan merchant photos are included in the critic batch');
  assert.ok(judged.length<=8,'still bounded by hosted visual model cap');
  for(const look of result.outfits)assert.ok(look.items.every(it=>it.isShopItem));
});

test('cloud full-deck review judges every authorized displayed card in one bounded batch', async () => {
  const baseline = await recommendFashionDeck(base);
  let reviewCalls = 0, reviewed = [];
  const result = await recommendFashionDeck({
    ...base, visualDeckReview: true, canJudgeOutfit: async () => true,
    visualCritic: async (candidates) => {
      reviewCalls++;
      reviewed = candidates;
      return candidates.map(candidate => admittedVisualVerdict(candidate));
    },
  });
  assert.equal(reviewCalls, 1);
  assert.ok(reviewed.length >= baseline.outfits.length && reviewed.length <= 8);
  assert.ok(baseline.outfits.every(({ id }) => reviewed.some(candidate => candidate.id === id)));
  assert.equal(result.diagnostics.visualVerdict, 'JUDGED');
  assert.equal(result.diagnostics.visualReviewCoverage.mode, 'displayed_deck');
  assert.equal(result.diagnostics.visualReviewCoverage.attempted, reviewed.length);
  assert.equal(result.diagnostics.visualReviewCoverage.valid, reviewed.length);
  assert.ok(result.outfits.some(look => Array.isArray(look.visualEvidence)));
  assert.ok(result.outfits.every(look =>
    look.items.every(item => item.price === 250000) &&
    look.items.some(item => item.id === 'shirt-white')));
});

test('cloud inference-budget six protects latency/cost without displaying unreviewed looks', async () => {
  let calls=0,judged=[];
  const result=await recommendFashionDeck({
    ...base, visualDeckReview:true,visualCandidateLimit:6,
    canJudgeOutfit:async()=>true,
    visualCritic:async candidates=>{calls++;judged=candidates;
      return candidates.map(candidate=>admittedVisualVerdict(candidate));},
  });
  assert.equal(calls,1);
  assert.ok(judged.length>=3 && judged.length<=6);
  assert.equal(result.diagnostics.visualReviewCoverage.inferenceCandidateCap,6);
  assert.equal(result.diagnostics.visualReviewCoverage.attempted,judged.length);
  assert.equal(result.diagnostics.visualReviewCoverage.unreviewedFillAdded,0);
  assert.ok(result.outfits.every(look=>Array.isArray(look.visualEvidence)));
  assert.ok(result.outfits.every(look=>judged.some(c=>c.id===look.id)));
});

test('cloud full-deck review does not send any item from denied image hosts', async () => {
  const baseline = await recommendFashionDeck(base);
  const authorizedId = baseline.outfits[1].id;
  let reviewed = [];
  const result = await recommendFashionDeck({
    ...base, visualDeckReview: true,
    canJudgeOutfit: async (look) => look.id === authorizedId,
    visualCritic: async (candidates) => {
      reviewed = candidates;
      return candidates.map(candidate => admittedVisualVerdict(candidate));
    },
  });
  assert.deepEqual(reviewed.map(c => c.id), [authorizedId]);
  assert.equal(result.diagnostics.visualReviewCoverage.attempted, 1);
  assert.equal(result.diagnostics.visualReviewCoverage.valid, 1);
  assert.equal(result.outfits.length, 1, 'one verified look must not acquire two unreviewed companions');
  assert.equal(result.status, 'partial');
  assert.deepEqual(result.outfits.map(look => look.id), [authorizedId]);
  assert.ok(result.outfits.every(look => Array.isArray(look.visualEvidence)));
  assert.equal(result.diagnostics.visualReviewCoverage.displayedEvidence, 1);
});

test('cloud full-deck veto removes a rejected secondary look without laundering visual evidence', async () => {
  const baseline = await recommendFashionDeck(base);
  const badId = baseline.outfits[1].id;
  const result = await recommendFashionDeck({
    ...base, visualDeckReview: true, canJudgeOutfit: async () => true,
    visualCritic: async (candidates) => candidates.map(candidate => admittedVisualVerdict(
      candidate, candidate.id === badId ? { reject: true, visualCohesion: 0.2 } : {},
    )),
  });
  assert.ok(result.diagnostics.visualReviewCoverage.rejected >= 1);
  assert.ok(result.outfits.every(look => look.id !== badId));
  assert.ok(result.outfits.every(look => !look.visualEvidence ||
    !result.diagnostics.visualReviewCoverage.rejectedCandidateIds?.includes(look.id)));
  assert.ok(result.outfits.every(look => look.items.some(item => item.id === 'shirt-white')));
});

test('visual-first selection is controlled by the complete MODEL-APPROVED deck, not the original lead', async () => {
  const initial = await recommendFashionDeck(base);
  let candidateIds = [];
  const result = await recommendFashionDeck({
    ...base, visualDeckReview: true, canJudgeOutfit: async () => true,
    visualCritic: async (candidates) => {
      candidateIds = candidates.map(x => x.id);
      return candidates.map((candidate, index) =>
        admittedVisualVerdict(candidate, {
          visualCohesion: index === 0 ? 0.74 : 0.98,
          referenceFidelity: index === 0 ? 0.74 : 0.98,
          styleFit: index === 0 ? 0.74 : 0.98,
          occasionFit: index === 0 ? 0.74 : 0.98,
          formalityFit: index === 0 ? 0.74 : 0.98,
        }));
    },
  });
  assert.ok(candidateIds.length >= 3 && candidateIds.length <= 8);
  assert.equal(result.status, 'complete');
  assert.notEqual(result.outfits[0].id, initial.outfits[0].id,
    'the model must be able to change which outfit is displayed first');
  assert.equal(result.outfits.length, 3);
  assert.ok(result.outfits.every(look => Array.isArray(look.visualEvidence)));
  assert.equal(result.diagnostics.visualReviewCoverage.displayedEvidence, 3);
  assert.ok(result.outfits.every(look => look.items.some(item => item.id === 'shirt-white')));
});

test('six authorized challengers rescue full deck after one timeout and two quality vetoes', async () => {
  let sent=[];
  const result=await recommendFashionDeck({
    ...base, request:{ sourceMode:'catalog', style:'classic', occasion:'Work',
      weather:{ temp:20 }, optionalLayer:true },
    referenceBriefs:[],visualDeckReview:true,canJudgeOutfit:async()=>true,
    visualCritic:async(batch)=>{
      sent=batch;
      // Mimics the observed NVIDIA outcome: one unknown/timed out plus
      // other rejected candidates. Only actual model-approved looks may show.
      return batch.slice(0,5).map((candidate,i)=>admittedVisualVerdict(candidate,
        i===1||i===3?{reject:true,visualCohesion:0.2}:{}));
    },
  });
  assert.ok(sent.length>=5, 'challenger pool must exceed the old four-call cap');
  assert.ok(sent.length<=8);
  // The mock approves three outfits, but some reuse the same actual SKU.
  // This is no longer a valid three-outfit personal-stylist result. An
  // honest partial result beats recycling a cardigan to make three cards.
  assert.equal(result.status,'partial');
  assert.equal(result.outfits.length,2);
  assert.equal(result.diagnostics.visualReviewCoverage.displayedEvidence,2);
  for(let i=0;i<result.outfits.length;i++)for(let j=i+1;j<result.outfits.length;j++){
    const ids=new Set(result.outfits[i].items.map(item=>item.id));
    assert.ok(result.outfits[j].items.every(item=>!ids.has(item.id)));
  }
  assert.ok(result.outfits.every(look=>Array.isArray(look.visualEvidence)));
  const rejected=new Set([sent[1].id,sent[3].id,sent[5]?.id]);
  assert.ok(result.outfits.every(look=>!rejected.has(look.id)));
});

test('rescue triple count rewards independent fallback directions, not near-duplicate SKUs', () => {
  const make=(id, top, bottom, shoes) => ({id, items:[
    garment(`${id}-top`, top+' shirt','top','shirt'),
    garment(`${id}-bottom`,bottom+' trousers','bottom','trouser'),
    garment(`${id}-shoes`,shoes+' derby','shoes','derby'),
  ]});
  const deck=[make('a','white','black','brown'),make('b','navy','grey','black'),make('c','cream','brown','tan')];
  assert.equal(visualRescueTripleCount(deck),1);
  const redundant=make('redundant','white','black','brown');
  const spare=make('spare','beige','navy','burgundy');
  const strong=visualRescueTripleCount([...deck,spare]);
  assert.ok(strong>visualRescueTripleCount([...deck,redundant]));
  assert.ok(strong>=2);
  assert.equal(rankVisualChallengerPool([
    {look:redundant,capacityGain:0,rescueGain:0,difference:0.4},
    {look:spare,capacityGain:0,rescueGain:strong-1,difference:0.2},
  ])[0],spare);
});

test('staging may disclose a third unreviewed but hard-admitted look without laundering NVIDIA rejection', async () => {
  const audited = [];
  const result = await recommendFashionDeck({
    ...base,
    request:{ sourceMode:'catalog', style:'classic', occasion:'Work',weather:{temp:20},optionalLayer:true },
    referenceBriefs:[], visualDeckReview:true, allowUnreviewedVisualFill:true,
    canJudgeOutfit:async()=>true,
    visualCritic:async(batch) => {
      audited.push(...batch.map(x=>x.id));
      return batch.slice(0,2).map(x=>admittedVisualVerdict(x));
    },
  });
  assert.equal(result.status,'complete', 'one honest unreviewed alternative can restore three usable cards');
  assert.equal(result.outfits.length,3);
  assert.equal(result.diagnostics.visualReviewCoverage.displayedEvidence,2);
  assert.equal(result.diagnostics.visualReviewCoverage.unreviewedFillAdded,1);
  assert.equal(result.diagnostics.visualReviewCoverage.fallbackUnreviewed,true);
  assert.equal(result.outfits.filter(x=>Array.isArray(x.visualEvidence)).length,2);
  const fallback=result.outfits.find(x=>!Array.isArray(x.visualEvidence));
  assert.ok(fallback && !audited.includes(fallback.id),
    'never show a rejected or timed-out NVIDIA candidate as deterministic filler');
  assert.ok(result.outfits.every(x=>x.items.every(i=>i.price===250000)));
});

test('the third unreviewed canary look needs decodable merchant photos or stays partial',async()=>{
  const skipped=[];
  const result=await recommendFashionDeck({
    ...base,
    request:{sourceMode:'catalog',style:'classic',occasion:'Work',weather:{temp:20},optionalLayer:true},
    referenceBriefs:[],visualDeckReview:true,allowUnreviewedVisualFill:true,
    canJudgeOutfit:async()=>true,
    // Simulates a source-admitted product with an actual 403 or invalid bytes.
    canDisplayOutfit:async(look)=>{skipped.push(look.id);return false;},
    visualCritic:async(batch)=>batch.slice(0,2).map(x=>admittedVisualVerdict(x)),
  });
  assert.ok(skipped.length>0,'real photo preflight must run before an unjudged third card');
  assert.equal(result.status,'partial');
  assert.equal(result.outfits.length,2);
  assert.equal(result.diagnostics.visualReviewCoverage.displayedEvidence,2);
  assert.equal(result.diagnostics.visualReviewCoverage.unreviewedFillAdded,0);
  assert.ok(result.outfits.every(x=>Array.isArray(x.visualEvidence)));
});

test('two visually approved looks produce a truthful partial result, no unreviewed third', async () => {
  let candidates = [];
  const result = await recommendFashionDeck({
    ...base, visualDeckReview: true, canJudgeOutfit: async () => true,
    visualCritic: async (batch) => {
      candidates = batch;
      return batch.map((candidate, i) => admittedVisualVerdict(candidate,
        i < 2 ? {} : { visualCohesion: 0.2, referenceFidelity: 0.2,
          styleFit: 0.2, occasionFit: 0.2, formalityFit: 0.2, reject: false }));
    },
  });
  assert.ok(candidates.length >= 3);
  assert.equal(result.status, 'partial');
  assert.equal(result.outfits.length, 2);
  assert.deepEqual(new Set(result.outfits.map(look => look.id)),
    new Set(candidates.slice(0, 2).map(candidate => candidate.id)));
  assert.equal(result.diagnostics.visualReviewCoverage.approved, 2);
  assert.equal(result.diagnostics.visualReviewCoverage.displayedEvidence, 2);
  assert.ok(result.outfits.every(look => look.items.every(item => item.price > 0)));
});

test('cloud whole-deck review does not approve beautiful colors with a genuinely wrong occasion', async () => {
  const testRequest={...base.request,style:'semi_classic',occasion:'Date night',
    weather:{temp:17},sourceMode:'catalog',referenceId:undefined,
    requiredIds:[],budget:{amount:1800000,currency:'UZS'}};
  const result=await recommendFashionDeck({
    ...base, request:testRequest,referenceBriefs:[],visualDeckReview:true,
    canJudgeOutfit:async()=>true,allowUnreviewedVisualFill:false,
    visualCritic:async batch=>batch.map(candidate=>admittedVisualVerdict(candidate, {
      visualCohesion:.97,styleFit:.97,occasionFit:.55,formalityFit:.97,
      confidence:.97,referenceFidelity:.97,
    })),
  });
  assert.equal(result.status,'no_feasible_looks',
    'a style/occasion mismatch must never be rescued by a 0.97 palette score');
  assert.equal(result.outfits.length,0);
  assert.equal(result.diagnostics.visualReviewCoverage.approved,0);
  assert.ok(result.diagnostics.visualQualityDecisions.some(x=>x.reason==='occasion_mismatch'));
});

test('valid visual verdicts rejecting every finalist fail closed rather than showing weak outfits', async () => {
  const result = await recommendFashionDeck({
    ...base, visualDeckReview: true, canJudgeOutfit: async () => true,
    visualCritic: async candidates => candidates.map(candidate =>
      admittedVisualVerdict(candidate, {
        visualCohesion: 0.15, referenceFidelity: 0.15, styleFit: 0.15,
        occasionFit: 0.15, formalityFit: 0.15, confidence: 0.95,
      })),
  });
  assert.equal(result.status, 'no_feasible_looks');
  assert.deepEqual(result.outfits, []);
  assert.equal(result.diagnostics.visualVerdict, 'JUDGED');
  assert.equal(result.diagnostics.visualReviewCoverage.approved, 0);
});

test('invalid or timed-out visual verdicts never claim review and keep deterministic fallback explicit', async () => {
  const baseline = await recommendFashionDeck(base);
  const result = await recommendFashionDeck({
    ...base, visualDeckReview: true, canJudgeOutfit: async () => true,
    visualCritic: async () => [],
  });
  assert.equal(result.status, baseline.status);
  assert.deepEqual(result.outfits, baseline.outfits);
  assert.equal(result.diagnostics.visualVerdict, 'ABSTAINED');
  assert.equal(result.diagnostics.visualReviewCoverage.fallbackUnreviewed, true);
  assert.equal(result.diagnostics.visualReviewCoverage.displayedEvidence, 0);
});

test('authorized close contenders use at most two visual judgments and retain the complete deck', async () => {
  const baseline = await recommendFashionDeck(base);
  let reviewed = [];
  const judged = await recommendFashionDeck({
    ...base, canJudgeOutfit: async () => true,
    visualCritic: async (candidates) => {
      reviewed = candidates;
      return candidates.map(({ id }, index) => ({
        candidateId: id, visualCohesion: index ? .95 : .75, referenceFidelity: index ? .95 : .75,
        styleFit: .9, occasionFit: .9, formalityFit: .9, confidence: .9, reject: false,
        evidence: ['mock top and trouser contrast', 'mock shoe relationship', 'mock reference comparison'],
      }));
    },
  });
  assert.equal(reviewed.length, 2);
  assert.equal(reviewed[0].id, baseline.outfits[0].id);
  assert.equal(judged.outfits[0].id, reviewed[1].id);
  assert.equal(judged.outfits.length, 3);
  assert.equal(judged.outfits.filter(({ visualEvidence }) => visualEvidence).length, 1);
});

test('the lead is visually confirmed alone when no close alternative is admissible', async () => {
  const baseline = await recommendFashionDeck(base);
  let reviewed = [];
  const judged = await recommendFashionDeck({
    ...base,
    // Only the lead is cleared: every alternative is denied, so the
    // near-tie swap cannot run and the bounded single-candidate
    // confirmation must run instead of skipping visual review entirely.
    canJudgeOutfit: async (look) => look.id === baseline.outfits[0].id,
    visualCritic: async (candidates) => {
      reviewed = candidates;
      return candidates.map(({ id }) => ({
        candidateId: id, visualCohesion: .9, referenceFidelity: .9,
        styleFit: .9, occasionFit: .9, formalityFit: .9, confidence: .9, reject: false,
        evidence: ['mock top and trouser contrast', 'mock shoe relationship', 'mock reference comparison'],
      }));
    },
  });
  assert.equal(reviewed.length, 1);
  assert.equal(reviewed[0].id, baseline.outfits[0].id);
  assert.equal(judged.diagnostics.visualVerdict, 'JUDGED');
  assert.deepEqual(judged.outfits.map(({ id }) => id), baseline.outfits.map(({ id }) => id));
  assert.equal(judged.outfits.filter(({ visualEvidence }) => visualEvidence).length, 1);
});


test('the critic judges a deck with no reference brief purely on product images', async () => {
  // No referenceBriefs and no request.referenceId: the visual review must
  // still run (reference-optional staging). The mock mirrors the hosted
  // critic's reference-less verdict shape: no referenceFidelity score, no
  // reference claims, marked referenceless.
  let seenReferenceIds = [], seenCalls = 0;
  const judged = await recommendFashionDeck({
    ...base,
    referenceBriefs: [],
    request: { ...base.request, referenceId: undefined },
    canJudgeOutfit: (look, referenceId) => {
      seenReferenceIds.push(referenceId);
      return true;
    },
    visualCritic: async (candidates) => {
      seenCalls += 1;
      return candidates.map(({ id }) => ({
        candidateId: id, referenceless: true, visualCohesion: .9, styleFit: .9,
        occasionFit: .9, formalityFit: .9, confidence: .9, reject: false,
        evidence: ['mock top and trouser contrast', 'mock shoe relationship', 'mock layer relationship'],
      }));
    },
  });
  assert.equal(seenCalls, 1);
  assert.ok(seenReferenceIds.every((id) => id === null), 'reference-less gate passes null reference id');
  assert.equal(judged.diagnostics.visualVerdict, 'JUDGED');
  assert.equal(judged.outfits.filter(({ visualEvidence }) => visualEvidence).length, 1);
  for (const outfit of judged.outfits) {
    assert.equal(outfit.concept.referenceId, null);
    assert.equal(outfit.fashion.referenceMetadata, null);
  }
  // Honest evidence: no reference claim without a reference asset.
  for (const line of judged.outfits.flatMap(({ visualEvidence }) => visualEvidence || [])) {
    assert.doesNotMatch(line, /\breferences?\b/i);
  }
});

test('an unspecified wedding uses formal composition while a beach wedding or explicit code stays distinct', async () => {
  const request = { ...base.request, occasion: 'Wedding', weather: { temp: 21 }, budget: { amount: 1000000, currency: 'UZS' } };
  assert.equal(fashionStylingContext(request).dressCode, 'formal');
  assert.equal(fashionStylingContext({ ...request, brief: 'A beach wedding' }).dressCode, 'casual');
  assert.equal(fashionStylingContext({ ...request, dressCode: 'smart' }).dressCode, 'smart');
  const result = await recommendFashionDeck({ ...base, request });
  assert.ok(result.outfits.length > 0);
  for (const { items } of result.outfits) {
    assert.ok(items.some(({ id }) => id.startsWith('blazer-')));
    assert.ok(items.some(({ id }) => id.startsWith('derby-')));
    assert.ok(items.every(({ id }) => id !== 'snow-parka' && id !== 'training-shoe'));
  }
});

test('a second tap with exclusions returns different products when alternatives exist', async () => {
  const catalog = [...products,
    garment('blazer-extra-1', 'grey wool blazer', 'outerwear', 'blazer'),
    garment('blazer-extra-2', 'navy wool blazer', 'outerwear', 'blazer'),
    garment('blazer-extra-3', 'black wool blazer', 'outerwear', 'blazer')];
  const request = { ...base.request, requiredIds: [] };
  const first = await recommendFashionDeck({ ...base, catalog, request });
  assert.equal(first.status, 'complete');
  const seen = first.outfits.flatMap(({ items }) => items.map(({ id }) => id));
  assert.ok(seen.length > 0);
  const second = await recommendFashionDeck({ ...base, catalog, request: { ...request, excludedIds: seen } });
  assert.ok(['complete', 'partial'].includes(second.status));
  const again = second.outfits.flatMap(({ items }) => items.map(({ id }) => id));
  assert.ok(again.length > 0);
  assert.deepEqual(again.filter((id) => seen.includes(id)), []);
  assert.equal(second.diagnostics.exclusions.requested, new Set(seen.map(String)).size);
  assert.deepEqual(second.diagnostics.exclusions.relaxedSlots, []);
});

test('excluding every bottom relaxes gracefully instead of failing the deck', async () => {
  const bottoms = products.filter(({ macroCategory }) => macroCategory === 'bottom').map(({ id }) => id);
  const result = await recommendFashionDeck({
    ...base, request: { ...base.request, requiredIds: [], excludedIds: bottoms },
  });
  assert.equal(result.status, 'complete');
  assert.deepEqual(result.diagnostics.exclusions.relaxedSlots, ['bottom']);
});

test('an unbudgeted premium anchor completes with disclosed dampening, never a rejection', async () => {
  const premium = garment('derby-premium', 'black leather derby shoes', 'shoes', 'derby', { price: 2500000 });
  const extra = ['vi', 'vii', 'viii'].map((color) => garment(`derby-${color}`, `${color} leather derby shoes`, 'shoes', 'derby'));
  const result = await recommendFashionDeck({
    ...base,
    catalog: [...products, ...extra, premium],
    request: { sourceMode: 'catalog', style: 'classic', occasion: 'Work', weather: { temp: 28 },
      optionalLayer: true, preferences: {}, requiredIds: ['derby-premium'] },
  });
  assert.equal(result.status, 'complete');
  assert.ok(result.outfits.length > 0);
  for (const look of result.outfits) {
    assert.ok(look.items.some(({ id }) => id === 'derby-premium'));
    assert.ok(look.fashion.evidence.includes('unbudgeted_premium_selection_dampened'));
  }
});

test('price bands are per slot and currency, need evidence and cap the dampener', () => {
  const shoe = (id, price, currency = 'UZS') => ({ id, price, currency, macroCategory: 'shoes' });
  const catalog = [
    ...Array.from({ length: 9 }, (_, i) => shoe(`u${i}`, 100)),
    shoe('usp', 100000),
    ...Array.from({ length: 5 }, (_, i) => shoe(`e${i}`, 50 + i, 'EUR')),
    shoe('top1', 100),
  ];
  const profile = buildPriceProfile(catalog);
  assert.equal(profile['UZS|shoes'].count, 11);
  assert.equal(profile['UZS|shoes'].median, 100);
  assert.ok(!('EUR|shoes' in profile));
  assert.ok(!('UZS|top' in profile));
  const plain = [{ ...shoe('u0', 100) }];
  assert.deepEqual(pricePenaltyForLook(plain, profile), { penalty: 0, flagged: 0 });
  const hit = pricePenaltyForLook([{ ...shoe('usp', 100000) }], profile);
  assert.equal(hit.flagged, 1);
  assert.ok(hit.penalty > 0 && hit.penalty <= UNBUDGETED_PRICE_PENALTY_MAX);
  const mild = pricePenaltyForLook([{ ...shoe('m', 150) }], profile);
  assert.equal(mild.flagged, 0);
  assert.ok(mild.penalty > 0 && mild.penalty < hit.penalty);
  const capped = pricePenaltyForLook(
    [{ ...shoe('a', 100000) }, { ...shoe('b', 90000) }, { ...shoe('c', 80000) }], profile);
  assert.equal(capped.penalty, UNBUDGETED_PRICE_PENALTY_MAX);
  // Multiples beyond 2x continue differentiating instead of plateauing flat.
  const triple = pricePenaltyForLook([{ ...shoe('x3', 300) }], profile);
  const tenx = pricePenaltyForLook([{ ...shoe('x10', 1000) }], profile);
  assert.equal(triple.flagged, 1);
  assert.ok(tenx.penalty > triple.penalty && tenx.penalty <= UNBUDGETED_PRICE_PENALTY_MAX);
  assert.deepEqual(pricePenaltyForLook([{ ...shoe('usp', 100000) }], {}), { penalty: 0, flagged: 0 });
});

test('default catalog excludes only substantiated extreme relative price outliers, not luxury requests',()=>{
  const data=[400000,640000,849900,1000000,1200000,6500000,7400000,8500000,9100000,12000000,13000000,14000000]
    .map((price,i)=>garment(`p${i}`,'Wool tailored trousers','bottom','tailored-trouser',
      {price,currency:'UZS',brand:i===11?'Canali':'Other'}));
  const profile=buildPriceProfile(data);
  const basic=data[2],luxury=data[11];
  const context={sourceMode:'catalog',style:'old_money',occasion:'Casual',request:'Old Money Casual'};
  assert.equal(extremeUnrequestedCatalogPriceReason(basic,profile,context),null);
  assert.equal(extremeUnrequestedCatalogPriceReason(luxury,profile,context),'extreme_unrequested_catalog_price_outlier');
  assert.equal(extremeUnrequestedCatalogPriceReason(luxury,profile,{...context,request:'Old Money luxury Canali'}),null);
  assert.equal(extremeUnrequestedCatalogPriceReason(luxury,profile,{...context,preferences:{favoriteBrands:['CANALI']}}),null);
  assert.equal(extremeUnrequestedCatalogPriceReason(luxury,profile,{...context,sourceMode:'wardrobe'}),null);
  assert.equal(extremeUnrequestedCatalogPriceReason(luxury,profile,{...context,budget:{amount:15000000,currency:'UZS'}}),null);
  assert.equal(extremeUnrequestedCatalogPriceReason({...luxury,currency:'CAD'},profile,context),null);
  assert.equal(extremeUnrequestedCatalogPriceReason(luxury,{'UZS|bottom':{accessible:800000,count:7}},context),null);
});

test('verified CBU FX compares a rare €595 jacket with affordable UZS layers',()=>{
  const usual=Array.from({length:24},(_,i)=>garment(`layer-${i}`,'Tailored linen jacket','outerwear','blazer',
    {price:900000+i*35000,currency:'UZS'}));
  const expensive=garment('aurelien-denim','Dark Blue Denim Leisure Blazer','outerwear','blazer',
    {price:595,currency:'EUR',brand:'Aurélien'});
  const rows=[...usual,expensive];
  const sameCurrency=buildPriceProfile(rows);
  assert.equal(extremeUnrequestedCatalogPriceReason(expensive,sameCurrency,
    {sourceMode:'catalog',request:'Old Money Casual'}),null,
    'no comparable EUR cohort exists, so never invent a benchmark');
  const cbu={UZS:1,EUR:15000,USD:12000,GBP:17500};
  const priced=buildPriceProfile(rows,cbu);
  const observed=observedPriceComparison(expensive,priced);
  assert.equal(observed.source,'verified_cbu_reference');
  assert.equal(observed.value,8925000);
  assert.ok(observed.band.count>=25);
  const context={sourceMode:'catalog',style:'old_money',occasion:'Casual',request:'Old Money Casual'};
  assert.equal(extremeUnrequestedCatalogPriceReason(expensive,priced,context),'extreme_unrequested_catalog_price_outlier');
  assert.ok(pricePenaltyForLook([expensive],priced).penalty>0.13);
  assert.ok(unbudgetedPriceSeedAdjustment(expensive,priced)<-0.22);
  assert.equal(extremeUnrequestedCatalogPriceReason(expensive,priced,{...context,request:'I want luxury clothes'}),null);
  assert.equal(extremeUnrequestedCatalogPriceReason(expensive,priced,{...context,preferences:{favoriteBrands:['Aurélien']}}),null);
  const badRates=buildPriceProfile(rows,{UZS:2,EUR:15000,USD:12000,GBP:17500});
  assert.equal(extremeUnrequestedCatalogPriceReason(expensive,badRates,context),null,
    'no conversion without normalized, trusted FX shape');
});

test('verified cross-currency cohort catches €950 layer even when local EUR cohort is entirely luxury',()=>{
  const accessible=Array.from({length:24},(_,i)=>garment(`local-${i}`,
    'Unstructured Wool Blazer','outerwear','blazer',
    {price:900000+i*25000,currency:'UZS',brand:'Zara'}));
  const highEur=Array.from({length:24},(_,i)=>garment(`euro-${i}`,
    'Premium Wool Overshirt','outerwear','overshirt',
    {price:390+i*4,currency:'EUR',brand:'Luxury'}));
  const live=garment('aurelien-16011499471178','Navy Cashmere Blend Leisure Blazer',
    'outerwear','blazer',{price:950,currency:'EUR',brand:'Aurélien'});
  const rows=[...accessible,...highEur,live];
  const context={sourceMode:'catalog',style:'old_money',occasion:'Casual',request:'Old Money Casual'};
  const native=buildPriceProfile(rows);
  assert.ok(native['EUR|outerwear'].count>=24,'EUR cohort is well populated');
  assert.equal(extremeUnrequestedCatalogPriceReason(live,native,context),null,
    'the luxury-skewed native EUR pool did not detect this expensive layer');
  const verified=buildPriceProfile(rows,{UZS:1,EUR:15000,USD:12000,GBP:17500});
  const observed=observedPriceComparison(live,verified);
  assert.equal(observed.source,'verified_cbu_reference',
    'prefer trustworthy accessible UZS equivalents over biased EUR cohort');
  assert.equal(observed.value,14250000);
  assert.ok(observed.band.count>=49);
  assert.equal(extremeUnrequestedCatalogPriceReason(live,verified,context),
    'extreme_unrequested_catalog_price_outlier');
  assert.ok(unbudgetedPriceSeedAdjustment(live,verified)<-0.2);
  assert.equal(extremeUnrequestedCatalogPriceReason(live,verified,
    {...context,request:'Old Money luxury cashmere blazer'}),null);
  assert.equal(extremeUnrequestedCatalogPriceReason(live,verified,
    {...context,preferences:{favoriteBrands:['Aurélien']}}),null);
  assert.equal(extremeUnrequestedCatalogPriceReason(live,verified,
    {...context,budget:{amount:2000,currency:'EUR'}}),null);
});

test('skewed luxury catalog uses observed accessible quartile to preserve affordable role choices',()=>{
  const rows=[];
  for(const [i,price] of [400000,640000,849900,1000000,1200000,6500000,7400000,8500000,9100000,12000000,13000000,14000000].entries())
    rows.push(garment(`bottom-${i}`,'Wool tailored trousers','bottom','tailored-trouser',{price,currency:'UZS'}));
  const profile=buildPriceProfile(rows);
  const band=profile['UZS|bottom'];
  assert.ok(band.accessible < band.median/2);
  const basic=rows[2],luxury=rows[9];
  assert.equal(unbudgetedPriceSeedAdjustment(basic,profile),0.065);
  assert.ok(unbudgetedPriceSeedAdjustment(luxury,profile)<-0.2);
  assert.ok(pricePenaltyForLook([luxury],profile).penalty > 0.1);
  assert.equal(unbudgetedPriceSeedAdjustment({...luxury,currency:'CAD'},profile),0,
    'never invent exchange rates for currencies without a price cohort');
});

test('unbudgeted styling strongly de-prioritizes a 9m UZS luxury knit without banning it', () => {
  const profile = {'UZS|top':{median:650000,count:30}};
  const ordinary = pricePenaltyForLook([{macroCategory:'top',price:800000,currency:'UZS'}],profile);
  const luxury = pricePenaltyForLook([{macroCategory:'top',price:8939235,currency:'UZS'}],profile);
  assert.ok(ordinary.penalty < 0.02);
  assert.ok(luxury.penalty > 0.16 && luxury.penalty <= UNBUDGETED_PRICE_PENALTY_MAX);
  assert.equal(luxury.flagged,1);
  assert.equal(explicitlyRequestsLuxury({style:'old_money',request:'work outfit'}),false);
  assert.equal(explicitlyRequestsLuxury({style:'old_money',request:'luxury designer workwear'}),true);
  assert.equal(explicitlyRequestsLuxury({style:'classic',request:'премиальный костюм'}),true);
});

test('price freshness dampener is bounded, shoppable-only and tier-scaled', () => {
  const offer = (priceFreshness, isShopItem = true) => ({
    id: 's', isShopItem, price: 100, currency: 'UZS', macroCategory: 'shoes', priceFreshness,
  });
  // Fresh and recent offers pay nothing; legacy items without the field pay nothing.
  assert.equal(priceFreshnessPenaltyForLook([offer('fresh')]), 0);
  assert.equal(priceFreshnessPenaltyForLook([offer('recent')]), 0);
  assert.equal(priceFreshnessPenaltyForLook([{ id: 'legacy', isShopItem: true }]), 0);
  // Stale and unknown offers pay a small, tier-scaled cost.
  const stale = priceFreshnessPenaltyForLook([offer('stale')]);
  const unknown = priceFreshnessPenaltyForLook([offer('unknown')]);
  assert.equal(stale, 0.0125);
  assert.equal(unknown, 0.007);
  assert.ok(stale > unknown);
  // Owned wardrobe pieces carry no merchant offer and pay nothing.
  assert.equal(priceFreshnessPenaltyForLook([offer('stale', false)]), 0);
  // Sum over a four-piece look is capped: beyond the near-tie margin, far
  // below the coherence weight, never enough to rescue a bad look.
  const fourStale = priceFreshnessPenaltyForLook(
    [offer('stale'), offer('stale'), offer('stale'), offer('stale')]);
  assert.equal(fourStale, PRICE_FRESHNESS_PENALTY_MAX);
  assert.equal(PRICE_FRESHNESS_PENALTY_MAX, 0.035);
  assert.ok(PRICE_FRESHNESS_PENALTY_MAX > 0.025, 'can reorder a near-tie');
  assert.ok(PRICE_FRESHNESS_PENALTY_MAX < 0.68 * 0.1, 'far below the coherence weight');
});

test('cached visual profiles attach only on exact SKU and source-hash match', () => {
  const row = {
    id: 'canali-jnt00137221',
    name: 'Navy blue lisle cotton jersey regular-fit crew neck',
    description: "Long-sleeved crew neck in lisle cotton jersey. The ribbed neckline and side slits define a simple, functional construction, while the quality of the yarn ensures a smooth, breathable feel. A style designed for the modern man's everyday outfits. Original listed price: 550 USD; converted with CBU reference rate dated 2026-10-28.",
    brand: 'Canali', category: 'tops', garment_type: 'upper_body',
    image_url: 'https://canalius.vtexassets.com/arquivos/ids/386110/T0804-MJ00002-300_a_.jpg.jpg?v=639275732505900000',
  };
  assert.equal(sourceHash(row), 'cf2824efe2e53bc2a9f7a50b43b90b44');
  assert.notEqual(sourceHash({ ...row, description: row.description + '!' }), 'cf2824efe2e53bc2a9f7a50b43b90b44');
  assert.notEqual(sourceHash({ ...row, image_url: row.image_url + '&x=1' }), 'cf2824efe2e53bc2a9f7a50b43b90b44');
});

test('the dev feasibility gate rejects unverified offers without throwing', () => {
  const ok = garment('shirt-white', 'white cotton shirt', 'top', 'shirt');
  // Listing evidence only: a pass never confirms live stock, size or delivery.
  assert.deepEqual(devLookFeasibility([ok]), { ok: true, evidence: 'listing-only', stock: 'not_checked' });
  assert.deepEqual(devLookFeasibility([{ ...ok, id: '  ' }]),
    { ok: false, reason: 'invalid_product_identity', evidence: 'listing-only', stock: 'not_checked' });
  assert.deepEqual(devLookFeasibility([{ ...ok, macroCategory: 'hat' }]),
    { ok: false, reason: 'unknown_slot', evidence: 'listing-only', stock: 'not_checked' });
  assert.deepEqual(devLookFeasibility([{ ...ok, imageUrl: 'http://plain/photo.jpg' }])
    , { ok: false, reason: 'unrenderable_photo', evidence: 'listing-only', stock: 'not_checked' });
  assert.deepEqual(devLookFeasibility([{ ...ok, shopUrl: 'not-a-url' }])
    , { ok: false, reason: 'missing_merchant_offer', evidence: 'listing-only', stock: 'not_checked' });
  assert.deepEqual(devLookFeasibility([{ ...ok, price: 0 }])
    , { ok: false, reason: 'unpriced_or_currency', evidence: 'listing-only', stock: 'not_checked' });
  assert.deepEqual(devLookFeasibility([{ ...ok, currency: 'usd' }])
    , { ok: false, reason: 'unpriced_or_currency', evidence: 'listing-only', stock: 'not_checked' });
  assert.deepEqual(devLookFeasibility(null), { ok: true, evidence: 'listing-only', stock: 'not_checked' });
});

test('single-currency looks earn the consistency preference, mixed looks do not', () => {
  const uzs = (id) => ({ id, currency: 'UZS' });
  assert.equal(singleCurrencyBonus([uzs('a'), uzs('b'), uzs('c')]), 0.02);
  assert.equal(singleCurrencyBonus([uzs('a'), { id: 'b', currency: 'USD' }]), 0);
  assert.equal(singleCurrencyBonus([]), 0);
  assert.equal(singleCurrencyBonus(null), 0);
});

// ---------------------------------------------------------------------------
// Elevated-register footwear tie-back: a bounded 0.015 score adjustment for
// trainer-only shoes at quiet styles on dressed occasions, only while the
// retrieval's admitted pool demonstrably contains a non-trainer option.
// Near-tie reorder only — never a reject, never casual/travel contexts.
// ---------------------------------------------------------------------------

test('elevated footwear register is an explicit quiet-style x dressed-occasion table', () => {
  const context = (style, occasion) => ({ style, occasion });
  assert.equal(elevatedFootwearRegisterContext(context('old_money', 'Gallery opening')), true);
  assert.equal(elevatedFootwearRegisterContext(context('classic', 'Conference')), true);
  assert.equal(elevatedFootwearRegisterContext(context('semi_classic', 'Dinner')), true);
  assert.equal(elevatedFootwearRegisterContext(context('old_money', 'Business meeting')), true);
  assert.equal(elevatedFootwearRegisterContext(context('classic', 'Work')), true);
  assert.equal(elevatedFootwearRegisterContext(context('old_money', 'Travel')), false);
  assert.equal(elevatedFootwearRegisterContext(context('old_money', 'Weekend')), false);
  assert.equal(elevatedFootwearRegisterContext(context('casual', 'Conference')), false);
  assert.equal(elevatedFootwearRegisterContext(context('minimalist', 'Dinner')), false);
  assert.equal(elevatedFootwearRegisterContext(context('streetwear', 'Business meeting')), false);
  assert.equal(elevatedFootwearRegisterContext(null), false);
  assert.equal(elevatedFootwearRegisterContext(context('old_money', null)), false);
});

test('trainer detection keeps the documented sleek low-top boundary', () => {
  const item = (name, subtype) => garment('shoe', name, 'shoes', subtype);
  assert.equal(trainerFootwearDescriptor({ slot: 'shoes', subtype: 'sneaker' }, item('Canvas Trainers', 'sneaker')), true);
  assert.equal(trainerFootwearDescriptor({ slot: 'shoes', subtype: 'sport-sneaker', athletic: true }, item('Running Sneakers', 'sport-sneaker')), true);
  // A generic "leather trainers" listing is still a trainer despite the
  // descriptor's minimal-sneaker aliasing.
  assert.equal(trainerFootwearDescriptor({ slot: 'shoes', subtype: 'minimal-sneaker' }, item('Casual Leather Trainers', 'minimal-sneaker')), true);
  // A deliberately minimal/clean/low-top silhouette keeps the quiet register.
  assert.equal(trainerFootwearDescriptor({ slot: 'shoes', subtype: 'minimal-sneaker' }, item('Clean White Low-Top Leather Sneakers', 'minimal-sneaker')), false);
  assert.equal(trainerFootwearDescriptor({ slot: 'shoes', subtype: 'loafer' }, item('Suede Loafers', 'loafer')), false);
  assert.equal(trainerFootwearDescriptor({ slot: 'shoes', subtype: 'derby' }, item('Leather Derby Shoes', 'derby')), false);
  assert.equal(trainerFootwearDescriptor({ slot: 'bottom', subtype: 'sneaker' }, item('Joggers', 'sneaker')), false);
});

test('footwear register penalty is bounded and skipped without feasible dress shoes', () => {
  const trainer = garment('trainer-generic', 'Casual Leather Trainers', 'shoes', 'minimal-sneaker');
  const sleek = garment('trainer-sleek', 'Clean White Low-Top Leather Sneakers', 'shoes', 'minimal-sneaker');
  const derby = garment('derby-brown', 'Brown Leather Derby Shoes', 'shoes', 'derby');
  const descriptorOf = (item) => ({ slot: item.macroCategory, subtype: item.subtype });
  const gallery = { style: 'old_money', occasion: 'Gallery opening' };
  // Trainer-only look at an elevated register with dress shoes feasible: the
  // bounded penalty fires (half the near-tie margin, never more).
  assert.equal(footwearRegisterPenaltyForLook([trainer], [descriptorOf(trainer)], gallery, true), FOOTWEAR_REGISTER_PENALTY);
  assert.ok(FOOTWEAR_REGISTER_PENALTY > 0 && FOOTWEAR_REGISTER_PENALTY < 0.025);
  // A sleek low-top is not treated as a trainer.
  assert.equal(footwearRegisterPenaltyForLook([sleek], [descriptorOf(sleek)], gallery, true), 0);
  assert.equal(footwearRegisterPenaltyForLook([derby], [descriptorOf(derby)], gallery, true), 0);
  // The only feasible shoes are trainers: nothing fires.
  assert.equal(footwearRegisterPenaltyForLook([trainer], [descriptorOf(trainer)], gallery, false), 0);
  // Casual styles and non-elevated occasions are outside the register.
  assert.equal(footwearRegisterPenaltyForLook([trainer], [descriptorOf(trainer)],
    { style: 'casual', occasion: 'Conference' }, true), 0);
  assert.equal(footwearRegisterPenaltyForLook([trainer], [descriptorOf(trainer)],
    { style: 'old_money', occasion: 'Travel' }, true), 0);
});

test('elevated-register deck keeps trainer looks eligible but visibly dampened and reordered', async () => {
  const catalog = [...products,
    garment('trainer-generic', 'Casual Leather Trainers', 'shoes', 'minimal-sneaker'),
    garment('trainer-canvas', 'Court Canvas Sneakers', 'shoes', 'sneaker')];
  // Semi-classic at a dressed occasion: trainers still pass the shared hard
  // gates (old_money/classic contexts reject sneaker subtypes outright), so
  // the register tie-back is the remaining near-tie correction.
  const request = { ...base.request, style: 'semi_classic', occasion: 'Conference',
    weather: { temp: 19 }, requiredIds: [], preferences: {}, budget: undefined };
  const result = await recommendFashionDeck({ ...base, catalog, request });
  assert.ok(result.outfits.length > 0);
  let trainerLooks = 0, dressLooks = 0;
  for (const look of result.outfits) {
    const shoes = look.items.filter(({ macroCategory }) => macroCategory === 'shoes');
    const allTrainer = shoes.length > 0 && shoes.every((shoe) =>
      trainerFootwearDescriptor({ slot: 'shoes', subtype: shoe.subtype }, shoe));
    assert.equal(look.fashion.evidence.includes('trainer_footwear_register_dampened'), allTrainer,
      'register evidence must match the actual footwear');
    if (allTrainer) trainerLooks += 1; else dressLooks += 1;
  }
  assert.ok(dressLooks >= 1, 'dress-shoe looks stay eligible and competitive');
  assert.ok(trainerLooks + dressLooks === result.outfits.length);
  // A trainer-only feasible pool (no dress shoes at all) pays nothing.
  const trainersOnly = [...products.filter(({ macroCategory }) => macroCategory !== 'shoes'),
    garment('trainer-generic', 'Casual Leather Trainers', 'shoes', 'minimal-sneaker'),
    garment('trainer-canvas', 'Court Canvas Sneakers', 'shoes', 'sneaker')];
  const sparse = await recommendFashionDeck({ ...base, catalog: trainersOnly, request });
  assert.ok(sparse.outfits.length > 0);
  for (const look of sparse.outfits) {
    assert.equal(look.fashion.evidence.includes('trainer_footwear_register_dampened'), false);
  }
});

test('the exclusion quality floor prefers a smaller honest deck over a much worse outfit', () => {
  const look = (id, score) => ({ id, fashion: { score } });
  const selected = [look('a', 0.85), look('b', 0.83), look('c', 0.50)];
  assert.deepEqual(applyExclusionQualityFloor(selected, 0.86, false), { kept: selected, limited: false });
  const floored = applyExclusionQualityFloor(selected, 0.86, true);
  assert.deepEqual(floored.kept.map(({ id }) => id), ['a', 'b']);
  assert.equal(floored.limited, true);
  const allWeak = applyExclusionQualityFloor([look('a', 0.5), look('b', 0.4)], 0.9, true);
  assert.deepEqual(allWeak.kept.map(({ id }) => id), ['a']);
  assert.equal(allWeak.limited, true);
});

test('shorts and athletic bottoms stay out of smart date nights but remain daytime options', async () => {
  const catalog = [...products,
    garment('short-denim', 'Blue denim shorts', 'bottom', 'shorts'),
    garment('short-athletic', 'Black training shorts', 'bottom', 'athletic-trouser')];
  const dateRequest = {
    sourceMode: 'catalog', style: 'semi_classic', occasion: 'Date night',
    weather: {}, optionalLayer: true, preferences: {}, requiredIds: [],
  };
  const date = await recommendFashionDeck({
    ...base, catalog, hardGate: async () => ({ ok: true }),
    priceToBudgetCurrency: (item, currency) => (item.currency === currency ? item.price : NaN),
    request: dateRequest,
  });
  assert.ok(date.outfits.length > 0);
  for (const look of date.outfits) {
    assert.ok(look.items.every(({ id }) => !id.startsWith('short-')));
  }
  const workRequest = { ...dateRequest, style: 'classic', occasion: 'Work' };
  const work = await recommendFashionDeck({
    ...base, catalog, hardGate: async () => ({ ok: true }),
    priceToBudgetCurrency: (item, currency) => (item.currency === currency ? item.price : NaN),
    request: workRequest,
  });
  assert.equal(work.status, 'complete');
});

test('optional briefs keep three-piece candidates in the shortlist beside layered looks', async () => {
  const brief = (id, layerPolicy, palette, silhouette) => ({
    id, name: id, roles: { top: ['shirt'], outerwear: ['blazer'], bottom: ['tailored-trouser'], shoes: ['derby'] },
    layerPolicy, palette, silhouette,
  });
  const retrieval = await shortlistRetrievalDeck({
    request: { sourceMode: 'catalog', style: 'classic', occasion: 'Work', weather: {},
      optionalLayer: true, preferences: {}, budget: { amount: 25000000, currency: 'UZS' } },
    catalog: products,
    hardGate: async () => ({ ok: true }),
    priceToBudgetCurrency: (item, currency) => (item.currency === currency ? item.price : NaN),
    creativeDirector: async () => [
      brief('opt-a', 'optional', ['white', 'navy'], 'clean'),
      brief('opt-b', 'optional', ['grey', 'black'], 'relaxed'),
      brief('opt-c', 'optional', ['beige', 'brown'], 'soft'),
      brief('req-a', 'required', ['white', 'grey'], 'tailored'),
      brief('req-b', 'required', ['navy', 'brown'], 'structured'),
      brief('req-c', 'required', ['black', 'charcoal'], 'sharp'),
    ],
    options: {},
  });
  const lens = new Set(retrieval.shortlist.map((c) => c.items.length));
  assert.ok(retrieval.shortlist.length > 0);
  assert.ok(lens.has(3), 'three-piece candidates must compete');
  assert.ok(lens.has(4), 'layered candidates must compete');
});

test('a confident visual rejection removes the lead even when the other candidate abstains', async () => {
  const baseline = await recommendFashionDeck(base);
  const result = await recommendFashionDeck({
    ...base, canJudgeOutfit: async () => true,
    visualCritic: async ([lead]) => [{
      candidateId: lead.id, visualCohesion: .4, referenceFidelity: .6, styleFit: .7,
      occasionFit: .7, formalityFit: .7, confidence: .9, reject: true,
      evidence: ['mock top and trouser contrast', 'mock shoe relationship', 'mock reference comparison'],
    }],
  });
  assert.equal(result.diagnostics.visualVerdict, 'ABSTAINED');
  assert.equal(result.outfits.length, 3);
  assert.ok(result.outfits.every(({ id, visualEvidence }) => id !== baseline.outfits[0].id && !visualEvidence));
});


test('New Clothing respects explicitly avoided brands without weakening structure and budget', async () => {
  const forbidden = garment('shirt-explicitly-avoided', 'white cotton shirt', 'top', 'shirt',
    {brand: 'Not My Brand'});
  const result = await recommendFashionDeck({
    ...base, catalog: [...products, forbidden],
    request: {
      ...base.request,
      preferences: {avoidColors: ['black'], avoidBrands: ['not my brand']},
    },
  });
  assert.equal(result.status, 'complete');
  for (const look of result.outfits) {
    assert.ok(look.items.every(i => i.brand !== 'Not My Brand'));
    assert.ok(look.items.some(i => i.id === 'shirt-white'));
    assert.ok(look.items.reduce((sum,i) => sum + i.price, 0) <= 750000);
  }
});

test('New Clothing emits product-grounded taste evidence for an explicit favorite', async () => {
  const catalog = products.map(i => i.id === 'shirt-white' ? {...i,brand:'Preferred House'} : i);
  const result = await recommendFashionDeck({
    ...base, catalog,
    request: {
      ...base.request,
      preferences: {
        favoriteColors: ['white'],
        favoriteBrands: ['PREFERRED HOUSE'],
        fitPreference: 'balanced',
        avoidColors: ['black'],
      },
    },
  });
  assert.equal(result.status, 'complete');
  for (const look of result.outfits) {
    assert.ok(look.items.some(i=>i.id==='shirt-white'));
    assert.ok(look.fashion.personalization.adjustment > 0);
    assert.equal(look.fashion.personalization.matchedItems >= 1, true);
    assert.ok(look.fashion.personalization.matches.some(m=>
      m.productId==='shirt-white' && m.type==='favorite_brand'));
    assert.ok(look.fashion.personalization.matches.every(m=>
      look.items.some(i=>i.id===m.productId)));
    assert.ok(look.fashion.personalization.adjustment<=.085);
  }
});

test('explicitly locked owned item overrides only its own brand exclusion', async () => {
  const owned = garment('shirt-white', 'white cotton shirt', 'top', 'shirt', {
    brand:'Avoided Label',isShopItem:false,isWardrobe:true,price:null,currency:null,
  });
  const catalog = [...products.filter(i=>i.id!=='shirt-white'),owned];
  const result = await recommendFashionDeck({
    ...base,catalog,
    request:{
      ...base.request,sourceMode:'mixed',requiredIds:['shirt-white'],
      preferences:{avoidBrands:['avoided label']},
      budget:undefined,
    },
  });
  assert.equal(result.status,'complete');
  for(const look of result.outfits){
    assert.ok(look.items.some(i=>i.id==='shirt-white'));
    assert.ok(look.items.every(i=>i.brand!=='Avoided Label'||i.id==='shirt-white'));
  }
});


test('short onboarding explores adjacent ideas without changing the requested dress code', async () => {
  assert.deepEqual(adjacentCreativeStyles('old_money','smart'),['semi_classic','minimalist']);
  assert.deepEqual(adjacentCreativeStyles('classic','formal'),[]);
  assert.deepEqual(adjacentCreativeStyles('casual','athletic'),[]);
  assert.deepEqual(adjacentCreativeStyles('unknown','smart'),[]);
  const result = await recommendFashionDeck({
    ...base,request:{
      ...base.request,style:'old_money',occasion:'Work',
      requiredIds:[],budget:{amount:1000000,currency:'UZS'},
      preferences:{}, // no large form, no guessed favorite colors/brands
    },
  });
  assert.ok(result.outfits.length > 0);
  assert.equal(result.diagnostics.neighborBriefs,2);
  assert.equal(result.diagnostics.directionVersion,'s6-three-directions-editorial-v5');
  assert.equal(result.outfits[0].direction.id,'signature');
  for(const outfit of result.outfits){
    assert.ok(['signature','refresh','discovery','alternative'].includes(outfit.direction.id));
    assert.ok(outfit.items.every(item=>item.id!=='snow-parka'&&item.id!=='training-shoe'));
    assert.ok(outfit.items.reduce((total,item)=>total+item.price,0)<=1000000);
    if(outfit.direction.id==='discovery'){
      assert.ok(outfit.direction.qualified);
      assert.ok(outfit.direction.evidence.structuralSlots.length>0 ||
        outfit.direction.evidence.paletteSlots.length>=3);
    }
  }
});

test('pinned anchor survives exploration and remains in every suggested direction', async()=>{
  const result=await recommendFashionDeck({
    ...base, request:{...base.request,preferences:{},requiredIds:['shirt-white']},
  });
  assert.ok(result.outfits.length>0);
  for(const look of result.outfits){
    assert.ok(look.items.some(x=>x.id==='shirt-white'));
    assert.ok(look.items.reduce((total,item)=>total+item.price,0)<=750000);
    assert.ok(look.direction);
  }
});


test('owner 2026-11-12: minimalist smart date excludes resort towelling tee and retro leather trainers', async () => {
  const smartDate = fashionStylingContext({style:'minimalist',occasion:'Date night',
    sourceMode:'catalog',weather:{},optionalLayer:true,preferences:{}});
  assert.equal(smartDate.dressCode,'smart');
  const casualDate = {...smartDate,dressCode:'casual'};
  const resortTee = garment('real-resort-tee',
    'Blue Turquoise Terry Towelling T-Shirt','top','t-shirt');
  const retroTrainer = garment('real-retro-shoe',
    'RETRO LEATHER TRAINERS','shoes','minimal-sneaker');
  const cleanSneaker = garment('quiet-low-top',
    'Clean White Low-Top Leather Sneakers','shoes','minimal-sneaker');
  assert.equal(rejectForUnsuitableDateNight(
    {slot:'bottom',subtype:'jeans',athletic:false},smartDate,
    garment('mislabelled-shorts','Blue denim shorts','bottom','jeans')),true);
  assert.equal(rejectForUnsuitableDateNight(
    {slot:'top',subtype:'t-shirt'},smartDate,resortTee),true);
  assert.equal(rejectForUnsuitableDateNight(
    {slot:'shoes',subtype:'minimal-sneaker',athletic:false},smartDate,retroTrainer),true);
  assert.equal(rejectForUnsuitableDateNight(
    {slot:'shoes',subtype:'minimal-sneaker',athletic:false},smartDate,cleanSneaker),false);
  assert.equal(rejectForUnsuitableDateNight(
    {slot:'top',subtype:'t-shirt'},casualDate,resortTee),false);
  assert.equal(rejectForUnsuitableDateNight(
    {slot:'shoes',subtype:'minimal-sneaker'},casualDate,retroTrainer),false);
  assert.equal(rejectUnpolishedDateNightLook([garment('tee','Plain White T-Shirt','top','t-shirt'),
    garment('chino','Grey Cotton Chinos','bottom','chino'),cleanSneaker],smartDate),true);
  assert.equal(rejectUnpolishedDateNightLook([garment('knit','Fine Merino Knit Sweater','top','fine-knit'),
    garment('chino','Grey Cotton Chinos','bottom','chino'),cleanSneaker],smartDate),false);
  assert.equal(rejectUnpolishedDateNightLook([garment('tee','Plain White T-Shirt','top','t-shirt'),
    garment('chino','Grey Cotton Chinos','bottom','chino'),cleanSneaker],casualDate),false);

  const catalog = [...products,resortTee,retroTrainer,cleanSneaker,
    garment('grey-knit','Fine Merino Knit Sweater Grey','top','fine-knit')];
  const result = await recommendFashionDeck({
    ...base,catalog,
    request:{sourceMode:'catalog',style:'minimalist',occasion:'Date night',
      weather:{},optionalLayer:true,preferences:{},requiredIds:[]},
  });
  assert.ok(result.outfits.length > 0);
  for(const look of result.outfits) {
    assert.ok(look.items.every(item=>!['real-resort-tee','real-retro-shoe'].includes(item.id)));
    assert.equal(rejectUnpolishedDateNightLook(look.items,smartDate),false);
  }
});


test('complete fashion decks expose a bounded whole-outfit assessment before Discovery selection', async()=>{
  const result = await recommendFashionDeck(base);
  assert.ok(result.outfits.length>=2);
  assert.equal(result.diagnostics.editorialQualityVersion,'s6-editorial-quality-v1');
  const lead=result.outfits[0];
  assert.ok(Number.isFinite(lead.fashion.editorial?.score));
  assert.ok(lead.fashion.editorial.score>=0 && lead.fashion.editorial.score<=1);
  for(const outfit of result.outfits) {
    const quality=outfit.fashion.editorial;
    assert.ok(quality&&Number.isFinite(quality.score));
    assert.ok(quality.adjustment>=-.08 && quality.adjustment<=.04);
    if(outfit.direction.id==='discovery'){
      assert.ok(quality.score>=.71);
      assert.ok(quality.score>=lead.fashion.editorial.score-.055);
    }
  }
  const rerun = await recommendFashionDeck(base);
  assert.deepEqual(rerun.outfits.map(x=>({id:x.id,quality:x.fashion.editorial})),
    result.outfits.map(x=>({id:x.id,quality:x.fashion.editorial})));
});

test('editorial v1 is isolated to New Clothing and does not rewrite mixed-source ranking', async()=>{
  const fromShop=await recommendFashionDeck(base);
  const mixed=await recommendFashionDeck({
    ...base, request:{...base.request,sourceMode:'mixed'},
  });
  assert.ok(fromShop.outfits.length>0&&mixed.outfits.length>0);
  assert.ok(fromShop.outfits.every(x=>Number.isFinite(x.fashion.editorial?.score)));
  assert.ok(mixed.outfits.every(x=>x.fashion.editorial.score===null&&x.fashion.editorial.adjustment===0));
});

test('Stylist 6 end-to-end returns three meaningfully named cards with short grounded reasoning',async()=>{
  const deck=await recommendFashionDeck(base);
  assert.equal(deck.diagnostics.deckCopyVersion,'s6-grounded-deck-copy-v5');
  assert.equal(deck.status,'complete');
  assert.equal(deck.outfits.length,3);
  assert.equal(new Set(deck.outfits.map(x=>x.concept.title.toLowerCase())).size,3);
  assert.ok(deck.outfits.every(x=>typeof x.description==='string'&&x.description.length>=35&&x.description.length<=300));
  assert.ok(deck.outfits.every(x=>!/(we saw your photo|your body type|perfect fit|guaranteed stock)/i.test(x.description)));
  assert.ok(deck.outfits[0].description.includes('Closest to your'));
  for(const card of deck.outfits.slice(1)){
    if(card.direction.id==='discovery') assert.ok(card.description.includes('new styling direction'));
    if(card.direction.id==='refresh') assert.ok(card.description.includes('different interpretation'));
    if(card.direction.id==='alternative') assert.ok(card.description.includes('not a separate discovery direction'));
  }
});

test('quiet smart styling declines explicit technical leisure outerwear, without a global jacket ban',()=>{
  const jacket = garment('tech','Navy Technical Leisure Jacket','outerwear','jacket');
  const ordinary = garment('blazer','Navy Technical Wool Blazer','outerwear','blazer');
  const desc = {slot:'outerwear',subtype:'jacket'};
  const quiet = {sourceMode:'catalog', style:'classic', dressCode:'smart',
    occasion:'Work',request:''};
  assert.equal(rejectQuietSmartLeisureLayer(desc,quiet,jacket),true);
  assert.equal(rejectQuietSmartLeisureLayer(desc,quiet,ordinary),false);
  assert.equal(rejectQuietSmartLeisureLayer(desc,{...quiet,style:'old_money',occasion:'Date night'},jacket),true);
  assert.equal(rejectQuietSmartLeisureLayer(desc,{...quiet,style:'old_money',occasion:'Casual',dressCode:'casual'},jacket),true,
    'the real old-money weekend preview must not sell a technical leisure blazer');
  const denimLeisure=garment('aurelien-16011498979658','Dark Blue Denim Leisure Blazer','outerwear','blazer');
  assert.equal(rejectQuietSmartLeisureLayer({slot:'outerwear',subtype:'blazer'},
    {...quiet,style:'old_money',occasion:'Casual',dressCode:'casual'},denimLeisure),true);
  assert.equal(rejectQuietSmartLeisureLayer({slot:'outerwear',subtype:'blazer'},
    {...quiet,style:'old_money',occasion:'Casual',dressCode:'casual',request:'Denim leisure blazer please'},denimLeisure),false);
  assert.equal(rejectQuietSmartLeisureLayer({slot:'outerwear',subtype:'blazer'},
    {...quiet,style:'casual',occasion:'Casual',dressCode:'casual'},denimLeisure),false);
  assert.equal(rejectQuietSmartLeisureLayer(desc,{...quiet,style:'old_money',occasion:'Casual',dressCode:'casual',request:'I want a technical leisure jacket'},jacket),false);
  assert.equal(rejectQuietSmartLeisureLayer(desc,{...quiet,style:'casual',occasion:'Travel'},jacket),false);
  assert.equal(rejectQuietSmartLeisureLayer(desc,{...quiet,sourceMode:'wardrobe'},jacket),false);
  assert.equal(rejectQuietSmartLeisureLayer(desc,{...quiet,request:'I want a technical leisure jacket'},jacket),false);
  assert.equal(rejectQuietSmartLeisureLayer({slot:'bottom'},quiet,jacket),false);
});

test('owner 2026-11-13: semi-classic smart date excludes clearly named reversible resort shell',()=>{
  const shell=garment('rust-resort','Rust Reversible Resort Jacket','outerwear','jacket');
  const tailored=garment('resort-blazer','Unstructured Resort Linen Blazer','outerwear','blazer');
  const reversibleTailored=garment('reversible-tailoring','Reversible Wool Tailored Blazer','outerwear','blazer');
  const descriptor={slot:'outerwear',subtype:'jacket'};
  const date={sourceMode:'catalog',style:'semi_classic',dressCode:'smart',
    occasion:'Romantic Date',request:''};
  assert.equal(rejectQuietSmartLeisureLayer(descriptor,date,shell),true);
  assert.equal(rejectQuietSmartLeisureLayer(descriptor,date,tailored),false);
  assert.equal(rejectQuietSmartLeisureLayer(descriptor,date,reversibleTailored),false);
  assert.equal(rejectQuietSmartLeisureLayer(descriptor,{...date,style:'casual'},shell),false);
  assert.equal(rejectQuietSmartLeisureLayer(descriptor,{...date,dressCode:'casual'},shell),false);
  assert.equal(rejectQuietSmartLeisureLayer(descriptor,{...date,sourceMode:'wardrobe'},shell),false);
  assert.equal(rejectQuietSmartLeisureLayer(descriptor,{...date,request:'I want a reversible resort jacket'},shell),false);
});

test('classic work deck avoids explicit technical leisure shells, but still returns valid looks',async()=>{
  const tech = garment('tech','Navy Technical Leisure Jacket','outerwear','jacket');
  const catalog=[...products,tech];
  const request={...base.request,weather:{temp:15},budget:{amount:1250000,currency:'UZS'},optionalLayer:false};
  const deck=await recommendFashionDeck({...base,request,catalog});
  assert.ok(deck.outfits.length>0);
  assert.ok(deck.outfits.every(look=>!look.items.some(item=>item.id===tech.id)));
});

test('quiet-style smart requests reject known clunky soles and unsolicited denim only where appropriate',()=>{
  const date={sourceMode:'catalog',style:'old_money',dressCode:'smart',occasion:'Date night',request:''};
  const work={...date,style:'semi_classic',occasion:'Work'};
  const jean={slot:'bottom',subtype:'jeans',casualSole:false};
  const loafer={slot:'shoes',subtype:'loafer',casualSole:true,soleProfile:'comfort'};
  const derby={slot:'shoes',subtype:'derby',casualSole:false,soleProfile:'dress'};
  assert.equal(rejectPolishedContextMismatch(jean,date,{}),true);
  assert.equal(rejectPolishedContextMismatch(jean,{...date,request:'Wear dark jeans'},{}),false);
  assert.equal(rejectPolishedContextMismatch(jean,{...date,style:'casual'},{}),false);
  assert.equal(rejectPolishedContextMismatch(loafer,date,{}),true);
  assert.equal(rejectPolishedContextMismatch(loafer,work,{}),true);
  assert.equal(rejectPolishedContextMismatch(derby,date,{}),false);
  assert.equal(rejectPolishedContextMismatch(loafer,{...date,dressCode:'casual'},{}),false);
  assert.equal(rejectPolishedContextMismatch(loafer,{...work,sourceMode:'wardrobe'},{}),false);
  assert.equal(rejectPolishedContextMismatch(loafer,{...work,request:'chunky loafers please'},{}),false);
});

test('Old Money smart date does not recommend known casual-soled shoes or unrequested jeans', async()=>{
  const catalog=[
    ...products,
    garment('jeans-green','Dark green straight denim jeans','bottom','jeans'),
    garment('geox-u66mac0009bc9999','Flextride Plus Leather Loafers','shoes','loafer'),
  ];
  const result=await recommendFashionDeck({
    ...base,catalog,
    request:{...base.request,style:'old_money',occasion:'Date night',brief:'Date night',
      requiredIds:[],preferences:{},budget:undefined},
  });
  assert.ok(result.outfits.length>=1);
  assert.ok(result.outfits.every(look => look.items.every(
    item => !['jeans-green','geox-u66mac0009bc9999'].includes(item.id))));
});

// ---------------------------------------------------------------------------
// Gate D: explicit feedback taste (s6-feedback-taste-v2). The adjustment is
// capped, decayed and applied after every hard gate; without it the deck is
// byte-identical legacy ranking.
// ---------------------------------------------------------------------------

test('feedback adjustment surfaces in scoring diagnostics only when explicit rows exist', async () => {
  const now = Date.now();
  const ratedAt = new Date(now - 86_400_000).toISOString();
  const rows = [
    { rating: 1, item_ids: ['shirt-navy', 'trouser-grey', 'derby-navy'], concept_id: 'precise-office', direction: 'signature', created_at: ratedAt },
    { rating: 1, item_ids: ['shirt-navy', 'trouser-grey', 'derby-navy'], concept_id: 'precise-office', direction: 'signature', created_at: ratedAt },
    { rating: -1, item_ids: ['shirt-white'], concept_id: 'precise-office', direction: 'signature', created_at: ratedAt },
    { rating: -1, item_ids: ['shirt-white'], concept_id: 'precise-office', direction: 'signature', created_at: ratedAt },
  ];
  const request = { ...base.request, requiredIds: [] };
  const run = (rows) => recommendFashionDeck({ ...base, request, ...(rows ? { feedbackTaste: feedbackTasteModel(rows, now) } : {}) });
  const baseline = await run(null);
  const liked = await run(rows);
  assert.equal(baseline.diagnostics.feedbackTasteActive, false);
  assert.equal(baseline.diagnostics.feedbackTasteVersion, 's6-feedback-taste-v2');
  for (const look of baseline.outfits) {
    assert.equal(look.fashion.personalization.feedback, undefined);
  }
  assert.equal(liked.diagnostics.feedbackTasteActive, true);
  const diagnostics = liked.outfits.map((look) => look.fashion.personalization.feedback).filter(Boolean);
  assert.ok(diagnostics.length >= 1, 'the adjustment must be visible per look');
  for (const entry of diagnostics) {
    assert.equal(entry.version, 's6-feedback-taste-v2');
    assert.ok(Math.abs(entry.adjustment) <= 0.06);
    assert.ok(entry.matches.every((match) => match.type === 'feedback_item'));
  }
  // Signature selection follows the liked lead without breaking any deck
  // invariant: complete deck, budgeted, same-currency, feasible garments.
  assert.equal(liked.status, 'complete');
  const signature = liked.outfits[0].items;
  assert.ok(signature.some(({ id }) => id === 'shirt-navy'), 'the liked shirt leads');
  assert.ok(signature.every(({ id }) => id !== 'shirt-white'), 'the twice-disliked shirt is avoided');
  assert.ok(signature.reduce((sum, { price }) => sum + price, 0) <= 750000);
});

test('opposite feedback flips deck ordering over the identical catalog', async () => {
  const now = Date.now();
  const ratedAt = new Date(now - 86_400_000).toISOString();
  const row = (rating, ids) => ({ rating, item_ids: ids, concept_id: 'precise-office', direction: 'signature', created_at: ratedAt });
  const liked = [row(1, ['shirt-white', 'trouser-grey']), row(1, ['shirt-white', 'trouser-grey'])];
  const disliked = [row(-1, ['shirt-white', 'trouser-grey']), row(-1, ['shirt-white', 'trouser-grey'])];
  const request = { ...base.request, requiredIds: [] };
  const run = (rows) => recommendFashionDeck({ ...base, request, feedbackTaste: feedbackTasteModel(rows, now) });
  const a = await run(liked);
  const b = await run(disliked);
  const serial = (result) => result.outfits.map((look) => look.items.map(({ id }) => id).join('+')).join('|');
  assert.notEqual(serial(a), serial(b), 'reversed taste must reorder the deck');
});

test('feedback is adjustment-only: a locked anchor survives a doubled dislike', async () => {
  const now = Date.now();
  const ratedAt = new Date(now - 86_400_000).toISOString();
  const rows = [
    { rating: -1, item_ids: ['shirt-white'], created_at: ratedAt },
    { rating: -1, item_ids: ['shirt-white'], created_at: ratedAt },
  ];
  const result = await recommendFashionDeck({
    ...base, feedbackTaste: feedbackTasteModel(rows, now),
  });
  assert.equal(result.status, 'complete');
  for (const look of result.outfits) {
    assert.ok(look.items.some(({ id }) => id === 'shirt-white'), 'locked anchor dropped');
  }
  // The dislike is still honestly visible in the anchor look's diagnostics.
  const anchorFeedback = result.outfits[0].fashion.personalization.feedback;
  assert.ok(anchorFeedback && anchorFeedback.adjustment < 0);
});

test('inert or malformed feedback models never perturb ranking', async () => {
  const baseline = await recommendFashionDeck(base);
  for (const feedbackTaste of [null, undefined, {}, { active: false }, feedbackTasteModel([], Date.now())]) {
    const result = await recommendFashionDeck({ ...base, feedbackTaste });
    assert.deepEqual(result.outfits, baseline.outfits);
  }
});

test('elevated register covers date nights and dampens unrequested henleys/tank tops cleanly', () => {
  assert.equal(elevatedFootwearRegisterContext({ style: 'old_money', occasion: 'Date night' }), true);
  assert.equal(elevatedFootwearRegisterContext({ style: 'smart_casual', occasion: 'Date night' }), true);
  assert.equal(elevatedFootwearRegisterContext({ style: 'classic', occasion: 'Wedding' }), true);
  const henley = garment('h1', 'Ribbed Long Sleeved Henley Shirt', 'top', 'shirt');
  const tank = garment('t1', 'Ribbed High Neck Tank Top', 'top', 'top');
  const poplin = garment('s1', 'Slim Fit Cotton Dress Shirt', 'top', 'shirt');
  const biz = { sourceMode: 'catalog', style: 'classic', occasion: 'Business meeting', request: '' };
  assert.equal(elevatedBasePolishPenaltyForLook([henley], [], biz), ELEVATED_BASE_POLISH_PENALTY);
  assert.equal(elevatedBasePolishPenaltyForLook([poplin], [], biz), 0);
  assert.equal(elevatedBasePolishPenaltyForLook([tank], [], { sourceMode: 'catalog', style: 'smart_casual', occasion: 'Date night', request: '' }), ELEVATED_BASE_POLISH_PENALTY);
  assert.equal(elevatedBasePolishPenaltyForLook([tank], [], { sourceMode: 'catalog', style: 'smart_casual', occasion: 'Date night', request: 'layered tank top' }), 0);
  assert.equal(elevatedBasePolishPenaltyForLook([henley], [], { sourceMode: 'wardrobe', style: 'classic', occasion: 'Business meeting', request: '' }), 0);
});

test('quiet-office finish is a bounded rank signal, not a gate or a wardrobe override', async()=>{
  const unanchored = await recommendFashionDeck({
    ...base, request:{...base.request,requiredIds:[],weather:{temp:18},
      budget:{amount:1200000,currency:'UZS'},preferences:{}},
  });
  assert.ok(unanchored.outfits.length>0);
  assert.equal(unanchored.diagnostics.fashionFinishVersion,'s6-quiet-office-finish-v1');
  for(const look of unanchored.outfits){
    const finish=look.fashion.finish;
    assert.equal(finish.active,true);
    assert.ok(finish.adjustment>=-.045&&finish.adjustment<=.038);
    assert.ok(finish.evidence.every(x=>typeof x==='string'));
  }
  const locked=await recommendFashionDeck(base);
  assert.ok(locked.outfits.length>0);
  assert.ok(locked.outfits.every(look=>look.fashion.finish.active===false &&
    look.fashion.finish.adjustment===0));
});


test('hosted visual challengers include different known outerwear shapes when rescue capacity ties',()=>{
  const alternative=(id,score)=>({id,items:[],fashion:{score}});
  const same=alternative('blazer-duplicate',.93);
  const different=alternative('real-overcoat-direction',.90);
  assert.deepEqual(rankVisualChallengerPool([
    {look:same,rescueGain:0,capacityGain:0,outerwearGain:0,unseenTop:1,difference:.5},
    {look:different,rescueGain:0,capacityGain:0,outerwearGain:1,unseenTop:0,difference:.4},
  ]).map(x=>x.id),['real-overcoat-direction','blazer-duplicate']);
});


test('independently verified product photos prioritize an explicitly permitted personal palette', () => {
  const colored=(id,color,slot)=>{
    const row=garment(id,'Color unspecified '+slot,slot,slot==='outerwear'?'blazer':slot==='bottom'?'tailored-trouser':slot==='shoes'?'derby':'shirt');
    const hash='f'.repeat(32);
    return {...row,visualSourceHash:hash,visualFashion:{version:1,sourceHash:hash,
      imageUrl:row.imageUrl,imageSha256:'a'.repeat(64),model:'siglip2',revision:'color-r1',
      attributes:{color:{value:color,confidence:0.93,margin:0.63,evidence:'image-model'}}}};
  };
  const allPermitted={items:[colored('p1','cream','top'),colored('p2','brown','outerwear'),
    colored('p3','beige','bottom'),colored('p4','black','shoes')]};
  const preference={favoriteColors:['cream','beige','brown'],avoidColors:['navy','grey']};
  const safe=photographicPaletteEvidence(allPermitted,preference);
  assert.equal(safe.verified,4);
  assert.equal(safe.preferred,3);
  assert.equal(safe.blocked,0);
  assert.ok(safe.score>1);
  const unknown={items:[...allPermitted.items.slice(0,2),
    garment('unknown-pants','Unknown tailored pants','bottom','tailored-trouser'),
    garment('unknown-shoes','Leather footwear','shoes','derby')]};
  const sparse=photographicPaletteEvidence(unknown,preference);
  assert.equal(sparse.verified,2);
  assert.ok(sparse.score<safe.score,'unknown photo colors must not outrank actual permitted photo evidence');
  const violating={items:[...allPermitted.items.slice(0,2),colored('navy','navy','bottom'),
    allPermitted.items[3]]};
  const banned=photographicPaletteEvidence(violating,preference);
  assert.equal(banned.blocked,1);
  assert.equal(banned.score,-1,'known excluded photographic color is not an alternative');
  const anchored=photographicPaletteEvidence(violating,preference,['navy']);
  assert.equal(anchored.blocked,0,'a specifically pinned garment is a direct user instruction');
  const noPreference=photographicPaletteEvidence(allPermitted,{});
  assert.ok(noPreference.score>=1,'no policy is invented when the user did not specify exclusions');
});


test('visual rescue inventory counts only independently styled triples with different actual SKUs',()=>{
  const make=(id,layer)=>({id,items:[
    garment(id+'-top','Refined upper garment '+id,'top',
      ({one:'shirt',two:'fine-knit',three:'polo',four:'dress-shirt'})[id]),
    garment(layer,'Wool Cardigan '+layer,'outerwear','cardigan'),
    garment(id+'-pants','Refined trousers '+id,'bottom',
      ({one:'tailored-trouser',two:'chino',three:'trouser',four:'jeans'})[id]),
    garment(id+'-shoes','Leather Derby '+id,'shoes','derby'),
  ]});
  const looks=[make('one','shared-cardigan'),make('two','shared-cardigan'),
    make('three','third-cardigan')];
  assert.equal(visualProductUniqueTripleCount(looks),0,
    'three NVIDIA-reviewed outfit architectures are not a unique deck if two reuse a SKU');
  assert.equal(visualProductUniqueTripleCount([...looks,make('four','fourth-cardigan')]),2);
  assert.ok(visualRescueTripleCount(looks)>0,'architecture difference alone is weaker evidence');
  assert.equal(visualProductUniqueTripleCount(looks,['shared-cardigan']),1,
    'the exact owner-requested piece may repeat without pretending two new merchant items are different');
  const candidates=[{look:{id:'unique',fashion:{score:.88}},uniqueTripleGain:2,
    rescueGain:1,capacityGain:0,difference:.6},
    {look:{id:'reused',fashion:{score:.93}},uniqueTripleGain:0,
    rescueGain:1,capacityGain:0,difference:.6}];
  assert.deepEqual(rankVisualChallengerPool(candidates).map(c=>c.id),['unique','reused']);
});


test('polished +13C dinner needs four intentional garments, while warm, casual and explicit no-jacket requests stay personal', async()=>{
  const baseDate={sourceMode:'catalog',style:'semi_classic',occasion:'Date night',
    brief:'Refined semi-classic outfit for a nice dinner date with polished footwear',
    weather:{temp:13},optionalLayer:true,preferences:{avoidColors:['navy','grey']}};
  const context=fashionStylingContext(baseDate);
  assert.equal(polishedEveningLayerRequired(context),true);
  assert.equal(polishedEveningLayerRequired(fashionStylingContext({...baseDate,weather:{temp:25}})),false);
  assert.equal(polishedEveningLayerRequired(fashionStylingContext({...baseDate,occasion:'Weekend'})),false);
  assert.equal(polishedEveningLayerRequired(fashionStylingContext({...baseDate,brief:'Relaxed casual walk and a date'})),false);
  assert.equal(polishedEveningLayerRequired(fashionStylingContext({...baseDate,brief:'Polished dinner, without jacket'})),false);
  assert.equal(polishedEveningLayerRequired(fashionStylingContext({...baseDate,sourceMode:'wardrobe'})),false);
  const r=await recommendFashionDeck({...base,request:{...baseDate,preferences:{}}});
  assert.ok(r.outfits.length>0,'source-backed catalog should furnish a date outfit in the known fixture');
  assert.ok(r.outfits.every(o=>o.items.some(item=>item.macroCategory==='outerwear') && o.items.length===4),
    'no bare short-sleeve polo and dress trousers at +13°C in a refined dinner deck');
});


test('strict personal colors preferentially spend paid visual calls on photographically verified safe alternatives',async()=>{
  const hash='d'.repeat(32);
  const withPhotoColor=item=>{
    const seen=/(?:white|navy|cream|grey|beige|brown|black|charcoal|tan|burgundy)/i.exec(item.id)?.[0]?.toLowerCase()||'brown';
    return {...item,visualSourceHash:hash,visualFashion:{version:1,sourceHash:hash,
      imageUrl:item.imageUrl,imageSha256:createHash('sha256').update(item.id).digest('hex'),
      model:'siglip2',revision:'qa-safe-1',
      attributes:{color:{value:seen==='charcoal'?'grey':seen,confidence:.94,margin:.53,evidence:'image-model'}}}};
  };
  const catalog=[...products.map(withPhotoColor),
    garment('camel-cardigan','Fine Wool Camel Cardigan','outerwear','cardigan'),
    garment('beige-chino','Beige cotton chinos','bottom','chino'),
    garment('cream-polo','Cream fine-knit polo','top','polo')].map(withPhotoColor);
  const preferences={avoidColors:['navy'],favoriteColors:['cream','beige','brown']};
  const request={...base.request,preferences};
  let sent=[];
  const result=await recommendFashionDeck({catalog,request,referenceBriefs:refs,
    hardGate:async()=>({ok:true}),priceToBudgetCurrency:i=>i.price,
    visualDeckReview:true,canJudgeOutfit:async()=>true,
    visualCritic:async batch=>{
      sent=batch;
      return batch.map(look=>admittedVisualVerdict(look,{
        slotColors:Object.fromEntries(look.items.map(i=>[i.macroCategory,'brown'])),
        referenceless:true,styleFit:.87,occasionFit:.91,formalityFit:.88,
      }));
    },
  });
  assert.ok(sent.length>=4,'creative fashion shortlist must provide photo-safe alternatives for NVIDIA');
  assert.ok(sent.every(look=>!photographicPaletteEvidence(look,preferences,request.requiredIds).blocked),
    'a known banned photographic hue must not consume one of eight 120B calls');
  assert.ok(sent.filter(look=>{const p=photographicPaletteEvidence(look,preferences,request.requiredIds);return p.verified>=Math.min(3,p.total)}).length>=3,
    'the model should see multiple actual-photo grounded alternatives, not only uncertain-color text metadata');
  assert.ok(sent.length<=8);
});


test('AI Fashion Director chooses real shortlisted whole looks before independent photo critique', async()=>{
  let called=0,reviewed=[];
  const request={sourceMode:'catalog',style:'classic',occasion:'Work',
    weather:{temp:20},optionalLayer:true,preferences:{},
    budget:{amount:1200000,currency:'UZS'},requiredIds:[]};
  const deck=await recommendFashionDeck({
    ...base,request,referenceBriefs:[],visualDeckReview:true,
    canJudgeOutfit:async()=>true,
    selectDirectorFinalists:async({options,context})=>{
      called++;
      assert.ok(options.length>=6,'real complete candidates must exist for the model');
      assert.equal(context.style,'classic');
      assert.ok(options.every(x=>!Object.hasOwn(x,'look')&&!Object.hasOwn(x,'id')),
        'neither raw SKUs nor original look objects reach the hosted selector');
      return {status:'selected',tokens:options.slice(0,6).map(x=>x.token),providerCalls:1};
    },
    visualCritic:async batch=>{
      reviewed=batch;return batch.map(candidate=>admittedVisualVerdict(candidate));
    },
  });
  assert.equal(called,1);
  assert.equal(deck.diagnostics.modelFashionSelection.status,'selected');
  assert.equal(deck.diagnostics.modelFashionSelection.chosen,6);
  assert.ok(reviewed.length>=6&&reviewed.length<=8);
  assert.ok(deck.outfits.every(look=>reviewed.some(x=>x.id===look.id)));
  assert.ok(deck.outfits.every(look=>Array.isArray(look.visualEvidence)),
    'only independently photo-approved model choices are displayed');
});
test('AI Fashion Director provider timeout restores legacy editorial admission',async()=>{
  const request={sourceMode:'catalog',style:'classic',occasion:'Work',
    weather:{temp:20},optionalLayer:true,preferences:{},
    budget:{amount:1200000,currency:'UZS'},requiredIds:[]};
  const deck=await recommendFashionDeck({
    ...base,request,referenceBriefs:[],visualDeckReview:true,
    canJudgeOutfit:async()=>true,
    selectDirectorFinalists:async()=>({status:'timeout',tokens:[],providerCalls:1}),
    visualCritic:async batch=>batch.map(candidate=>admittedVisualVerdict(candidate)),
  });
  assert.equal(deck.diagnostics.modelFashionSelection.status,'timeout');
  assert.equal(deck.diagnostics.modelFashionSelection.chosen,0);
  assert.ok(deck.outfits.every(look=>Array.isArray(look.visualEvidence)),
    'an AI outage is not an unreviewed-fashion approval');
});


test('AI-led catalog defers stylistic product vetoes but preserves user hard exclusions',()=>{
  const {item,context}={item:garment('cotton-tee','Plain Cotton T-shirt','top','t-shirt'),
    context:{sourceMode:'catalog',style:'classic',occasion:'work',dressCode:'smart',preferences:{}}};
  // Import the domain gate via a direct dynamic module to prove its real
  // default contract is unchanged for production S5 and legacy S6.
  return import('../../src/domain/garmentContext.ts').then(({garmentContextReason})=>{
    assert.equal(garmentContextReason(item,context),'work_requires_smart_base');
    assert.equal(garmentContextReason(item,context,{deferStylistic:true}),null);
    assert.equal(garmentContextReason(item,{...context,preferences:{avoidColors:['red']}},
      {deferStylistic:true}),null);
    const red=garment('red','Red Cotton T-shirt','top','t-shirt');
    assert.equal(garmentContextReason(red,{...context,preferences:{avoidColors:['red']}},
      {deferStylistic:true}),'excluded_color');
  });
});


test('model-led taste deferral never leaks banned product into legacy timeout recovery',async()=>{
  const improbable=garment('terry-resort-top',
    'Yellow Terry Towelling Resort Shirt','top','shirt');
  const request={sourceMode:'catalog',style:'classic',occasion:'Work',
    weather:{temp:20},optionalLayer:true,preferences:{},
    budget:{amount:1200000,currency:'UZS'},requiredIds:[]};
  let called=0;
  const result=await recommendFashionDeck({
    ...base,catalog:[...products,improbable],request,referenceBriefs:[],
    visualDeckReview:true,canJudgeOutfit:async()=>true,
    selectDirectorFinalists:async()=>{
      called++;return {status:'timeout',tokens:[],providerCalls:1};
    },
    visualCritic:async batch=>batch.map(c=>admittedVisualVerdict(c)),
  });
  assert.equal(called,1);
  assert.equal(result.diagnostics.modelFashionSelection.status,'timeout');
  assert.ok(result.outfits.every(look=>
    look.items.every(item=>item.id!==improbable.id)),
    'timeout must restore full old product-level stylistic admission, not only the late cross-slot veto');
});

test('AI photo review portfolio rescues three truly disjoint, independent outfits before paid vision',()=>{
  const mk=(id,top,bottom,shoes,layer)=>({
    id,briefId:layer?'layer':'base',fashion:{score:0.89},
    items:[
      garment(id+'-top',top+' shirt','top',top),
      ...(layer?[garment(id+'-layer',layer+' coat','outerwear',layer)]:[]),
      garment(id+'-bottom',bottom+' trousers','bottom',bottom),
      garment(id+'-shoes',shoes+' shoes','shoes',shoes),
    ],
  });
  const clones=Array.from({length:6},(_,i)=>({
    ...mk('clone-'+i,'shirt','trouser','loafer',null),
    items:[garment('COMMON-TOP','White Shirt','top','shirt'),
      garment('COMMON-BOTTOM','Grey Trousers','bottom','trouser'),
      garment('shoe-'+i,'Loafer '+i,'shoes','loafer')],
    fashion:{score:0.95},
  }));
  const knit=mk('knit','fine-knit','chino','chelsea','cardigan');
  const suit=mk('suit','dress-shirt','tailored-trouser','oxford','blazer');
  const relaxed=mk('relaxed','polo','jeans','minimal-sneaker',null);
  const options=[...clones,knit,suit,relaxed];
  assert.equal(visualProductUniqueTripleCount(clones),0);
  const trio=chooseIndependentPhotoReviewTrio(options,clones);
  assert.equal(trio.length,3);
  assert.equal(visualProductUniqueTripleCount(trio),1);
  assert.ok(trio.some(x=>x.id==='knit'));
  assert.ok(trio.some(x=>x.id==='suit'));
  assert.ok(trio.filter(x=>!x.id.startsWith('clone-')).length>=2,
    'the slate must include two structural directions rather than repeated base items');
  assert.ok(trio.every(look=>options.some(x=>x.id===look.id)));
});

test('photo critic gets a portfolio of independent SKU-unique rescue looks, not repeated model favorites',()=>{
  const mk=(i,top,bottom,shoe,layer)=>({
    id:'portfolio-'+i,briefId:'portfolio-brief-'+i,
    fashion:{score:0.94-i*0.006},
    items:[
      garment('portfolio-'+i+'-top',top+' shirt','top',top),
      ...(layer?[garment('portfolio-'+i+'-coat',layer+' layer','outerwear',layer)]:[]),
      garment('portfolio-'+i+'-bottom',bottom+' trouser','bottom',bottom),
      garment('portfolio-'+i+'-shoes',shoe+' shoes','shoes',shoe),
    ],
  });
  const goods=[
    mk(0,'dress-shirt','tailored-trouser','oxford','blazer'),
    mk(1,'fine-knit','chino','loafer','cardigan'),
    mk(2,'polo','jeans','chelsea','coat'),
    mk(3,'shirt','trouser','derby','overshirt'),
    mk(4,'dress-shirt','chino','loafer','coat'),
    mk(5,'fine-knit','trouser','chelsea','blazer'),
    mk(6,'shirt','tailored-trouser','derby','cardigan'),
    mk(7,'polo','jeans','oxford','overshirt'),
  ];
  const nominated=goods.slice(0,6);
  const slate=assembleIndependentVisualPortfolio(goods,nominated,[],undefined,8);
  assert.equal(slate.length,8,'all eight paid slots can reach independent photos');
  assert.ok(visualProductUniqueTripleCount(slate)>0);
  assert.ok(visualPortfolioSingleVetoResilience(slate)>0,
    'any ONE rejected NVIDIA candidate must still leave a real independently purchasable triple');
  for(let i=0;i<slate.length;i++)for(let j=i+1;j<slate.length;j++){
    assert.ok(slate[i].items.every(a=>slate[j].items.every(b=>a.id!==b.id)),
      'no single SKU should be used in multiple paid-review slots when inventory suffices');
  }
});

test('catalog-only public fashion title context never leaks from owned wardrobe preview',()=>{
  const testLook={id:'abc',briefId:'house',fashion:{score:0.91},
    items:[
      garment('private-top','PUBLIC Resort Terry Shirt','top','shirt'),
      garment('private-bottom','Stone Tailored Trousers','bottom','trouser'),
      garment('private-shoes','Brown Leather Loafers','shoes','loafer'),
    ]};
  const privateLook=directorSlatePreview([testLook]);
  assert.equal(privateLook[0].items[0].merchantLabel,undefined,
    'default and wardrobe preview must never transmit user-owned names');
  const catalogLook=directorSlatePreview([testLook],[],
    {includePublicCatalogNames:true});
  assert.ok(catalogLook[0].items[0].merchantLabel.includes('Resort Terry'),
    'public catalog titles should tell the AI the garment purpose');
  assert.equal(catalogLook[0].qualityHint,0.91);
});


test('visual portfolio resilience is grounded in product-unique triples, not fabricated approvals',()=>{
  assert.equal(visualPortfolioSingleVetoResilience([],[]),0);
  assert.equal(visualPortfolioSingleVetoResilience([{id:'x'}],[]),0);
  const make=(n,slot)=>garment('r-'+slot+'-'+n,slot+' '+n,slot,slot);
  const three=[0,1,2].map(n=>({
    id:'slot-'+n,items:[
      make(n,'top'),make(n,'bottom'),make(n,'shoes')],
    fashion:{score:0.91},
  }));
  assert.equal(visualPortfolioSingleVetoResilience(three),0,
    'a three-card slate has no backup even when all photos pass');
  const broken=[...three,{...three[0],id:'duplicate',items:three[0].items}];
  assert.equal(visualPortfolioSingleVetoResilience(broken),0,
    'a repeat of one source product is not a genuine independent backup');
});
