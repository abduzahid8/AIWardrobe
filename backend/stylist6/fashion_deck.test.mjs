import test from 'node:test';
import assert from 'node:assert/strict';
import { adjacentCreativeStyles, applyExclusionQualityFloor, buildPriceProfile, fashionStylingContext, pricePenaltyForLook, recommendFashionDeck, rejectForUnsuitableDateNight, rejectUnpolishedDateNightLook, singleCurrencyBonus } from './fashion_deck.mjs';
import { devLookFeasibility, sourceHash } from './catalog_adapter.mjs';
import { shortlistRetrievalDeck } from './prototype.mjs';

function garment(id, name, macroCategory, subtype, extra = {}) {
  return { id, name, macroCategory, subtype, isShopItem: true, currency: 'UZS', price: 100,
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
    optionalLayer: true, referenceId: 'rt_001', budget: { amount: 300, currency: 'UZS' },
    preferences: { avoidColors: ['black'] }, requiredIds: ['shirt-white'] },
};

test('warm work outfits keep the exact anchor, avoid excluded colors and omit the unnecessary layer', async () => {
  const result = await recommendFashionDeck(base);
  assert.equal(result.status, 'complete');
  assert.equal(result.diagnostics.visualVerdict, 'NOT_RUN');
  for (const look of result.outfits) {
    assert.equal(look.items.length, 3);
    assert.ok(look.items.some(({ id }) => id === 'shirt-white'));
    assert.ok(look.items.every(({ id }) => !id.includes('black') && id !== 'training-shoe'));
    assert.ok(look.items.reduce((sum, { price }) => sum + price, 0) <= 300);
    assert.ok(Number.isFinite(look.fashion.referenceMetadata.slotColorFit));
  }
  assert.ok(new Set(result.outfits.flatMap(({ items }) => items.map(({ id }) => id))).size >= 7);
});

test('cold looks retain layers and an infeasible budget never gains a fallback outfit', async () => {
  const request = { ...base.request, weather: { temp: 9 }, budget: { amount: 400, currency: 'UZS' } };
  const cold = await recommendFashionDeck({ ...base, request });
  assert.equal(cold.status, 'complete');
  assert.ok(cold.outfits.every(({ items }) => items.length === 4 && items.some(({ id }) => id === 'shirt-white')));
  const impossible = await recommendFashionDeck({ ...base, request: { ...request, budget: { amount: 399, currency: 'UZS' } } });
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


test('an unspecified wedding uses formal composition while a beach wedding or explicit code stays distinct', async () => {
  const request = { ...base.request, occasion: 'Wedding', weather: { temp: 21 }, budget: { amount: 400, currency: 'UZS' } };
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
  const premium = garment('derby-premium', 'black leather derby shoes', 'shoes', 'derby', { price: 50000 });
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
  assert.ok(hit.penalty > 0 && hit.penalty <= 0.06);
  const mild = pricePenaltyForLook([{ ...shoe('m', 150) }], profile);
  assert.equal(mild.flagged, 0);
  assert.ok(mild.penalty > 0 && mild.penalty < hit.penalty);
  const capped = pricePenaltyForLook(
    [{ ...shoe('a', 100000) }, { ...shoe('b', 90000) }, { ...shoe('c', 80000) }], profile);
  assert.equal(capped.penalty, 0.06);
  assert.deepEqual(pricePenaltyForLook([{ ...shoe('usp', 100000) }], {}), { penalty: 0, flagged: 0 });
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
      optionalLayer: true, preferences: {}, budget: { amount: 10000, currency: 'UZS' } },
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
    assert.ok(look.items.reduce((sum,i) => sum + i.price, 0) <= 300);
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
      requiredIds:[],budget:{amount:400,currency:'UZS'},
      preferences:{}, // no large form, no guessed favorite colors/brands
    },
  });
  assert.ok(result.outfits.length > 0);
  assert.equal(result.diagnostics.neighborBriefs,2);
  assert.equal(result.diagnostics.directionVersion,'s6-three-directions-editorial-v2');
  assert.equal(result.outfits[0].direction.id,'signature');
  for(const outfit of result.outfits){
    assert.ok(['signature','refresh','discovery','alternative'].includes(outfit.direction.id));
    assert.ok(outfit.items.every(item=>item.id!=='snow-parka'&&item.id!=='training-shoe'));
    assert.ok(outfit.items.reduce((total,item)=>total+item.price,0)<=400);
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
    assert.ok(look.items.reduce((total,item)=>total+item.price,0)<=300);
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
