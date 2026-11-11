import test from 'node:test';
import assert from 'node:assert/strict';
import { buildPriceProfile, fashionStylingContext, pricePenaltyForLook, recommendFashionDeck } from './fashion_deck.mjs';
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
  assert.deepEqual(devLookFeasibility([ok]), { ok: true });
  assert.deepEqual(devLookFeasibility([{ ...ok, id: '  ' }]), { ok: false, reason: 'invalid_product_identity' });
  assert.deepEqual(devLookFeasibility([{ ...ok, macroCategory: 'hat' }]), { ok: false, reason: 'unknown_slot' });
  assert.deepEqual(devLookFeasibility([{ ...ok, imageUrl: 'http://plain/photo.jpg' }]), { ok: false, reason: 'unrenderable_photo' });
  assert.deepEqual(devLookFeasibility([{ ...ok, shopUrl: 'not-a-url' }]), { ok: false, reason: 'unverified_merchant_offer' });
  assert.deepEqual(devLookFeasibility([{ ...ok, price: 0 }]), { ok: false, reason: 'unpriced_or_currency' });
  assert.deepEqual(devLookFeasibility([{ ...ok, currency: 'usd' }]), { ok: false, reason: 'unpriced_or_currency' });
  assert.deepEqual(devLookFeasibility(null), { ok: true });
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
