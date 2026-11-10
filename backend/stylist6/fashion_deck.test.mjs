import test from 'node:test';
import assert from 'node:assert/strict';
import { fashionStylingContext, recommendFashionDeck } from './fashion_deck.mjs';

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
