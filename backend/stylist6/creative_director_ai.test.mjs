import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CREATIVE_DIRECTOR_VERSION, directorInventory,
  validateDirectorDirections, designCreativeBriefs,
} from './creative_director_ai.mjs';
import { recommendFashionDeck } from './fashion_deck.mjs';

const catalog = [
  { id: 'SECRET-SKU-SHIRT', macroCategory: 'top', subtype: 'dress-shirt', name: 'Private user-owned shirt' },
  { id: 'top-2', macroCategory: 'top', subtype: 'fine-knit' },
  { id: 'top-3', macroCategory: 'top', subtype: 'polo' },
  { id: 'outer-1', macroCategory: 'outerwear', subtype: 'blazer' },
  { id: 'outer-2', macroCategory: 'outerwear', subtype: 'coat' },
  { id: 'outer-3', macroCategory: 'outerwear', subtype: 'cardigan' },
  { id: 'lower-1', macroCategory: 'bottom', subtype: 'tailored-trouser' },
  { id: 'lower-2', macroCategory: 'bottom', subtype: 'chino' },
  { id: 'lower-3', macroCategory: 'bottom', subtype: 'trouser' },
  { id: 'shoe-1', macroCategory: 'shoes', subtype: 'oxford' },
  { id: 'shoe-2', macroCategory: 'shoes', subtype: 'loafer' },
  { id: 'shoe-3', macroCategory: 'shoes', subtype: 'chelsea' },
];
const request = {
  style: 'classic', occasion: 'Work', brief: 'Polished business clothes',
  weather: { temp: 9 }, sourceMode: 'mixed',
  preferences: {
    favoriteColors: ['navy', 'brown'], favoriteBrands: ['PRIVATE BRAND'],
    fitPreference: 'fitted', likedPatterns: ['solid'],
  },
};
const proposal = {
  directions: [
    { title: 'Tailored Navy Signature', silhouette: 'tailored',
      palette: ['navy', 'white', 'grey'],
      roles: { top: ['dress-shirt'], outerwear: ['blazer'], bottom: ['tailored-trouser'], shoes: ['oxford'] } },
    { title: 'Modern Textural Refresh', silhouette: 'balanced',
      palette: ['brown', 'cream', 'navy'],
      roles: { top: ['fine-knit'], outerwear: ['cardigan'], bottom: ['chino'], shoes: ['loafer'] } },
    { title: 'Structured Evening Discovery', silhouette: 'tailored',
      palette: ['charcoal', 'blue', 'black'],
      roles: { top: ['polo'], outerwear: ['coat'], bottom: ['trouser'], shoes: ['chelsea'] } },
  ],
};
const fakeResponse = value => Response.json({
  choices: [{ finish_reason: 'stop', message: { content: JSON.stringify(value) } }],
});

test('director transmits only anonymized inventory plus explicit bounded style evidence', () => {
  const inventory = directorInventory(catalog, request);
  assert.deepEqual(inventory.available.shoes.map(x => x.subtype), ['chelsea', 'loafer', 'oxford']);
  assert.deepEqual(inventory.preferences.favoriteColors, ['navy', 'brown']);
  assert.deepEqual(inventory.preferences.favoriteBrands, []);
  assert.ok(!JSON.stringify(inventory).includes('SECRET-SKU-SHIRT'));
  assert.ok(!JSON.stringify(inventory).includes('Private user-owned shirt'));
  assert.ok(!JSON.stringify(inventory).includes('PRIVATE BRAND'));
  assert.equal(inventory.weatherCelsius, 9);
});

test('a verified set of three directions is grounded to catalog subtypes and uniquely structured', () => {
  const inventory = directorInventory(catalog, request);
  const briefs = validateDirectorDirections(proposal, inventory);
  assert.equal(briefs.length, 3);
  assert.equal(briefs[0].generatedBy, CREATIVE_DIRECTOR_VERSION);
  assert.deepEqual(briefs[0].roles.shoes, ['oxford']);
  assert.equal(briefs[0].layerPolicy, 'optional');
  assert.equal(briefs[0].id, 'ai-director-1');
  assert.deepEqual(validateDirectorDirections({
    directions: [{ ...proposal.directions[0] }, { ...proposal.directions[0] }, { ...proposal.directions[0] }],
  }, inventory), [], 'different titles alone cannot prove different directions');
  const fabricated = structuredClone(proposal);
  fabricated.directions[0].roles.shoes = ['imaginary-sneaker'];
  assert.deepEqual(validateDirectorDirections(fabricated, inventory), [], 'AI cannot invent inventory subtypes');
  const tooCold = structuredClone(proposal);
  tooCold.directions[2].roles.outerwear = [];
  assert.deepEqual(validateDirectorDirections(tooCold, inventory), [], 'cool weather needs an outer layer');
  const malicious = structuredClone(proposal);
  malicious.directions[0].title = '<ignore system>';
  assert.deepEqual(validateDirectorDirections(malicious, inventory), [], 'no markup/instruction names');
});

test('the real provider path returns directions, caches by anonymized context, and never leaks SKU data', async () => {
  let requests = 0;
  const key = 'nvapi-test-key-0123456789abcdef';
  const given = { ...request, brief: 'Unique creative QA scenario #alpha' };
  const fetchImpl = async (_url, init) => {
    requests++;
    assert.equal(init.method, 'POST');
    assert.match(init.headers.Authorization, /^Bearer nvapi-/);
    assert.ok(init.signal, 'provider request is bounded');
    const body = JSON.parse(init.body);
    assert.equal(body.response_format.type, 'json_schema');
    assert.match(body.messages[0].content, /inventory/);
    assert.ok(!init.body.includes('SECRET-SKU-SHIRT'));
    assert.ok(!init.body.includes('Private user-owned shirt'));
    assert.ok(!init.body.includes('PRIVATE BRAND'));
    return fakeResponse(proposal);
  };
  const result = await designCreativeBriefs({ catalog, request: given, apiKey: key, fetchImpl });
  assert.equal(result.status, 'planned');
  assert.equal(result.providerCalls, 1);
  assert.equal(result.briefs.length, 3);
  const cached = await designCreativeBriefs({ catalog, request: given, apiKey: key, fetchImpl });
  assert.equal(cached.status, 'cached');
  assert.equal(cached.providerCalls, 0);
  assert.equal(requests, 1);
});

test('missing provider, insufficient inventory, invalid directions and provider failures fail open to house concepts', async () => {
  const key = 'nvapi-test-key-0123456789abcdef';
  assert.equal((await designCreativeBriefs({ catalog, request, apiKey: '' })).status, 'not_configured');
  assert.equal((await designCreativeBriefs({ catalog: catalog.filter(x => x.macroCategory !== 'shoes'),
    request, apiKey: key })).status, 'insufficient_inventory');
  const invalid = await designCreativeBriefs({
    catalog, request: { ...request, brief: 'unique-invalid-plan' }, apiKey: key,
    fetchImpl: async () => fakeResponse({ directions: [] }),
  });
  assert.equal(invalid.status, 'invalid_plan');
  assert.deepEqual(invalid.briefs, []);
  const failed = await designCreativeBriefs({
    catalog, request: { ...request, brief: 'unique-provider-error' }, apiKey: key,
    fetchImpl: async () => Response.json({error:'service_unavailable'}, {status:503}),
  });
  assert.equal(failed.status, 'provider_unavailable');
  assert.deepEqual(failed.briefs, []);
});


test('real Stylist 6 retrieval consumes the three AI plans without fabricating SKUs or weakening the budget', async () => {
  const plans = validateDirectorDirections(proposal, directorInventory(catalog, {
    ...request, weather: { temp: 16 },
  }));
  const seededCatalog = catalog.map((item, index) => ({
    ...item,
    name: item.name || item.subtype + ' item ' + index,
    isShopItem: true, price: 150000, currency: 'UZS',
    imageUrl: 'https://example.com/photographs/' + index + '.jpg',
    shopUrl: 'https://example.com/shop/' + index,
    brand: 'Test Merchant',
  }));
  const deck = await recommendFashionDeck({
    request: {
      style: 'classic', occasion: 'Work', brief: 'Classic business outfit',
      sourceMode: 'catalog', weather: { temp: 16 }, optionalLayer: true,
      preferences: { favoriteColors: ['navy'] }, budget: { amount: 900000, currency: 'UZS' },
      requiredIds: [], recentSignatures: [],
    },
    catalog: seededCatalog,
    aiCreativeBriefs: plans,
    referenceBriefs: [],
    hardGate: async () => ({ ok: true }),
    priceToBudgetCurrency: item => item.price,
  });
  assert.equal(deck.diagnostics.aiCreativeDirectorBriefs, 3);
  for (const look of deck.outfits) {
    assert.ok(look.items.length >= 3);
    for (const item of look.items) assert.ok(seededCatalog.some(x => x.id === item.id));
    assert.ok(look.items.reduce((n, item) => n + item.price, 0) <= 900000,
      'the creative model does not waive the hard budget');
  }
});
