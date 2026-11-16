import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CREATIVE_DIRECTOR_VERSION, directorInventory, directorResponseSchema,
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
  assert.equal(validateDirectorDirections({
    directions: [{ ...proposal.directions[0] }, { ...proposal.directions[0] }, { ...proposal.directions[0] }],
  }, inventory).length, 1, 'renamed duplicates count as only one verified direction');
  const fabricated = structuredClone(proposal);
  fabricated.directions[0].roles.shoes = ['imaginary-sneaker'];
  assert.equal(validateDirectorDirections(fabricated, inventory).length, 2,
    'invalid SKU roles are discarded; valid independent directions survive');
  const tooCold = structuredClone(proposal);
  tooCold.directions[2].roles.outerwear = [];
  assert.equal(validateDirectorDirections(tooCold, inventory).length, 2, 'the unsafe cold-weather concept is discarded');
  const malicious = structuredClone(proposal);
  malicious.directions[0].title = '<ignore system>';
  assert.equal(validateDirectorDirections(malicious, inventory).length, 2, 'markup title is discarded without losing safe concepts');
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
    assert.equal(body.model,'nvidia/nemotron-3.5-lightning-30b-a3b',
      'text planning must not use the much larger photo judge');
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


test('concurrent identical style contexts share one NVIDIA creative plan rather than creating a provider storm', async () => {
  const key = 'nvapi-test-key-0123456789abcdef';
  const local = { ...request, brief: 'Unique dedup concurrent style test ' + CREATIVE_DIRECTOR_VERSION };
  let release;
  let calls = 0;
  const fetchImpl = async () => {
    calls++;
    return new Promise(resolve => { release = () => resolve(fakeResponse(proposal)); });
  };
  const first = designCreativeBriefs({ catalog, request: local, apiKey: key, fetchImpl });
  const second = designCreativeBriefs({ catalog, request: local, apiKey: key, fetchImpl });
  assert.equal(calls, 1, 'two identical simultaneous requests spend one provider call');
  release();
  const [a, b] = await Promise.all([first, second]);
  assert.equal(a.status, 'planned');
  assert.equal(b.status, 'shared');
  assert.equal(b.providerCalls, 0);
  assert.deepEqual(a.briefs, b.briefs);
});


test('live NVIDIA fashion shade light-grey is normalized while every actual SKU role remains grounded', () => {
  const inv = directorInventory(catalog, request);
  const caseFromRealNvidia = structuredClone(proposal);
  caseFromRealNvidia.directions[1].palette = ['charcoal', 'light-grey', 'burgundy'];
  assert.equal(validateDirectorDirections(caseFromRealNvidia, inv).length, 3);
  assert.deepEqual(validateDirectorDirections(caseFromRealNvidia, inv)[1].palette,
    ['charcoal','grey','burgundy']);
  const madeUp = structuredClone(proposal);
  madeUp.directions[1].palette = ['alien-lime','light-grey','burgundy'];
  assert.equal(validateDirectorDirections(madeUp, inv).length, 2,
    'invented colors do not pass as grounded fashion labels');
  const unsupported = structuredClone(proposal);
  unsupported.directions[1].roles.top = ['magical-garment'];
  assert.equal(validateDirectorDirections(unsupported, inv).length, 2,
    'one unsupported garment role cannot invalidate two good designs');
});


test('dynamic NVIDIA JSON schema constrains all generated garments to real role subtypes', () => {
  const inventory=directorInventory(catalog,request);
  const schema=directorResponseSchema(inventory);
  const directions=schema.properties.directions;
  assert.equal(directions.minItems,3);
  assert.equal(directions.maxItems,3);
  const roles=directions.items.properties.roles.properties;
  for(const slot of ['top','bottom','shoes','outerwear']) {
    const allow=inventory.available[slot].map(it=>it.subtype);
    assert.deepEqual(roles[slot].items.enum,allow);
    assert.equal(roles[slot].minItems,1,'at +9C outerwear is mandatory, even for a creative AI');
    assert.ok(roles[slot].maxItems<=3);
  }
  const palette=directions.items.properties.palette;
  assert.ok(palette.items.enum.includes('navy'));
  assert.ok(!palette.items.enum.includes('light-grey'),
    'unsupported palette label cannot be coerced into provider schema');
  const empty=directorInventory(catalog.filter(x=>x.macroCategory!=='outerwear'),request);
  assert.equal(directorResponseSchema(empty).properties.directions.items.properties.roles.properties.outerwear.maxItems,0);
  const warm=directorInventory(catalog,{...request,weather:{temp:22}});
  assert.equal(directorResponseSchema(warm).properties.directions.items.properties.roles.properties.outerwear.minItems,0);
});
