import test from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import {
  generateStylist6Deck,
  handleDevBridgeSwapSlot,
  handleDevBridgeAnalyzeAppearance,
  createDevBridgeServer,
  hydrateFullLiveCatalog,
  fetchLiveCbuFxSnapshot,
  clearDevBridgeCaches,
  mapDevVisualStatus,
  referenceFileFor,
  STYLIST6_DEV_ENGINE_VERSION,
} from './dev_bridge.mjs';
import { sourceHash } from '../../backend/stylist6/catalog_adapter.mjs';

test('visual statuses distinguish applied, unchanged, abstained and not-run decks', () => {
  assert.equal(mapDevVisualStatus({ diagnostics: { visualVerdict: 'NOT_RUN' }, outfits: [] }), 'NVIDIA_NOT_RUN');
  assert.equal(mapDevVisualStatus({ diagnostics: { visualVerdict: 'ABSTAINED' }, outfits: [] }), 'NVIDIA_ABSTAINED');
  assert.equal(
    mapDevVisualStatus({ diagnostics: { visualVerdict: 'JUDGED' }, outfits: [{ visualEvidence: ['a', 'b', 'c'] }] }),
    'NVIDIA_APPLIED',
  );
  assert.equal(
    mapDevVisualStatus({ diagnostics: { visualVerdict: 'JUDGED' }, outfits: [{ id: 's6_x' }] }),
    'NVIDIA_EVALUATED_NO_CHANGE',
  );
  assert.equal(mapDevVisualStatus(null), 'NVIDIA_NOT_RUN');
});

test('reference admission requires a real local photo, not a metadata entry', () => {
  // rt_001 has a metadata entry and a matching local lookbook photo.
  assert.ok(referenceFileFor('rt_001')?.endsWith('.jpg'));
  // Unknown ids (metadata without bytes) fail admission honestly.
  assert.equal(referenceFileFor('rt_999'), null);
  assert.equal(referenceFileFor(''), null);
});

test('local bridge engine identity is bumped after screenshot-driven remediation',()=>{
  assert.equal(STYLIST6_DEV_ENGINE_VERSION,'stylist-6-dev-local-20261112-r3');
});

const bridgeCatalogFixture = () => [
  { id: 't1', name: 'White Oxford Dress Shirt', category: 'tops', garment_type: 'upper_body', fashion_slot: 'top', fashion_subtype: 'dress-shirt', price: 650000, currency: 'UZS', image_url: 'https://frgroup.uz/media/t1.jpg', source_url: 'https://frgroup.uz/ru/store/t1', brand: 'Boss', is_active: true },
  { id: 't2', name: 'Cream Fine-Knit Merino Sweater', category: 'tops', garment_type: 'upper_body', fashion_slot: 'top', fashion_subtype: 'fine-knit', price: 720000, currency: 'UZS', image_url: 'https://frgroup.uz/media/t2.jpg', source_url: 'https://frgroup.uz/ru/store/t2', brand: 'Massimo Dutti', is_active: true },
  { id: 't3', name: 'Sky Blue Cotton Poplin Shirt', category: 'tops', garment_type: 'upper_body', fashion_slot: 'top', fashion_subtype: 'shirt', price: 590000, currency: 'UZS', image_url: 'https://frgroup.uz/media/t3.jpg', source_url: 'https://frgroup.uz/ru/store/t3', brand: 'GANT', is_active: true },
  { id: 'o1', name: 'Navy Wool Tailored Blazer', category: 'tops', garment_type: 'upper_body', fashion_slot: 'outerwear', fashion_subtype: 'blazer', price: 2400000, currency: 'UZS', image_url: 'https://frgroup.uz/media/o1.jpg', source_url: 'https://frgroup.uz/ru/store/o1', brand: 'Boss', is_active: true },
  { id: 'o2', name: 'Camel Hair Coat Overcoat', category: 'tops', garment_type: 'upper_body', fashion_slot: 'outerwear', fashion_subtype: 'coat', price: 3200000, currency: 'UZS', image_url: 'https://frgroup.uz/media/o2.jpg', source_url: 'https://frgroup.uz/ru/store/o2', brand: 'Massimo Dutti', is_active: true },
  { id: 'o3', name: 'Charcoal Fine Merino Cardigan', category: 'tops', garment_type: 'upper_body', fashion_slot: 'outerwear', fashion_subtype: 'cardigan', price: 1500000, currency: 'UZS', image_url: 'https://frgroup.uz/media/o3.jpg', source_url: 'https://frgroup.uz/ru/store/o3', brand: 'GANT', is_active: true },
  { id: 'b1', name: 'Charcoal Tailored Trousers', category: 'bottoms', garment_type: 'lower_body', fashion_slot: 'bottom', fashion_subtype: 'tailored-trouser', price: 950000, currency: 'UZS', image_url: 'https://frgroup.uz/media/b1.jpg', source_url: 'https://frgroup.uz/ru/store/b1', brand: 'Boss', is_active: true },
  { id: 'b2', name: 'Stone Tailored Wool Trousers', category: 'bottoms', garment_type: 'lower_body', fashion_slot: 'bottom', fashion_subtype: 'tailored-trouser', price: 780000, currency: 'UZS', image_url: 'https://frgroup.uz/media/b2.jpg', source_url: 'https://frgroup.uz/ru/store/b2', brand: 'GANT', is_active: true },
  { id: 'b3', name: 'Navy Pleated Dress Trousers', category: 'bottoms', garment_type: 'lower_body', fashion_slot: 'bottom', fashion_subtype: 'tailored-trouser', price: 990000, currency: 'UZS', image_url: 'https://frgroup.uz/media/b3.jpg', source_url: 'https://frgroup.uz/ru/store/b3', brand: 'Massimo Dutti', is_active: true },
  { id: 's1', name: 'Brown Suede Loafers', category: 'shoes', garment_type: 'shoes', fashion_slot: 'shoes', fashion_subtype: 'loafer', price: 1450000, currency: 'UZS', image_url: 'https://frgroup.uz/media/s1.jpg', source_url: 'https://frgroup.uz/ru/store/s1', brand: 'Boss', is_active: true },
  { id: 's2', name: 'Black Leather Derby Shoes', category: 'shoes', garment_type: 'shoes', fashion_slot: 'shoes', fashion_subtype: 'derby', price: 1650000, currency: 'UZS', image_url: 'https://frgroup.uz/media/s2.jpg', source_url: 'https://frgroup.uz/ru/store/s2', brand: 'Massimo Dutti', is_active: true },
  { id: 's3', name: 'Dark Brown Chelsea Boots', category: 'shoes', garment_type: 'shoes', fashion_slot: 'shoes', fashion_subtype: 'chelsea', price: 1750000, currency: 'UZS', image_url: 'https://frgroup.uz/media/s3.jpg', source_url: 'https://frgroup.uz/ru/store/s3', brand: 'GANT', is_active: true },
];

test('generateStylist6Deck attaches offerVerification, language, direction and editorialRegister meta', async () => {
  const res = await generateStylist6Deck({
    request: { style: 'classic', occasion: 'Work', brief: 'Work, classic', language: 'ru' },
    catalogRows: bridgeCatalogFixture(),
  }, { skipHydration: true });
  assert.equal(res.status, 200);
  assert.equal(res.body.success, true);
  assert.equal(res.body.language, 'ru');
  assert.equal(res.body.sourceMode, 'catalog');
  assert.ok(res.body.outfits.length >= 1);
  assert.equal(res.body.meta.language, 'ru');
  assert.equal(res.body.meta.offerTruth.version, 's6-merchant-offers-v1');
  assert.ok(res.body.meta.editorialRegister?.version);
  for (const outfit of res.body.outfits) {
    assert.ok('direction' in outfit);
    assert.ok(outfit.offerVerification);
    for (const item of outfit.items) {
      assert.equal(item.offerVerification.offerStatus, 'unverified_catalog_listing');
      assert.equal(item.offerVerification.purchaseUrlValid, true);
      assert.equal(item.offerVerification.checkoutAvailability, 'unverified');
    }
  }
});

test('generateStylist6Deck enforces requiredIds anchor and supports mixed/wardrobe modes', async () => {
  const missing = await generateStylist6Deck({
    request: { style: 'classic', occasion: 'Work', brief: 'Work', requiredIds: ['does-not-exist'] },
    catalogRows: bridgeCatalogFixture(),
  }, { skipHydration: true });
  assert.equal(missing.status, 422);
  assert.equal(missing.body.code, 'anchor_not_found');

  const mixed = await generateStylist6Deck({
    request: { style: 'classic', occasion: 'Work', brief: 'Work', sourceMode: 'mixed', requiredIds: ['w-shirt-anchor'] },
    catalogRows: bridgeCatalogFixture(),
    wardrobeItems: [
      { id: 'w-shirt-anchor', name: 'White Poplin Shirt', macroCategory: 'top', type: 'shirt', category: 'tops', color: 'white', imageUrl: 'https://frgroup.uz/media/w1.jpg' },
    ],
  }, { skipHydration: true });
  assert.equal(mixed.status, 200);
  assert.equal(mixed.body.sourceMode, 'mixed');
  for (const look of mixed.body.outfits) {
    assert.ok(look.items.some((item) => item.id === 'w-shirt-anchor' && item.isWardrobe === true && item.isShopItem === false));
  }

  const wardrobeOnly = await generateStylist6Deck({
    request: { style: 'classic', occasion: 'Work', brief: 'Work', sourceMode: 'wardrobe' },
    catalogRows: [],
    wardrobeItems: [
      { id: 'wt1', name: 'White Shirt', macroCategory: 'top', type: 'shirt', category: 'tops', color: 'white', imageUrl: 'https://example.com/wt1.jpg' },
      { id: 'wt2', name: 'Cream Knit Sweater', macroCategory: 'top', type: 'fine-knit', category: 'tops', color: 'cream', imageUrl: 'https://example.com/wt2.jpg' },
      { id: 'wo1', name: 'Navy Blazer', macroCategory: 'outerwear', type: 'blazer', category: 'tops', color: 'navy', imageUrl: 'https://example.com/wo1.jpg' },
      { id: 'wb1', name: 'Charcoal Trousers', macroCategory: 'bottom', type: 'tailored trousers', category: 'bottoms', color: 'charcoal', imageUrl: 'https://example.com/wb1.jpg' },
      { id: 'wb2', name: 'Beige Chinos', macroCategory: 'bottom', type: 'chino', category: 'bottoms', color: 'beige', imageUrl: 'https://example.com/wb2.jpg' },
      { id: 'ws1', name: 'Brown Loafers', macroCategory: 'shoes', type: 'loafer', category: 'shoes', color: 'brown', imageUrl: 'https://example.com/ws1.jpg' },
      { id: 'ws2', name: 'Black Derbies', macroCategory: 'shoes', type: 'derby', category: 'shoes', color: 'black', imageUrl: 'https://example.com/ws2.jpg' },
    ],
  }, { skipHydration: true });
  assert.equal(wardrobeOnly.status, 200);
  assert.equal(wardrobeOnly.body.sourceMode, 'wardrobe');
  for (const look of wardrobeOnly.body.outfits) {
    assert.ok(look.items.every((item) => item.isWardrobe === true && item.isShopItem === false));
  }
});

test('generateStylist6Deck supports visualDeckReview with textual blueprints or local reference media', async () => {
  const bigJpeg = await sharp({
    create: { width: 256, height: 256, channels: 3, noise: { type: 'gaussian', mean: 128, sigma: 25 } },
  }).jpeg({ quality: 92 }).toBuffer();
  assert.ok(bigJpeg.length >= 1024);
  const rows = bridgeCatalogFixture().map((row) => ({
    ...row,
    image_url: row.image_url.replace('https://frgroup.uz/media/', 'https://media.frgroup.uz/images/'),
  }));
  const fetchImpl = async (url, init) => {
    const target = String(url);
    if (target.startsWith('https://media.frgroup.uz/')) {
      return new Response(bigJpeg, { status: 200, headers: { 'Content-Type': 'image/jpeg' } });
    }
    const body = JSON.parse(init.body);
    const promptText = body.messages[0].content[0].text;
    const ctx = JSON.parse(promptText.split('\n').find((line) => line.startsWith('{"style"')));
    const candidateId = promptText.match(/candidateId must be exactly "([^"]+)"/)?.[1];
    const slotObservations = Object.fromEntries(
      ctx.imageSlots.map((slot) => [slot, `clean ${slot} garment rendered with accurate silhouette`]),
    );
    const referenceless = promptText.includes('No reference image is supplied');
    const s = ctx.imageSlots;
    const refSlots = [...new Set([s[0], ...(ctx.verifiedReferenceOnlySlots || [])])];
    const relationships = referenceless
      ? [
          { kind: 'outfit', slots: [s[0], s[1]], observation: 'Top and second piece harmonize in tone and proportion' },
          { kind: 'outfit', slots: [s[0], s[2]], observation: 'Upper garment pairs cleanly with tailored lower silhouette' },
          { kind: 'outfit', slots: [s[s.length - 2], s[s.length - 1]], observation: 'Trouser hem balances footwear shape cleanly' },
        ]
      : [
          { kind: 'reference', slots: refSlots, observation: 'Silhouette follows the reference formality cleanly' },
          { kind: 'outfit', slots: [s[0], s[1]], observation: 'Upper garments layer smoothly with balanced collar contrast' },
          { kind: 'outfit', slots: [s[s.length - 2], s[s.length - 1]], observation: 'Lower tailoring anchors the footwear proportion cleanly' },
        ];
    const verdict = {
      candidateId,
      slotObservations,
      relationships,
      visualCohesion: 0.9,
      ...(referenceless ? {} : { referenceFidelity: 0.85 }),
      styleFit: 0.9,
      occasionFit: 0.9,
      formalityFit: 0.9,
      confidence: 0.9,
      reject: false,
      evidence: relationships.map((r) => r.observation),
    };
    return new Response(JSON.stringify({
      choices: [{ finish_reason: 'stop', message: { content: JSON.stringify(verdict) } }],
      usage: { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };
  const res = await generateStylist6Deck({
    request: { style: 'classic', occasion: 'Work', brief: 'Work, classic', enableVisual: true },
    catalogRows: rows,
  }, { skipHydration: true, nvidiaApiKey: 'nvapi-test-key-0123456789abcdef', fetchImpl });
  assert.equal(res.status, 200);
  assert.equal(res.body.visualStatus, 'NVIDIA_APPLIED');
  assert.equal(res.body.compositionProtocol, 'four-slot-v1');
  assert.equal(res.body.visualReceipt.displayedJudged, res.body.outfits.length);
  assert.ok(res.body.visualReceipt.validVerdicts >= res.body.outfits.length);
  assert.equal(res.body.visualReceipt.visualApproval, 'full_three_verified');
  assert.ok(res.body.outfits.every((look) =>
    look.items.length === 4 &&
    look.selectionSource === 'ai' &&
    look.fashionDiagnostics.visualVerdict === 'JUDGED' &&
    typeof look.stylistNote === 'string' && look.stylistNote.length >= 15 &&
    typeof look.whyItFlattersYou === 'string' &&
    Array.isArray(look.stylingTips) &&
    typeof look.colorStory === 'string' && look.colorStory.length > 0 &&
    typeof look.silhouetteFormula === 'string' && look.silhouetteFormula.length > 0));
});

test('V3 dev_bridge runs full hygiene + PersonalStyleDna + visualAtelierCompose + exact 4-item contract + Stylist Notes', async () => {
  const bigJpeg = await sharp({
    create: { width: 256, height: 256, channels: 3, noise: { type: 'gaussian', mean: 128, sigma: 25 } },
  }).jpeg({ quality: 92 }).toBuffer();

  const rows = [
    ...bridgeCatalogFixture().map((row) => ({
      ...row,
      image_url: row.image_url.replace('https://frgroup.uz/media/', 'https://media.frgroup.uz/images/'),
    })),
    // Defective rows that filterHygienicCatalog MUST strip:
    {
      id: 'bad-dead-pdp', name: 'Armani Dead Homepage Shirt', category: 'tops', garment_type: 'upper_body',
      fashion_slot: 'top', fashion_subtype: 'dress-shirt', price: 600000, currency: 'UZS',
      image_url: 'https://media.frgroup.uz/images/bad1.jpg', source_url: 'https://www.armani.com/en-us/',
      brand: 'Armani', is_active: true,
    },
    {
      id: 'bad-lowres', name: 'Low Res Shirt', category: 'tops', garment_type: 'upper_body',
      fashion_slot: 'top', fashion_subtype: 'shirt', price: 500000, currency: 'UZS',
      image_url: 'https://media.frgroup.uz/images/bad2.jpg?w=262&h=393', source_url: 'https://frgroup.uz/ru/store/bad2',
      brand: 'C&A', is_active: true,
    },
  ];

  let atelierCalled = 0;
  const visualAtelierCompose = async ({ catalog }) => {
    atelierCalled += 1;
    const byId = (id) => catalog.find((item) => item.id === id);
    return {
      status: 'applied',
      provider: 'openrouter',
      model: 'google/gemini-2.5-flash',
      latencyMs: 420,
      outfits: [
        { directionIndex: 0, title: 'Signature Tailored', top: byId('t1'), outerwear: byId('o1'), bottom: byId('b1'), shoes: byId('s2') },
        { directionIndex: 1, title: 'Warm Camel Layer', top: byId('t2'), outerwear: byId('o2'), bottom: byId('b2'), shoes: byId('s1') },
        { directionIndex: 2, title: 'Refined Knit Smart', top: byId('t3'), outerwear: byId('o3'), bottom: byId('b3'), shoes: byId('s3') },
      ],
    };
  };

  const fetchImpl = async (url, init) => {
    const target = String(url);
    if (target.startsWith('https://media.frgroup.uz/')) {
      return new Response(bigJpeg, { status: 200, headers: { 'Content-Type': 'image/jpeg' } });
    }
    const body = JSON.parse(init.body);
    const content = body.messages?.[0]?.content;
    if (!Array.isArray(content)) {
      return new Response(JSON.stringify({ error: 'critic_only_mock' }), { status: 503 });
    }
    const promptText = content[0]?.text || '';
    const ctxLine = promptText.split('\n').find((line) => line.startsWith('{"style"'));
    if (!ctxLine) {
      return new Response(JSON.stringify({ error: 'critic_only_mock' }), { status: 503 });
    }
    const ctx = JSON.parse(ctxLine);
    const candidateId = promptText.match(/candidateId must be exactly "([^"]+)"/)?.[1];
    const s = ctx.imageSlots;
    const verdict = {
      candidateId,
      slotObservations: Object.fromEntries(s.map((slot) => [slot, `clean ${slot} garment`])),
      slotColors: Object.fromEntries(s.map((slot) => [slot, slot === 'top' ? 'white' : slot === 'outerwear' ? 'navy' : slot === 'bottom' ? 'grey' : 'brown'])),
      relationships: [
        { kind: 'outfit', slots: [s[0], s[1]], observation: 'Upper layers harmonize cleanly' },
        { kind: 'outfit', slots: [s[0], s[2]], observation: 'Top pairs cleanly with trousers' },
        { kind: 'outfit', slots: [s[2], s[3]], observation: 'Trouser hem balances shoes cleanly' },
      ],
      visualCohesion: 0.92,
      styleFit: 0.91,
      occasionFit: 0.93,
      formalityFit: 0.92,
      confidence: 0.91,
      editorialScore: 8.9,
      reject: false,
      evidence: ['Clean full-garment proportion and color rhythm'],
    };
    return new Response(JSON.stringify({
      choices: [{ finish_reason: 'stop', message: { content: JSON.stringify(verdict) } }],
      usage: { prompt_tokens: 120, completion_tokens: 60, total_tokens: 180 },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };

  const res = await generateStylist6Deck({
    request: {
      style: 'classic',
      occasion: 'Work',
      brief: 'Executive boardroom',
      presentation: 'menswear',
      bodyProfile: { gender: 'male', heightCm: 182, weightKg: 78, bodyType: 'athletic' },
      appearanceProfile: {
        consentGranted: true,
        skinUndertone: 'warm',
        contrastLevel: 'high',
        colorSeason: 'deep_autumn',
        flatteringColors: ['camel', 'navy', 'cream'],
      },
      preferences: { favoriteColors: ['#0A1931', 'camel'], avoidColors: ['pink'] },
      enableVisual: true,
    },
    catalogRows: rows,
  }, {
    skipHydration: true,
    nvidiaApiKey: 'nvapi-test-key-0123456789abcdef',
    fetchImpl,
    visualAtelierCompose,
    allowLocalShowroomVisualComposition: true,
  });

  assert.equal(res.status, 200);
  assert.equal(atelierCalled, 1, 'visualAtelierCompose must run in local dev bridge');
  assert.equal(res.body.visualAtelierStatus, 'applied');
  assert.equal(res.body.diagnostics.visualAtelierStatus, 'applied');
  assert.equal(res.body.compositionProtocol, 'four-slot-v1');
  assert.equal(res.body.outfits.length, 3);
  const allIds = new Set(res.body.outfits.flatMap((o) => o.items.map((it) => it.id)));
  assert.equal(allIds.has('bad-dead-pdp'), false, 'dead homepage PDP must be stripped');
  assert.equal(allIds.has('bad-lowres'), false, 'low-res image must be stripped');
  assert.equal(allIds.size, 12, 'all 3 looks must be 4-item SKU-disjoint');
  for (const outfit of res.body.outfits) {
    assert.equal(outfit.items.length, 4);
    assert.ok(outfit.stylistNote.length >= 20);
    assert.ok(outfit.whyItFlattersYou.length >= 15);
    assert.ok(outfit.colorStory.length >= 8);
    assert.ok(outfit.silhouetteFormula.length >= 8);
    assert.ok(Array.isArray(outfit.stylingTips) && outfit.stylingTips.length >= 1);
  }
});

test('dev_bridge supports action: swap_slot via POST /swap-slot and POST /generate with FX budget, locked 3 slots & updated Stylist Notes', async () => {
  const todayUz = new Date(Date.now() + 5 * 3600 * 1000).toISOString().slice(0, 10);
  const fxSnap = {
    id: `CBU:${todayUz}:12800:13600:16100`,
    source: 'CBU',
    asOf: todayUz,
    fetchedAt: new Date().toISOString(),
    uzsPerUnit: { UZS: 1, USD: 12800, EUR: 13600, GBP: 16100 },
  };
  const catalogRows = [
    ...bridgeCatalogFixture(),
    {
      id: 's4', name: 'Burgundy Calfskin Monk Straps', category: 'shoes', garment_type: 'shoes',
      fashion_slot: 'shoes', fashion_subtype: 'monk', price: 110, currency: 'EUR',
      image_url: 'https://frgroup.uz/media/s4.jpg', source_url: 'https://frgroup.uz/ru/store/s4',
      brand: 'Boss', is_active: true,
    },
  ];
  const deckRes = await generateStylist6Deck({
    request: { style: 'classic', occasion: 'Work', brief: 'Work, classic', budget: { amount: 700, currency: 'USD' } },
    catalogRows,
  }, { skipHydration: true, fxSnapshot: fxSnap });
  assert.equal(deckRes.status, 200);
  const firstLook = deckRes.body.outfits[0];
  assert.equal(firstLook.items.length, 4);
  const oldShoe = firstLook.items.find((i) => i.macroCategory === 'shoes');
  assert.ok(oldShoe);

  // Test via HTTP server on BOTH /swap-slot and /generate (with action: 'swap_slot')
  const server = createDevBridgeServer({ liveServer: false, skipHydration: true, fxSnapshot: fxSnap });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const baseUrl = `http://127.0.0.1:${server.address().port}`;
  try {
    const healthRes = await fetch(`${baseUrl}/health`);
    assert.equal(healthRes.status, 200);
    const health = await healthRes.json();
    assert.equal(health.ok, true);
    assert.equal(health.engine, STYLIST6_DEV_ENGINE_VERSION);
    assert.ok(health.endpoints.includes('/swap-slot'));
    assert.ok(health.endpoints.includes('/analyze-appearance'));

    const swapPayload = {
      outfit: firstLook,
      targetSlot: 'shoes',
      reason: 'wrong_shoes',
      context: { style: 'classic', occasion: 'Work', sourceMode: 'catalog', budget: { amount: 700, currency: 'USD' }, enableVisual: false },
      excludedIds: [oldShoe.id],
      catalogRows,
    };

    const res1 = await fetch(`${baseUrl}/swap-slot`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(swapPayload),
    });
    assert.equal(res1.status, 200);
    const body1 = await res1.json();
    assert.equal(body1.success, true);
    assert.equal(body1.action, 'swap_slot');
    assert.equal(body1.status, 'swapped');
    assert.equal(body1.swappedSlot, 'shoes');
    assert.equal(body1.previousItemId, oldShoe.id);
    assert.notEqual(body1.replacementItem.id, oldShoe.id);
    assert.equal(body1.outfit.items.length, 4);
    assert.equal(body1.currencyConversion.active, true);
    assert.ok(typeof body1.outfit.stylistNote === 'string' && body1.outfit.stylistNote.length >= 15);
    // Verify the 3 non-shoes locked slots stayed identical
    for (const locked of firstLook.items.filter((i) => i.macroCategory !== 'shoes')) {
      assert.ok(body1.outfit.items.some((i) => i.macroCategory === locked.macroCategory && i.id === locked.id));
    }

    // Also verify POST /generate with action: 'swap_slot'
    const res2 = await fetch(`${baseUrl}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'swap_slot', ...swapPayload }),
    });
    assert.equal(res2.status, 200);
    const body2 = await res2.json();
    assert.equal(body2.success, true);
    assert.equal(body2.action, 'swap_slot');
    assert.notEqual(body2.replacementItem.id, oldShoe.id);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('dev_bridge supports action: analyze_appearance via POST /analyze-appearance and POST /generate with fail-closed consent & caching', async () => {
  const portraitBytes = await sharp({
    create: { width: 256, height: 256, channels: 3, background: { r: 205, g: 168, b: 132 } },
  }).jpeg({ quality: 90 }).toBuffer();

  let providerCalls = 0;
  const fetchImpl = async (url) => {
    if (String(url) === 'https://studio.aiwardrobe.test/dev-portrait.jpg') {
      return new Response(portraitBytes, { status: 200, headers: { 'Content-Type': 'image/jpeg' } });
    }
    throw new Error(`unexpected url: ${url}`);
  };
  const server = createDevBridgeServer({
    liveServer: false,
    fetchImpl,
    dnsLookupFn: async () => [{ address: '93.184.216.34', family: 4 }],
    providerFn: async () => {
      providerCalls += 1;
      return {
        skinUndertone: 'warm',
        skinDepth: 'medium',
        contrastLevel: 'medium',
        colorSeason: 'warm_autumn',
        flatteringColors: ['camel', 'olive', 'cream'],
        powerNeutrals: ['brown', 'camel'],
        avoidColors: ['neon'],
        metalTone: 'gold',
        lightingQuality: 'good',
        confidence: 0.92,
      };
    },
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const baseUrl = `http://127.0.0.1:${server.address().port}`;
  try {
    // Consent denied -> 0 provider calls
    const denied = await fetch(`${baseUrl}/analyze-appearance`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ photoUrl: 'https://studio.aiwardrobe.test/dev-portrait.jpg', consentGranted: false }),
    });
    assert.equal(denied.status, 200);
    const deniedBody = await denied.json();
    assert.equal(deniedBody.status, 'consent_required');
    assert.equal(deniedBody.appearance, null);
    assert.equal(providerCalls, 0);

    // Consent granted -> analyzed
    const analyzed = await fetch(`${baseUrl}/analyze-appearance`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ photoUrl: 'https://studio.aiwardrobe.test/dev-portrait.jpg', consentGranted: true }),
    });
    assert.equal(analyzed.status, 200);
    const analyzedBody = await analyzed.json();
    assert.equal(analyzedBody.status, 'analyzed');
    assert.equal(analyzedBody.appearance.colorSeason, 'warm_autumn');
    assert.equal(analyzedBody.appearance.consentGranted, true);
    assert.equal(providerCalls, 1);

    // Second call via POST /generate with action: 'analyze_appearance' hits SHA-256 cache (0 new provider calls)
    const cached = await fetch(`${baseUrl}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action: 'analyze_appearance',
        photoUrl: 'https://studio.aiwardrobe.test/dev-portrait.jpg',
        consentGranted: true,
      }),
    });
    assert.equal(cached.status, 200);
    const cachedBody = await cached.json();
    assert.equal(cachedBody.status, 'analyzed');
    assert.equal(cachedBody.cached, true);
    assert.equal(providerCalls, 1);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('hydrateFullLiveCatalog (10-min TTL) & fetchLiveCbuFxSnapshot (15-min TTL) cache Supabase & CBU reads without affecting offline fixtures', async () => {
  clearDevBridgeCaches();
  const fixture = bridgeCatalogFixture();
  const t1Hash = sourceHash(fixture[0]);
  const todayUz = new Date(Date.now() + 5 * 3600 * 1000).toISOString().slice(0, 10);
  const fxSnap = {
    id: `CBU:${todayUz}:12800:13600:16100`,
    source: 'CBU',
    asOf: todayUz,
    fetchedAt: new Date().toISOString(),
    uzsPerUnit: { UZS: 1, USD: 12800, EUR: 13600, GBP: 16100 },
  };
  let catalogFetches = 0;
  let fxFetches = 0;
  const fetchImpl = async (url) => {
    const u = new URL(String(url));
    if (u.pathname.endsWith('/shop_catalog')) {
      catalogFetches += 1;
      return Response.json(fixture);
    }
    if (u.pathname.endsWith('/catalog_garment_intelligence')) {
      return Response.json(fixture.map((r) => ({ product_id: r.id, source_hash: sourceHash(r), slot: r.fashion_slot, subtype: r.fashion_subtype })));
    }
    if (u.pathname.endsWith('/catalog_visual_fashion')) {
      return Response.json([{
        product_id: fixture[0].id,
        version: 1,
        source_hash: t1Hash,
        image_url: fixture[0].image_url,
        profile: {
          version: 1,
          model: 'siglip2',
          revision: 'v1',
          imageSha256: 'a'.repeat(64),
          sourceHash: t1Hash,
          imageUrl: fixture[0].image_url,
          attributes: { color: { value: 'white', confidence: 0.94, margin: 0.6 } },
        },
      }]);
    }
    if (u.pathname.endsWith('/budget_exchange_rates')) {
      fxFetches += 1;
      return Response.json([{ snapshot: fxSnap }]);
    }
    throw new Error(`unexpected url: ${url}`);
  };

  const now = Date.now();
  const opts = {
    supabaseUrl: 'https://fake-supabase.example',
    supabaseAnonKey: 'anon-fixture',
    supabaseServiceKey: 'service-fixture',
    fetchImpl,
    now,
  };
  const firstCat = await hydrateFullLiveCatalog(opts);
  assert.equal(firstCat.status, 'ok');
  assert.equal(firstCat.totalRows, fixture.length);
  assert.equal(firstCat.visualAttached, 1);
  assert.equal(catalogFetches, 1);

  // Within 10-min TTL -> served from cache (0 extra HTTP calls)
  const secondCat = await hydrateFullLiveCatalog({ ...opts, now: now + 5 * 60 * 1000 });
  assert.equal(secondCat.status, 'cached');
  assert.equal(catalogFetches, 1);

  const firstFx = await fetchLiveCbuFxSnapshot(opts);
  assert.equal(firstFx.source, 'supabase_budget_exchange_rates');
  assert.equal(firstFx.snapshot.uzsPerUnit.USD, 12800);
  assert.equal(fxFetches, 1);

  // Within 15-min TTL -> served from cache (0 extra HTTP calls)
  const secondFx = await fetchLiveCbuFxSnapshot({ ...opts, now: now + 10 * 60 * 1000 });
  assert.equal(secondFx.source, 'cached');
  assert.equal(fxFetches, 1);

  // Live server mode auto-hydrates catalog when client omits catalogRows
  const liveRes = await generateStylist6Deck({
    request: { style: 'classic', occasion: 'Work', brief: 'Work, classic', budget: { amount: 650, currency: 'USD' } },
    catalogRows: [],
  }, { ...opts, liveServer: true, now: now + 60 * 1000 });
  assert.equal(liveRes.status, 200);
  assert.equal(liveRes.body.meta.catalogSource, 'supabase_live_cached');
  assert.equal(liveRes.body.currencyConversion.active, true);
  assert.equal(liveRes.body.outfits.length, 3);
  clearDevBridgeCaches();
});
