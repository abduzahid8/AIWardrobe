import test from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import {
  generateStylist6Deck,
  mapDevVisualStatus,
  referenceFileFor,
  STYLIST6_DEV_ENGINE_VERSION,
} from './dev_bridge.mjs';

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
  { id: 'o3', name: 'Charcoal Wool Overshirt', category: 'tops', garment_type: 'upper_body', fashion_slot: 'outerwear', fashion_subtype: 'overshirt', price: 1500000, currency: 'UZS', image_url: 'https://frgroup.uz/media/o3.jpg', source_url: 'https://frgroup.uz/ru/store/o3', brand: 'GANT', is_active: true },
  { id: 'b1', name: 'Charcoal Tailored Trousers', category: 'bottoms', garment_type: 'lower_body', fashion_slot: 'bottom', fashion_subtype: 'tailored-trouser', price: 950000, currency: 'UZS', image_url: 'https://frgroup.uz/media/b1.jpg', source_url: 'https://frgroup.uz/ru/store/b1', brand: 'Boss', is_active: true },
  { id: 'b2', name: 'Beige Cotton Chinos', category: 'bottoms', garment_type: 'lower_body', fashion_slot: 'bottom', fashion_subtype: 'chino', price: 780000, currency: 'UZS', image_url: 'https://frgroup.uz/media/b2.jpg', source_url: 'https://frgroup.uz/ru/store/b2', brand: 'GANT', is_active: true },
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
  assert.equal(res.body.visualReceipt.displayedJudged, res.body.outfits.length);
  assert.ok(res.body.visualReceipt.validVerdicts >= res.body.outfits.length);
  assert.equal(res.body.visualReceipt.visualApproval, 'full_three_verified');
  assert.ok(res.body.outfits.every((look) => look.selectionSource === 'ai' && look.fashionDiagnostics.visualVerdict === 'JUDGED'));
});
