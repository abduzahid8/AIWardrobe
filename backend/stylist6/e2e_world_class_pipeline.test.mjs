import test from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import { signStylistTransport } from '../stylist-node/transport/signature.mjs';
import {
  handleStylist6CloudGenerate,
  projectWardrobeItem,
  STYLIST6_CLOUD_VERSION,
} from './cloud_handler.mjs';
import { recommendFashionDeck } from './fashion_deck.mjs';
import { createQualityFlywheel } from './catalog_quality_flywheel.mjs';
import { createStylistCache } from './distributed_cache.mjs';
import { normalizePersonalStyleDna, buildStyleDnaPromptBlock } from '../../src/domain/personalStyleDna.ts';

const SECRET = 'e2e-world-class-shared-secret-0123456789abcdef';
const JWT = 'e2e-valid-jwt-token';
const USER = 'user-e2e-stylist6';
const SB_IMG_BASE = 'https://fyqpifmrsftsfqibhwhy.supabase.co/storage/v1/object/shop-catalog';

function makeCatalogRow({
  id,
  slot,
  subtype,
  name,
  brand = 'Massimo Dutti',
  color = 'navy',
  price = 450000,
  currency = 'UZS',
  sourceUrl = `https://www.massimodutti.com/uz/product/${id}.html`,
  imageUrl = `${SB_IMG_BASE}/${id}.jpg`,
  extra = {},
}) {
  const rawCategory = slot === 'bottom' ? 'bottoms' : slot === 'shoes' ? 'shoes' : 'tops';
  const garmentType = slot === 'bottom' ? 'lower_body' : slot === 'shoes' ? 'shoes' : 'upper_body';
  return {
    id,
    name,
    brand,
    category: rawCategory,
    garment_type: garmentType,
    fashion_slot: slot,
    fashion_subtype: subtype,
    color,
    price,
    currency,
    image_url: imageUrl,
    source_url: sourceUrl,
    is_active: true,
    ...extra,
  };
}

const FULL_HYGIENIC_AND_DEFECTIVE_CATALOG = [
  // Direction 1: Signature Tailoring (navy blazer, white dress-shirt, charcoal tailored-trouser, dark brown oxford)
  makeCatalogRow({ id: 'top-shirt-white', slot: 'top', subtype: 'dress-shirt', name: 'White Poplin Dress Shirt', color: 'white', price: 380000 }),
  makeCatalogRow({ id: 'top-shirt-sky', slot: 'top', subtype: 'dress-shirt', name: 'Sky Blue Oxford Dress Shirt', color: 'blue', price: 390000 }),
  makeCatalogRow({ id: 'top-knit-cream', slot: 'top', subtype: 'fine-knit', name: 'Cream Merino Fine Knit Pullover', color: 'cream', price: 420000 }),
  makeCatalogRow({ id: 'top-polo-navy', slot: 'top', subtype: 'polo', name: 'Navy Knitted Silk-Cotton Polo', color: 'navy', price: 360000 }),
  makeCatalogRow({ id: 'top-shirt-ivory', slot: 'top', subtype: 'shirt', name: 'Ivory Linen-Cotton Tailored Shirt', color: 'ivory', price: 370000 }),
  makeCatalogRow({ id: 'top-knit-camel', slot: 'top', subtype: 'fine-knit', name: 'Camel Cashmere Crewneck Sweater', color: 'camel', price: 510000 }),

  // Outerwear across distinct archetypes
  makeCatalogRow({ id: 'outer-blazer-navy', slot: 'outerwear', subtype: 'blazer', name: 'Navy Structured Wool Blazer', color: 'navy', price: 890000 }),
  makeCatalogRow({ id: 'outer-coat-camel', slot: 'outerwear', subtype: 'coat', name: 'Camel Single-Breasted Wool Overcoat', color: 'camel', price: 1100000 }),
  makeCatalogRow({ id: 'outer-cardigan-olive', slot: 'outerwear', subtype: 'cardigan', name: 'Olive Merino Shawl-Collar Cardigan', color: 'olive', price: 650000 }),
  makeCatalogRow({ id: 'outer-overshirt-taupe', slot: 'outerwear', subtype: 'overshirt', name: 'Taupe Suede Utility Overshirt', color: 'beige', price: 780000 }),
  makeCatalogRow({ id: 'outer-blazer-charcoal', slot: 'outerwear', subtype: 'blazer', name: 'Charcoal Flannel Tailored Blazer', color: 'charcoal', price: 920000 }),
  makeCatalogRow({ id: 'outer-jacket-brown', slot: 'outerwear', subtype: 'jacket', name: 'Espresso Cropped Blouson Jacket', color: 'brown', price: 840000 }),

  // Bottoms across distinct archetypes
  makeCatalogRow({ id: 'bottom-trouser-charcoal', slot: 'bottom', subtype: 'tailored-trouser', name: 'Charcoal Pleated Wool Tailored Trousers', color: 'charcoal', price: 520000 }),
  makeCatalogRow({ id: 'bottom-trouser-stone', slot: 'bottom', subtype: 'tailored-trouser', name: 'Stone Gabardine Straight Trousers', color: 'beige', price: 480000 }),
  makeCatalogRow({ id: 'bottom-chino-sand', slot: 'bottom', subtype: 'chino', name: 'Sand Tapered Cotton Twill Chinos', color: 'beige', price: 410000 }),
  makeCatalogRow({ id: 'bottom-trouser-navy', slot: 'bottom', subtype: 'trouser', name: 'Navy Worsted Wool Dress Trousers', color: 'navy', price: 530000 }),
  makeCatalogRow({ id: 'bottom-chino-olive', slot: 'bottom', subtype: 'chino', name: 'Olive Relaxed Tailored Chinos', color: 'olive', price: 430000 }),
  makeCatalogRow({ id: 'bottom-trouser-brown', slot: 'bottom', subtype: 'tailored-trouser', name: 'Chocolate Flannel Tailored Trousers', color: 'brown', price: 540000 }),

  // Shoes (including a clashing sport sneaker and strong dress/loafer replacements)
  makeCatalogRow({ id: 'shoes-oxford-brown', slot: 'shoes', subtype: 'oxford', name: 'Dark Brown Calfskin Cap-Toe Oxfords', color: 'brown', price: 720000 }),
  makeCatalogRow({ id: 'shoes-loafer-cognac', slot: 'shoes', subtype: 'loafer', name: 'Cognac Suede Penny Loafers', color: 'brown', price: 680000 }),
  makeCatalogRow({ id: 'shoes-derby-black', slot: 'shoes', subtype: 'derby', name: 'Black Polished Leather Plain-Toe Derbies', color: 'black', price: 710000 }),
  makeCatalogRow({ id: 'shoes-chelsea-espresso', slot: 'shoes', subtype: 'chelsea', name: 'Espresso Calfskin Chelsea Boots', color: 'brown', price: 760000 }),
  makeCatalogRow({ id: 'shoes-monk-burgundy', slot: 'shoes', subtype: 'monk', name: 'Burgundy Leather Double Monk Straps', color: 'burgundy', price: 740000 }),
  makeCatalogRow({ id: 'shoes-loafer-chocolate', slot: 'shoes', subtype: 'loafer', name: 'Chocolate Hand-Welted Tassel Loafers', color: 'brown', price: 690000 }),

  // DEFECTIVE CATALOG ROWS that MUST be stripped by filterHygienicCatalog:
  makeCatalogRow({
    id: 'bad-armani-homepage',
    slot: 'top',
    subtype: 'dress-shirt',
    name: 'Armani Dead Homepage Shirt',
    brand: 'Armani',
    sourceUrl: 'https://www.armani.com/en-us/',
  }),
  makeCatalogRow({
    id: 'bad-lowres-ca',
    slot: 'top',
    subtype: 'shirt',
    name: 'Low Resolution Thumbnail Shirt',
    brand: 'C&A',
    imageUrl: `${SB_IMG_BASE}/bad-lowres.jpg?w=262&h=393`,
  }),
  makeCatalogRow({
    id: 'bad-swim-shorts',
    slot: 'bottom',
    subtype: 'shorts',
    name: 'Aurélien Beachside Swim Shorts',
    brand: 'Aurélien',
  }),
];

function createTestHarness(overrides = {}) {
  const claims = new Map();
  const flywheel = overrides.qualityFlywheel || createQualityFlywheel();
  const appearanceCache = overrides.appearanceCache || createStylistCache({ namespace: 'test:appearance' });
  const shopRows = overrides.shopRows || FULL_HYGIENIC_AND_DEFECTIVE_CATALOG;
  const wardrobeRows = overrides.wardrobeRows || [];
  const feedbackRows = overrides.feedbackRows || [];
  let providerAppearanceCalls = 0;

  const io = {
    qualityFlywheel: flywheel,
    appearanceCache,
    dnsLookupFn: async () => ({ address: '93.184.216.34', family: 4 }),
    async analyzeAppearanceProvider(req) {
      providerAppearanceCalls += 1;
      if (overrides.analyzeAppearanceProvider) {
        return overrides.analyzeAppearanceProvider(req);
      }
      return {
        skinUndertone: 'warm',
        skinDepth: 'medium',
        contrastLevel: 'high',
        colorSeason: 'deep_autumn',
        flatteringColors: ['camel', 'olive', 'navy', 'cream', 'espresso'],
        powerNeutrals: ['charcoal', 'cream', 'espresso'],
        avoidColors: ['neon_green'],
        metalTone: 'gold',
        lightingQuality: 'good',
        confidence: 0.91,
      };
    },
    async verifyUser(jwt) {
      if (jwt !== JWT) throw new Error('stylist6_unauthorized');
      return USER;
    },
    async checkRate() {},
    async claimRequest(_userId, requestId, fingerprint) {
      if (claims.has(requestId)) {
        const prev = claims.get(requestId);
        return {
          duplicate: true,
          conflict: prev.fingerprint !== fingerprint,
          prior: { status: prev.status, code: prev.code },
          durable: true,
        };
      }
      claims.set(requestId, { fingerprint, status: 0, code: '' });
      return { duplicate: false, conflict: false, prior: null, durable: true };
    },
    async debitQuota() {},
    async finishRequest(_userId, requestId, status, code) {
      const prev = claims.get(requestId);
      if (prev) {
        prev.status = status;
        prev.code = code;
      }
      return true;
    },
    async saveCompletedResult() {
      return true;
    },
    async fetchShopRows() {
      return shopRows.map((r) => ({ ...r }));
    },
    async fetchWardrobeRows() {
      return wardrobeRows.map((r) => ({ ...r }));
    },
    async fetchVisualProfiles() {
      return [];
    },
    async fetchFeedbackTaste() {
      return feedbackRows;
    },
    async fetchBudgetFxSnapshot() {
      return {
        id: 'cbu-2026-11-19',
        source: 'cbu_uz',
        asOf: new Date().toISOString(),
        uzsPerUnit: { UZS: 1, USD: 12800, EUR: 13600, GBP: 16100 },
      };
    },
    ...overrides.io,
  };

  return {
    io,
    flywheel,
    appearanceCache,
    getAppearanceCalls: () => providerAppearanceCalls,
  };
}

async function signedPost(payload, harness, envOverrides = {}, fetchImpl = undefined, timeoutMs = 72000) {
  const body = JSON.stringify(payload);
  const authorization = `Bearer ${JWT}`;
  const sig = await signStylistTransport({ secret: SECRET, body, authorization });
  return handleStylist6CloudGenerate({
    method: 'POST',
    headers: { authorization, ...sig },
    bodyText: body,
    env: {
      STYLIST6_NODE_SHARED_SECRET: SECRET,
      SUPABASE_URL: 'https://fyqpifmrsftsfqibhwhy.supabase.co',
      SUPABASE_ANON_KEY: 'anon-key',
      SUPABASE_SERVICE_ROLE_KEY: 'service-key',
      ...envOverrides,
    },
    io: harness.io,
    fetchImpl,
    timeoutMs,
  });
}

test('(a) PersonalStyleDna + appearance/body/color-theory propagation & dress-slot separation', async () => {
  // 1. Verify dress-mapped-to-top bug is fixed: dresses map to 'dress', never 'top', while dress shirts stay 'top'.
  const silkDress = projectWardrobeItem({
    id: 'w-dress-1',
    name: 'Black Silk Midi Dress',
    category: 'Dresses',
    type: 'dress',
    color: 'black',
    image_url: 'https://example.com/w/dress1.jpg',
  });
  const ruDress = projectWardrobeItem({
    id: 'w-dress-ru',
    name: 'Вечернее шелковое платье',
    category: 'Платье',
    color: 'burgundy',
    image_url: 'https://example.com/w/dress2.jpg',
  });
  const dressShirt = projectWardrobeItem({
    id: 'w-shirt-1',
    name: 'White Poplin Dress Shirt',
    category: 'Tops',
    type: 'shirt',
    color: 'white',
    image_url: 'https://example.com/w/shirt1.jpg',
  });
  assert.equal(silkDress?.macroCategory, 'dress');
  assert.equal(ruDress?.macroCategory, 'dress');
  assert.equal(dressShirt?.macroCategory, 'top');

  // 2. Verify PersonalStyleDna normalization & prompt block generation from #0A1931, gray, body, and appearance.
  const dna = normalizePersonalStyleDna({
    preferences: {
      favoriteColors: ['#0A1931', 'gray', 'camel'],
      avoidColors: ['pink'],
      dislikedPatterns: ['graphic'],
      fitPreference: 'tailored',
    },
    presentation: 'menswear',
    body: {
      gender: 'male',
      heightCm: 184,
      weightKg: 79,
      bodyType: 'inverted_triangle',
    },
    appearance: {
      consentGranted: true,
      skinUndertone: 'warm',
      skinDepth: 'medium',
      contrastLevel: 'high',
      colorSeason: 'deep_autumn',
      flatteringColors: ['camel', 'olive', 'navy'],
      metalTone: 'gold',
      provenance: 'opt_in_photo_vlm',
    },
  });
  assert.ok(dna.preferences.favoriteColors.includes('navy'), '#0A1931 must normalize to navy');
  assert.ok(dna.preferences.favoriteColors.includes('grey'), 'gray must normalize to grey');
  assert.equal(dna.appearance.skinUndertone, 'warm');
  assert.equal(dna.body.bodyType, 'inverted_triangle');
  const promptBlock = buildStyleDnaPromptBlock(dna);
  assert.ok(promptBlock.includes('warm'), 'prompt block includes undertone');
  assert.ok(promptBlock.includes('inverted_triangle'), 'prompt block includes body architecture');
});

test('(b) filterHygienicCatalog + per-direction retrieval + Stylist Notes end-to-end cloud generation', async () => {
  const harness = createTestHarness();
  const res = await signedPost(
    {
      request: {
        style: 'classic',
        occasion: 'Work',
        brief: 'Executive boardroom presentation',
        sourceMode: 'catalog',
        weather: { temp: 14, condition: 'clear' },
        presentation: 'menswear',
        bodyProfile: {
          gender: 'male',
          heightCm: 182,
          weightKg: 78,
          bodyType: 'athletic',
        },
        appearanceProfile: {
          consentGranted: true,
          skinUndertone: 'warm',
          contrastLevel: 'high',
          colorSeason: 'deep_autumn',
          flatteringColors: ['camel', 'navy', 'cream', 'olive'],
        },
        preferences: {
          favoriteColors: ['#0A1931', 'camel', 'cream'],
          avoidColors: ['pink'],
          dislikedPatterns: ['graphic'],
        },
        generationRequestId: 'e2e-hygiene-and-notes-1',
      },
    },
    harness,
  );

  assert.equal(res.status, 200);
  assert.equal(res.body.success, true);
  assert.equal(res.body.generationVersion, STYLIST6_CLOUD_VERSION);
  assert.equal(res.body.outfits.length, 3);

  const returnedIds = new Set(res.body.outfits.flatMap((o) => o.items.map((it) => it.id)));
  assert.equal(returnedIds.has('bad-armani-homepage'), false, 'dead homepage PDP URL must be filtered out');
  assert.equal(returnedIds.has('bad-lowres-ca'), false, 'low-res <400px image must be filtered out');
  assert.equal(returnedIds.has('bad-swim-shorts'), false, 'swim shorts must be excluded from Work');

  for (const outfit of res.body.outfits) {
    assert.equal(outfit.items.length, 4, 'every catalog look has 4 real purchasable items');
    assert.ok(typeof outfit.stylistNote === 'string' && outfit.stylistNote.length >= 20, 'stylistNote populated');
    assert.ok(typeof outfit.whyItFlattersYou === 'string' && outfit.whyItFlattersYou.length >= 15, 'whyItFlattersYou populated');
    assert.ok(Array.isArray(outfit.stylingTips) && outfit.stylingTips.length >= 1, 'stylingTips populated');
    assert.ok(typeof outfit.colorStory === 'string' && outfit.colorStory.length >= 10, 'colorStory populated');
    assert.ok(typeof outfit.silhouetteFormula === 'string' && outfit.silhouetteFormula.length >= 10, 'silhouetteFormula populated');
    assert.ok(outfit.fashionDiagnostics?.score > 0);
  }
});

test('(c) visual critic rejection with weakestSlot: "shoes" is surgically rescued into a 3-of-3 verified deck', async () => {
  // Build a focused catalog where 3 directions exist, and one candidate look initially has clashing shoes
  // that the visual critic rejects with `weakestSlot: 'shoes'` + `repairDirective`.
  const badShoes = makeCatalogRow({
    id: 'shoes-clashing-trainer',
    slot: 'shoes',
    subtype: 'derby',
    name: 'Chunky Contrast Sole Derby',
    color: 'white',
    price: 500000,
  });
  const poolItems = [
    ...FULL_HYGIENIC_AND_DEFECTIVE_CATALOG.filter((r) => !r.id.startsWith('bad-')),
    badShoes,
  ];
  const { adaptCatalog, devLookFeasibility } = await import('./catalog_adapter.mjs');
  const { items: adaptedCatalog } = adaptCatalog(poolItems);

  let criticCalls = 0;
  const mockVisualCritic = async (candidates) => {
    criticCalls += 1;
    return candidates.map((cand, idx) => {
      const shoeItem = (cand.products || cand.items || []).find((p) => (p.slot || p.macroCategory) === 'shoes');
      const hasClashingShoe = shoeItem && String(shoeItem.id) === 'shoes-clashing-trainer';
      if (hasClashingShoe || (criticCalls === 1 && idx >= 2)) {
        // First-wave look #3 (or any look with clashing shoes) fails ONLY because of its footwear slot!
        return {
          candidateId: cand.id,
          referenceless: true,
          visualCohesion: 0.58,
          styleFit: 0.62,
          occasionFit: 0.78,
          formalityFit: 0.55,
          confidence: 0.92,
          reject: true,
          weakestSlot: 'shoes',
          defectCategory: 'footwear_clash',
          repairDirective: 'Replace shoes with dark brown leather loafer or oxford to ground the tailoring',
          editorialScore: 5.8,
          slotColors: { top: 'white', outerwear: 'navy', bottom: 'grey', shoes: 'white' },
          slotObservations: {
            top: 'Crisp shirt collar',
            outerwear: 'Structured wool blazer',
            bottom: 'Clean tailored trouser line',
            shoes: 'Footwear sole clashes with tailored trouser hem',
          },
          relationships: [
            'Blazer and shirt frame the upper torso cleanly',
            'Trouser drape is sharp',
            'Shoe sole clashes with formal trouser hem',
          ],
          evidence: [
            'Blazer and shirt frame the upper torso cleanly',
            'Trouser drape is sharp',
            'Footwear sole clashes with tailored trouser hem',
          ],
        };
      }
      // Approved look (including surgical replacement on criticCalls >= 2!)
      return {
        candidateId: cand.id,
        referenceless: true,
        visualCohesion: 0.88,
        styleFit: 0.89,
        occasionFit: 0.91,
        formalityFit: 0.87,
        confidence: 0.93,
        reject: false,
        weakestSlot: null,
        defectCategory: null,
        repairDirective: null,
        editorialScore: 8.8,
        slotColors: { top: 'white', outerwear: 'navy', bottom: 'grey', shoes: 'brown' },
        slotObservations: {
          top: 'Crisp tailored shirt',
          outerwear: 'Refined outer layer',
          bottom: 'Tailored trouser drape',
          shoes: 'Polished brown leather dress shoe grounds the silhouette',
        },
        relationships: [
          'Upper layers harmonize in weight and proportion',
          'Trouser break meets the brown leather shoe cleanly',
          'Cohesive executive palette',
        ],
        evidence: [
          'Upper layers harmonize in weight and proportion',
          'Trouser break meets the brown leather shoe cleanly',
          'Polished brown leather dress shoe grounds the silhouette',
        ],
      };
    });
  };

  const flywheel = createQualityFlywheel();
  const deck = await recommendFashionDeck({
    request: {
      style: 'classic',
      occasion: 'Work',
      brief: 'Polished executive office looks',
      sourceMode: 'catalog',
      optionalLayer: false,
      weather: { temp: 15 },
      preferences: { favoriteColors: ['navy', 'camel'] },
    },
    catalog: adaptedCatalog,
    hardGate: async (items) => devLookFeasibility(items),
    priceToBudgetCurrency: (item) => item.price,
    visualDeckReview: true,
    visualCandidateLimit: 6,
    selectDirectorFinalists: async ({ options }) => ({
      status: 'selected',
      tokens: options.slice(0, 6).map((o) => o.token),
      providerCalls: 1,
    }),
    selectCatalogCompositions: async ({ pool }) => {
      // Compose 3 orthogonal, 100% product-disjoint 4-SKU looks where Look 3 initially has the clashing shoe!
      const tokenFor = (id) => pool.find((p) => String(p.item?.id) === id)?.token;
      const looks = [
        {
          title: 'Signature Navy Tailoring',
          top: tokenFor('top-shirt-white'),
          outerwear: tokenFor('outer-blazer-navy'),
          bottom: tokenFor('bottom-trouser-charcoal'),
          shoes: tokenFor('shoes-oxford-brown'),
        },
        {
          title: 'Warm Camel Overcoat Editorial',
          top: tokenFor('top-knit-cream'),
          outerwear: tokenFor('outer-coat-camel'),
          bottom: tokenFor('bottom-trouser-stone'),
          shoes: tokenFor('shoes-loafer-cognac'),
        },
        {
          title: 'Olive Knit Layering (Clashing Footwear)',
          top: tokenFor('top-shirt-sky'),
          outerwear: tokenFor('outer-cardigan-olive'),
          bottom: tokenFor('bottom-chino-sand'),
          shoes: tokenFor('shoes-clashing-trainer'),
        },
      ].filter((l) => l.top && l.outerwear && l.bottom && l.shoes);
      return {
        status: 'composed',
        looks,
        providerCalls: 1,
        providerFamily: 'nemotron',
      };
    },
    visualCritic: mockVisualCritic,
    canJudgeOutfit: async () => true,
    qualityFlywheel: flywheel,
    personalStyleDna: normalizePersonalStyleDna({
      preferences: { favoriteColors: ['navy', 'camel'] },
      presentation: 'menswear',
      body: { heightCm: 183, bodyType: 'athletic' },
    }),
  });

  assert.equal(deck.status, 'complete', 'Surgical 1-slot refiner rescues the rejected 3rd look into a complete 3-look deck');
  assert.equal(deck.outfits.length, 3);
  assert.ok(criticCalls >= 2, 'Visual critic was re-invoked by surgical refiner to verify the 1-slot shoe swap');
  assert.ok(deck.diagnostics?.modelFashionSelection?.surgicalRefinement?.approved >= 1, 'Surgical refinement receipt recorded');
  for (const look of deck.outfits) {
    assert.equal(look.items.length, 4);
    assert.ok(Array.isArray(look.visualEvidence) && look.visualEvidence.length > 0, 'Every displayed look is visually verified');
    assert.ok(typeof look.stylistNote === 'string' && look.stylistNote.length > 10);
    assert.ok(typeof look.whyItFlattersYou === 'string' && look.whyItFlattersYou.length > 10);
    assert.ok(Array.isArray(look.stylingTips) && look.stylingTips.length > 0);
  }
});

test('(d) action: "swap_slot" executes a 1-tap surgical swap locking the other 3 items and recalculating notes & offers', async () => {
  const harness = createTestHarness();
  const originalOutfit = {
    id: 'outfit_1',
    items: [
      {
        id: 'top-shirt-white',
        macroCategory: 'top',
        type: 'dress-shirt',
        name: 'White Poplin Dress Shirt',
        brand: 'Massimo Dutti',
        color: 'white',
        price: 380000,
        currency: 'UZS',
        isShopItem: true,
        imageUrl: `${SB_IMG_BASE}/top-shirt-white.jpg`,
        shopUrl: 'https://www.massimodutti.com/uz/product/top-shirt-white.html',
      },
      {
        id: 'outer-blazer-navy',
        macroCategory: 'outerwear',
        type: 'blazer',
        name: 'Navy Structured Wool Blazer',
        brand: 'Massimo Dutti',
        color: 'navy',
        price: 890000,
        currency: 'UZS',
        isShopItem: true,
        imageUrl: `${SB_IMG_BASE}/outer-blazer-navy.jpg`,
        shopUrl: 'https://www.massimodutti.com/uz/product/outer-blazer-navy.html',
      },
      {
        id: 'bottom-trouser-charcoal',
        macroCategory: 'bottom',
        type: 'tailored-trouser',
        name: 'Charcoal Pleated Wool Tailored Trousers',
        brand: 'Massimo Dutti',
        color: 'charcoal',
        price: 520000,
        currency: 'UZS',
        isShopItem: true,
        imageUrl: `${SB_IMG_BASE}/bottom-trouser-charcoal.jpg`,
        shopUrl: 'https://www.massimodutti.com/uz/product/bottom-trouser-charcoal.html',
      },
      {
        id: 'shoes-derby-black',
        macroCategory: 'shoes',
        type: 'derby',
        name: 'Black Polished Leather Plain-Toe Derbies',
        brand: 'Massimo Dutti',
        color: 'black',
        price: 710000,
        currency: 'UZS',
        isShopItem: true,
        imageUrl: `${SB_IMG_BASE}/shoes-derby-black.jpg`,
        shopUrl: 'https://www.massimodutti.com/uz/product/shoes-derby-black.html',
      },
    ],
  };

  const res = await signedPost(
    {
      action: 'swap_slot',
      outfit: originalOutfit,
      targetSlot: 'shoes',
      reason: 'Prefer brown suede loafer instead of black derby',
      context: {
        style: 'classic',
        occasion: 'Work',
        sourceMode: 'catalog',
        budget: { amount: 3000000, currency: 'UZS' },
      },
      excludedIds: ['shoes-derby-black'],
    },
    harness,
  );

  assert.equal(res.status, 200);
  assert.equal(res.body.success, true);
  assert.equal(res.body.action, 'swap_slot');
  assert.equal(res.body.swappedSlot, 'shoes');
  assert.equal(res.body.previousItemId, 'shoes-derby-black');
  assert.ok(res.body.replacementItem && res.body.replacementItem.id !== 'shoes-derby-black');

  const swappedItems = res.body.outfit.items;
  assert.equal(swappedItems.length, 4);
  assert.equal(swappedItems.find((i) => i.macroCategory === 'top')?.id, 'top-shirt-white', 'locked top unchanged');
  assert.equal(swappedItems.find((i) => i.macroCategory === 'outerwear')?.id, 'outer-blazer-navy', 'locked outerwear unchanged');
  assert.equal(swappedItems.find((i) => i.macroCategory === 'bottom')?.id, 'bottom-trouser-charcoal', 'locked bottom unchanged');
  assert.notEqual(swappedItems.find((i) => i.macroCategory === 'shoes')?.id, 'shoes-derby-black', 'shoes slot swapped');
  assert.ok(typeof res.body.outfit.stylistNote === 'string' && res.body.outfit.stylistNote.length > 10);
});

test('(e) action: "analyze_appearance" enforces fail-closed consent gate and SHA-256 content-addressed caching', async () => {
  const harness = createTestHarness();
  const sampleJpeg = await sharp({
    create: { width: 64, height: 64, channels: 3, background: { r: 210, g: 170, b: 140 } },
  }).jpeg().toBuffer();

  const stubFetch = async () => new Response(sampleJpeg, {
    status: 200,
    headers: { 'Content-Type': 'image/jpeg' },
  });

  // 1. Without consent -> fail-closed consent_required, 0 provider calls
  const denied = await signedPost(
    {
      action: 'analyze_appearance',
      photoUrl: 'https://example.com/user-selfie.jpg',
      consentGranted: false,
    },
    harness,
    {},
    stubFetch,
  );
  assert.equal(denied.status, 200);
  assert.equal(denied.body.status, 'consent_required');
  assert.equal(denied.body.appearance, null);
  assert.equal(harness.getAppearanceCalls(), 0);

  // 2. With consent -> analyzes photo and caches by SHA-256
  const analyzed1 = await signedPost(
    {
      action: 'analyze_appearance',
      photoUrl: 'https://example.com/user-selfie.jpg',
      consentGranted: true,
    },
    harness,
    {},
    stubFetch,
  );
  assert.equal(analyzed1.status, 200);
  assert.equal(analyzed1.body.status, 'analyzed');
  assert.equal(analyzed1.body.cached, false);
  assert.equal(analyzed1.body.appearance.skinUndertone, 'warm');
  assert.equal(analyzed1.body.appearance.colorSeason, 'deep_autumn');
  assert.equal(harness.getAppearanceCalls(), 1);

  // 3. Second call with identical photo bytes hits SHA-256 cache with 0 extra provider calls
  const analyzed2 = await signedPost(
    {
      action: 'analyze_appearance',
      photoUrl: 'https://example.com/user-selfie-copy.jpg',
      consentGranted: true,
    },
    harness,
    {},
    stubFetch,
  );
  assert.equal(analyzed2.status, 200);
  assert.equal(analyzed2.body.status, 'analyzed');
  assert.equal(analyzed2.body.cached, true);
  assert.equal(harness.getAppearanceCalls(), 1, 'SHA-256 cache hit must skip provider invocation');
});

test('(f) quality flywheel automatically quarantines a SKU after 3 photo defects and excludes it from generation', async () => {
  const flywheel = createQualityFlywheel();
  // Record 3 photo_defect outcomes blaming 'shoes-oxford-brown'
  for (let i = 0; i < 3; i++) {
    await flywheel.recordOutcome({
      items: [{ id: 'shoes-oxford-brown', slot: 'shoes' }],
      approved: false,
      weakestSlot: 'shoes',
      defectCategory: 'photo_defect',
      editorialScore: 3.5,
    });
  }
  assert.equal(await flywheel.isQuarantined('shoes-oxford-brown'), true, 'SKU must be quarantined after 3 photo defects');

  const harness = createTestHarness({ qualityFlywheel: flywheel });
  const res = await signedPost(
    {
      request: {
        style: 'classic',
        occasion: 'Work',
        sourceMode: 'catalog',
        generationRequestId: 'e2e-quarantine-check-1',
      },
    },
    harness,
  );
  assert.equal(res.status, 200);
  const allReturnedIds = new Set(res.body.outfits.flatMap((o) => o.items.map((it) => it.id)));
  assert.equal(allReturnedIds.has('shoes-oxford-brown'), false, 'quarantined SKU must be excluded by filterHygienicCatalog');
});

test('(g) fastVisualMode E2E: OpenRouter Gemini 2.5 Flash audits 1024x1024 boards, rejects cropped shirt-cuff defect, rescues via 1-slot swap to 3/3, and preserves verified triple', async () => {
  const croppedShirtRow = makeCatalogRow({
    id: 'top-cuff-zoom-defect',
    slot: 'top',
    subtype: 'dress-shirt',
    name: 'White Poplin Cuff-Zoom Defect Shirt',
    color: 'white',
    price: 340000,
    imageUrl: `${SB_IMG_BASE}/top-cuff-zoom-defect.jpg`,
  });
  const e2eCatalog = [
    croppedShirtRow,
    ...FULL_HYGIENIC_AND_DEFECTIVE_CATALOG.filter((r) => !r.id.startsWith('bad-')),
  ];
  const harness = createTestHarness({ shopRows: e2eCatalog });

  // Deterministic colored JPEG generator per URL
  const jpegCache = new Map();
  async function getJpeg(url) {
    if (jpegCache.has(url)) return jpegCache.get(url);
    const seed = [...url].reduce((acc, ch) => (acc * 31 + ch.charCodeAt(0)) & 0xff, 17);
    const buf = await sharp({
      create: { width: 256, height: 320, channels: 3, background: { r: (seed * 3) % 220 + 20, g: (seed * 5) % 220 + 20, b: (seed * 7) % 220 + 20 } },
    }).jpeg({ quality: 82 }).toBuffer();
    jpegCache.set(url, buf);
    return buf;
  }

  let openrouterVisionCalls = 0;
  let compositeBoardsAudited = 0;
  let croppedDefectRejected = 0;
  let photoEditorCalls = 0;
  let activeVisionCalls = 0;
  let peakVisionConcurrency = 0;

  const stubFetch = async (url, options = {}) => {
    const parsed = new URL(String(url));
    if (parsed.hostname === 'openrouter.ai' || parsed.hostname === 'integrate.api.nvidia.com') {
      const body = JSON.parse(options.body);
      const messages = body.messages || [];
      const firstContent = messages[0]?.content;
      const promptText = Array.isArray(firstContent)
        ? firstContent.filter((p) => p.type === 'text').map((p) => p.text).join('\n')
        : String(firstContent || '');

      // Photo editorial portfolio selector call (1800x1000 sheet)
      if (promptText.includes('portfolio') && /"portfolio"\s*:\s*"P/.test(promptText)) {
        photoEditorCalls += 1;
        // Even if the model tries to pick a 2-look pair, photo_editor_final must preserve a verified distinct 3-look triple!
        const pairMatch = promptText.match(/"(P\d+)"[^}]*"size"\s*:\s*2/);
        const chosen = pairMatch ? pairMatch[1] : 'P1';
        return Response.json({
          choices: [{ finish_reason: 'stop', message: { content: JSON.stringify({ portfolio: chosen, reason: 'clean editorial review' }) } }],
          usage: { prompt_tokens: 600, completion_tokens: 40, total_tokens: 640 },
        });
      }

      // Per-candidate 1024x1024 Visual Critic call
      if (promptText.includes('candidateId must be exactly')) {
        if (parsed.hostname === 'openrouter.ai') openrouterVisionCalls += 1;
        const imageParts = Array.isArray(firstContent) ? firstContent.filter((p) => p.type === 'image_url') : [];
        if (promptText.includes('1024x1024 studio flat-lay board') && imageParts.length === 1) compositeBoardsAudited += 1;
        activeVisionCalls += 1;
        peakVisionConcurrency = Math.max(peakVisionConcurrency, activeVisionCalls);
        await new Promise((r) => setTimeout(r, 20));
        activeVisionCalls -= 1;

        const id = promptText.match(/candidateId must be exactly "([^"]+)"/)[1];
        const ctxLine = promptText.split('\n').find((l) => l.startsWith('{"style"'));
        const { imageSlots, merchantGarmentEvidence } = JSON.parse(ctxLine);
        const isCropped = Array.isArray(merchantGarmentEvidence)
          && merchantGarmentEvidence.some((g) => String(g?.title || '').includes('Cuff-Zoom Defect'));
        if (isCropped) {
          croppedDefectRejected += 1;
          return Response.json({
            choices: [{
              finish_reason: 'stop',
              message: {
                content: JSON.stringify({
                  candidateId: id,
                  slotObservations: Object.fromEntries(imageSlots.map((s) => [s, s === 'top' ? 'cropped shirt cuff close-up only' : `full ${s} garment`])),
                  visualCohesion: 0.48,
                  styleFit: 0.62,
                  occasionFit: 0.72,
                  formalityFit: 0.68,
                  confidence: 0.94,
                  reject: true,
                  weakestSlot: 'top',
                  defectCategory: 'photo_quality_defect',
                  repairDirective: 'Replace cropped shirt cuff close-up with a clean full-garment white or sky blue dress shirt',
                  editorialScore: 4.5,
                  relationships: [
                    { kind: 'outfit', slots: ['outerwear', 'bottom'], observation: 'blazer and trouser line is clean' },
                    { kind: 'outfit', slots: ['bottom', 'shoes'], observation: 'trouser hem meets dress shoe cleanly' },
                    { kind: 'outfit', slots: ['top', 'outerwear'], observation: 'top photo is only a zoomed shirt cuff, not a full garment' },
                  ],
                }),
              },
            }],
            usage: { prompt_tokens: 720, completion_tokens: 80, total_tokens: 800 },
          });
        }
        return Response.json({
          choices: [{
            finish_reason: 'stop',
            message: {
              content: JSON.stringify({
                candidateId: id,
                slotObservations: Object.fromEntries(imageSlots.map((s) => [s, `full studio ${s} garment`])),
                visualCohesion: 0.91,
                styleFit: 0.9,
                occasionFit: 0.92,
                formalityFit: 0.9,
                confidence: 0.95,
                reject: false,
                weakestSlot: null,
                defectCategory: null,
                repairDirective: null,
                editorialScore: 9.0,
                relationships: [
                  { kind: 'outfit', slots: ['top', 'outerwear'], observation: 'shirt collar frames blazer lapel cleanly' },
                  { kind: 'outfit', slots: ['outerwear', 'bottom'], observation: 'jacket drape balances trouser silhouette' },
                  { kind: 'outfit', slots: ['bottom', 'shoes'], observation: 'tailored hem meets leather footwear cleanly' },
                ],
              }),
            },
          }],
          usage: { prompt_tokens: 720, completion_tokens: 70, total_tokens: 790 },
        });
      }

      // Text-based director/composer calls: return unavailable so deterministic shortlist feeds our visual pipeline cleanly
      return new Response('unavailable', { status: 503 });
    }

    // Merchant/Supabase product photo fetch
    const buf = await getJpeg(String(url));
    return new Response(buf, { status: 200, headers: { 'Content-Type': 'image/jpeg' } });
  };

  const res = await signedPost(
    {
      request: {
        style: 'classic',
        occasion: 'Work',
        brief: 'Executive tailoring',
        sourceMode: 'catalog',
        enableVisual: true,
        weather: { temp: 15, condition: 'clear' },
        budget: { amount: 4500000, currency: 'UZS' },
        generationRequestId: 'e2e-fast-vision-board-rescue-1',
      },
    },
    harness,
    {
      OPENROUTER_API_KEY: 'sk-or-v1-e2e-test-key-0123456789abcdef',
      NVIDIA_API_KEY: 'nvapi-e2e-test-key-0123456789abcdef',
      STYLIST6_FAST_VISUAL_MODE: '1',
      STYLIST6_CREATIVE_DIRECTOR_ENABLED: '0',
    },
    stubFetch,
    72000,
  );

  assert.equal(res.status, 200);
  assert.equal(res.body.success, true);
  assert.equal(res.body.outfits.length, 3, 'verified 3-look triple preserved end-to-end');
  assert.equal(res.body.visualReceipt.fastVisualMode, true);
  assert.equal(res.body.visualReceipt.model, 'google/gemini-2.5-flash');
  assert.ok(openrouterVisionCalls >= 3, `OpenRouter Gemini 2.5 Flash audited look boards (calls=${openrouterVisionCalls})`);
  assert.ok(compositeBoardsAudited >= 3, `1024x1024 composite boards sent to critic (count=${compositeBoardsAudited})`);
  assert.ok(peakVisionConcurrency >= 2, `parallel visual critique concurrency >= 2 (peak=${peakVisionConcurrency})`);
  for (const look of res.body.outfits) {
    assert.equal(look.items.length, 4);
    assert.ok(look.items.every((it) => it.id !== 'top-cuff-zoom-defect'),
      'cropped shirt-cuff defect must never survive into the final 3-look deck');
    assert.equal(look.fashionDiagnostics.visualVerdict, 'JUDGED');
  }
});
