import test from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import {
  SURGICAL_REFINER_VERSION,
  parseRepairDirective,
  rankSlotReplacementCandidates,
  refineLookSurgically,
  swapSingleSlot,
  computeCritiqueCompositeScore,
} from './surgical_refiner.mjs';
import { renderEditorialLookBoard, createPhotoByteCache } from './editorial_board_renderer.mjs';

const baseLook = {
  id: 'look_work_01',
  title: 'Tailored Autumn Ease',
  items: [
    { id: 'outer-camel-coat', macroCategory: 'outerwear', subtype: 'coat', color: 'camel', texture: 'wool', formality: 'smart', brand: 'Loro Piana', price: 420, currency: 'USD', imageUrl: 'https://static.zara.net/coat.jpg' },
    { id: 'top-ivory-knit', macroCategory: 'top', subtype: 'fine-knit', color: 'ivory', texture: 'merino', formality: 'smart', brand: 'COS', price: 110, currency: 'USD', imageUrl: 'https://static.zara.net/knit.jpg' },
    { id: 'bottom-charcoal-trouser', macroCategory: 'bottom', subtype: 'tailored-trouser', color: 'charcoal', texture: 'wool', formality: 'smart', brand: 'Massimo Dutti', price: 150, currency: 'USD', imageUrl: 'https://static.zara.net/trouser.jpg' },
    { id: 'shoes-chunky-trainer', macroCategory: 'shoes', subtype: 'sport-sneaker', color: 'white', texture: 'smooth', formality: 'sport', brand: 'RunFast', price: 140, currency: 'USD', imageUrl: 'https://static.zara.net/trainer.jpg' },
  ],
};

const catalogPool = [
  ...baseLook.items,
  { id: 'shoes-brown-suede-loafer', macroCategory: 'shoes', subtype: 'loafer', color: 'brown', texture: 'suede', formality: 'smart', brand: 'Velasca', price: 260, currency: 'USD', imageUrl: 'https://static.zara.net/loafer.jpg', qualityScore: 0.91 },
  { id: 'shoes-black-oxford', macroCategory: 'shoes', subtype: 'oxford', color: 'black', texture: 'leather', formality: 'formal', brand: 'TLB Mallorca', price: 290, currency: 'USD', imageUrl: 'https://static.zara.net/oxford.jpg', qualityScore: 0.85 },
  { id: 'shoes-overbudget-loafer', macroCategory: 'shoes', subtype: 'loafer', color: 'brown', texture: 'suede', formality: 'smart', brand: 'Luxury', price: 950, currency: 'USD', imageUrl: 'https://static.zara.net/lux-loafer.jpg', qualityScore: 0.98 },
  { id: 'shoes-quarantined-loafer', macroCategory: 'shoes', subtype: 'loafer', color: 'brown', texture: 'suede', formality: 'smart', brand: 'BrokenPhoto', price: 180, currency: 'USD', imageUrl: 'https://static.zara.net/bad.jpg', quarantined: true },
  { id: 'shoes-deck-used-loafer', macroCategory: 'shoes', subtype: 'loafer', color: 'brown', texture: 'suede', formality: 'smart', brand: 'AlreadyUsed', price: 210, currency: 'USD', imageUrl: 'https://static.zara.net/used.jpg', qualityScore: 0.95 },
  { id: 'shoes-budget-derby', macroCategory: 'shoes', subtype: 'derby', color: 'brown', texture: 'leather', formality: 'smart', brand: 'Jack Erwin', price: 95, currency: 'USD', imageUrl: 'https://static.zara.net/derby.jpg', qualityScore: 0.78 },
  { id: 'shoes-avoided-red-loafer', macroCategory: 'shoes', subtype: 'loafer', color: 'red', texture: 'suede', formality: 'smart', brand: 'Loud', price: 200, currency: 'USD', imageUrl: 'https://static.zara.net/red.jpg', qualityScore: 0.9 },
  { id: 'top-cream-dress-shirt', macroCategory: 'top', subtype: 'dress-shirt', color: 'cream', texture: 'poplin', formality: 'formal', brand: 'Charles Tyrwhitt', price: 105, currency: 'USD', imageUrl: 'https://static.zara.net/shirt.jpg', qualityScore: 0.88 },
];

test('parseRepairDirective extracts target color, subtype, texture, avoidance, and formality shift', () => {
  assert.match(SURGICAL_REFINER_VERSION, /surgical-refiner/);
  const parsed = parseRepairDirective(
    'Replace chunky white sneaker with dark brown suede loafer for a more refined office finish',
    'footwear_mismatch',
  );
  assert.ok(parsed.targetColors.includes('brown'));
  assert.ok(parsed.targetSubtypes.includes('loafer'));
  assert.ok(parsed.targetTextures.includes('suede'));
  assert.ok(parsed.avoidSubtypes.includes('sneaker'));
  assert.equal(parsed.formalityShift, 'more_formal');
  assert.equal(parsed.targetTemperature, 'warm');
});

test('rankSlotReplacementCandidates respects budget headroom, zero SKU reuse, quarantine, and personal exclusions', () => {
  const ranked = rankSlotReplacementCandidates({
    look: baseLook,
    targetSlot: 'shoes',
    directive: 'Replace chunky sneaker with dark brown suede loafer',
    defectCategory: 'footwear_mismatch',
    catalogPool,
    context: { occasion: 'Work', budget: { amount: 1000, currency: 'USD' } }, // locked sum = 680 -> headroom = 320
    personalStyleDna: { avoidedColors: ['red'], favoriteColors: ['brown', 'camel'] },
    excludedIds: new Set(['shoes-deck-used-loafer']),
    limit: 5,
  });

  const ids = ranked.map((r) => r.item.id);
  assert.equal(ids[0], 'shoes-brown-suede-loafer', 'directive-aligned suede brown loafer should rank #1');
  assert.ok(!ids.includes('shoes-overbudget-loafer'), 'must enforce 320 USD remaining budget headroom');
  assert.ok(!ids.includes('shoes-quarantined-loafer'), 'must skip quarantined SKUs');
  assert.ok(!ids.includes('shoes-deck-used-loafer'), 'must enforce zero SKU reuse across deck');
  assert.ok(!ids.includes('shoes-avoided-red-loafer'), 'must enforce avoidedColors');
  assert.ok(!ids.includes('shoes-chunky-trainer'), 'must not suggest the weak item being replaced');
});

test('refineLookSurgically locks 3 strong items, re-renders with photo cache, verifies hard gates & critic, and commits swap', async () => {
  const jpg = await sharp({ create: { width: 120, height: 150, channels: 3, background: '#7c6b58' } }).jpeg().toBuffer();
  const photoCache = createPhotoByteCache({ maxEntries: 32 });
  let fetchCount = 0;
  const fetchImpl = async () => { fetchCount++; return new Response(jpg, { status: 200 }); };

  // Pre-render initial look so 3 locked items are in the photo cache (simulating Wave 1 -> Surgical Swap!)
  await renderEditorialLookBoard(baseLook.items, { photoCache, fetchImpl, requireAllPhotos: true });
  assert.equal(fetchCount, 4);

  const initialCritique = {
    candidateId: baseLook.id,
    reject: true,
    editorialScore: 5.4,
    weakestSlot: 'shoes',
    defectCategory: 'footwear_mismatch',
    repairDirective: 'Replace chunky white sneaker with dark brown suede loafer',
    visualCohesion: 0.62,
    styleFit: 0.65,
    occasionFit: 0.55,
    formalityFit: 0.45,
    confidence: 0.92,
  };

  let criticSawBoard = false;
  const result = await refineLookSurgically({
    look: baseLook,
    critique: initialCritique,
    catalogPool,
    context: { occasion: 'Work', style: 'classic', budget: { amount: 1000, currency: 'USD' } },
    excludedItemIds: ['shoes-deck-used-loafer'],
    hardGateValidator: async (items) => ({ ok: items.length === 4 }),
    boardRenderer: (items, opts) => renderEditorialLookBoard(items, { ...opts, photoCache, fetchImpl }),
    visualCritic: async ([candidate]) => {
      assert.ok(candidate.editorialBoardJpeg?.jpegBuffer?.length > 1000);
      criticSawBoard = true;
      const shoe = candidate.products.find((p) => p.slot === 'shoes');
      assert.equal(shoe.id, 'shoes-brown-suede-loafer');
      return [{
        candidateId: candidate.id,
        reject: false,
        editorialScore: 8.6,
        weakestSlot: null,
        defectCategory: 'none',
        repairDirective: null,
        visualCohesion: 0.89,
        styleFit: 0.86,
        occasionFit: 0.88,
        formalityFit: 0.87,
        confidence: 0.94,
        referenceless: true,
      }];
    },
  });

  assert.equal(criticSawBoard, true);
  assert.equal(result.status, 'refined_and_approved');
  assert.equal(result.committed, true);
  assert.equal(result.approved, true);
  assert.equal(result.swappedSlot, 'shoes');
  assert.equal(result.selectedItemId, 'shoes-brown-suede-loafer');
  assert.equal(result.previousScore, 5.4);
  assert.equal(result.newScore, 8.6);
  assert.equal(fetchCount, 5, '3 locked items hit photoCache -> only 1 CDN fetch for the swapped shoe!');

  // Verify the 3 strong items stayed 100% locked
  const lockedIds = result.look.items.filter((i) => i.macroCategory !== 'shoes').map((i) => i.id);
  assert.deepEqual(lockedIds, ['outer-camel-coat', 'top-ivory-knit', 'bottom-charcoal-trouser']);
});

test('refineLookSurgically NEVER lowers the quality floor when improvements stay below threshold', async () => {
  const initialCritique = {
    candidateId: baseLook.id,
    reject: true,
    editorialScore: 5.0,
    weakestSlot: 'shoes',
    defectCategory: 'footwear_mismatch',
    repairDirective: 'Replace chunky sneaker with dark brown suede loafer',
    confidence: 0.9,
  };

  const result = await refineLookSurgically({
    look: baseLook,
    critique: initialCritique,
    catalogPool,
    context: { occasion: 'Work', budget: { amount: 1000, currency: 'USD' } },
    hardGateValidator: () => ({ ok: true }),
    maxRounds: 1,
    visualCritic: async ([candidate]) => [{
      candidateId: candidate.id,
      reject: false,
      editorialScore: 6.4, // Improved from 5.0 -> 6.4, STILL below 7.0 editorial threshold!
      weakestSlot: 'top',
      defectCategory: 'editorial_mediocrity',
      repairDirective: 'Try crisp cream dress shirt',
      confidence: 0.9,
      referenceless: true,
    }],
  });

  assert.equal(result.status, 'improved_below_threshold');
  assert.equal(result.committed, false, 'must NEVER commit a look below editorial threshold');
  assert.equal(result.approved, false);
  assert.equal(result.look.id, baseLook.id, 'original look preserved uncommitted');
  assert.ok(result.bestVerifiedImprovement);
  assert.equal(result.bestVerifiedImprovement.score10, 6.4);
  assert.equal(result.diagnostic, 'best_candidate_below_editorial_threshold');
});

test('refineLookSurgically supports 2-round rescue and enforces hard gates and locked anchors', async () => {
  // Locked anchor check
  const lockedRes = await refineLookSurgically({
    look: baseLook,
    critique: { candidateId: baseLook.id, reject: true, editorialScore: 5.2, weakestSlot: 'shoes', confidence: 0.9 },
    lockedSlots: ['shoes'],
    catalogPool,
  });
  assert.equal(lockedRes.status, 'weakest_slot_locked');
  assert.equal(lockedRes.committed, false);

  // Hard gate blocks first candidate, Round 1 improves to 6.5 (next weakest: top), Round 2 clears 8.3!
  const gateBlocked = 'shoes-brown-suede-loafer';
  let criticCalls = 0;
  const twoRound = await refineLookSurgically({
    look: baseLook,
    critique: {
      candidateId: baseLook.id,
      reject: true,
      editorialScore: 5.1,
      weakestSlot: 'shoes',
      defectCategory: 'footwear_mismatch',
      repairDirective: 'Replace chunky sneaker with dark brown suede loafer',
      confidence: 0.9,
    },
    catalogPool,
    context: { occasion: 'Work', budget: { amount: 1000, currency: 'USD' } },
    excludedItemIds: ['shoes-deck-used-loafer'],
    maxRounds: 2,
    topCandidatesPerRound: 1,
    hardGateValidator: (items) => ({ ok: !items.some((i) => i.id === gateBlocked), reason: 'weather_or_stock_gate' }),
    visualCritic: async ([candidate]) => {
      criticCalls++;
      const hasDressShirt = candidate.products.some((p) => p.id === 'top-cream-dress-shirt');
      if (!hasDressShirt) {
        return [{
          candidateId: candidate.id,
          reject: false,
          editorialScore: 6.6,
          weakestSlot: 'top',
          defectCategory: 'formality_mismatch',
          repairDirective: 'Swap top for cream poplin dress shirt',
          confidence: 0.9,
          referenceless: true,
        }];
      }
      return [{
        candidateId: candidate.id,
        reject: false,
        editorialScore: 8.3,
        weakestSlot: null,
        defectCategory: 'none',
        repairDirective: null,
        confidence: 0.95,
        referenceless: true,
      }];
    },
  });

  assert.equal(twoRound.status, 'refined_and_approved');
  assert.equal(twoRound.committed, true);
  assert.equal(twoRound.rounds, 2);
  assert.equal(criticCalls, 2, 'hard-gate-blocked candidate must never trigger visual critic');
  assert.equal(twoRound.attempts[0].hardGatePassed, false);
  assert.equal(twoRound.attempts[0].criticRan, false);
  assert.ok(twoRound.look.items.some((i) => i.id === 'top-cream-dress-shirt'));
});

test('swapSingleSlot powers 1-tap UI slot replacement with over_budget preference and honest visualVerified flag', async () => {
  const fastSwap = await swapSingleSlot({
    look: baseLook,
    targetSlot: 'shoes',
    reason: 'over_budget',
    catalogPool,
    context: { occasion: 'Work', budget: { amount: 1000, currency: 'USD' } },
    excludedItemIds: ['shoes-deck-used-loafer'],
    hardGateValidator: () => ({ ok: true }),
  });

  assert.equal(fastSwap.status, 'swapped');
  assert.equal(fastSwap.swapped, true);
  assert.equal(fastSwap.visualVerified, false, 'must honestly report visualVerified=false when no visualCritic ran');
  assert.equal(fastSwap.newItem.id, 'shoes-budget-derby', 'over_budget reason selects cheaper leather derby ($95 < $140)');
  assert.equal(fastSwap.look.totalPrice, 420 + 110 + 150 + 95);

  const verifiedSwap = await swapSingleSlot({
    look: baseLook,
    targetSlot: 'shoes',
    reason: 'wrong_shoes prefer brown suede loafer',
    catalogPool,
    context: { occasion: 'Work', budget: { amount: 1000, currency: 'USD' } },
    excludedItemIds: ['shoes-deck-used-loafer'],
    hardGateValidator: () => ({ ok: true }),
    visualCritic: async ([candidate]) => [{
      candidateId: candidate.id,
      reject: false,
      editorialScore: 8.4,
      confidence: 0.92,
      referenceless: true,
    }],
  });

  assert.equal(verifiedSwap.status, 'swapped');
  assert.equal(verifiedSwap.visualVerified, true);
  assert.equal(verifiedSwap.newItem.id, 'shoes-brown-suede-loafer');
  assert.equal(verifiedSwap.editorialScore, 8.4);

  const scoreInfo = computeCritiqueCompositeScore({ visualCohesion: 0.85, styleFit: 0.8, occasionFit: 0.8, formalityFit: 0.8, confidence: 0.9, reject: false, referenceless: true });
  assert.equal(scoreInfo.passedFloor, true);
});
