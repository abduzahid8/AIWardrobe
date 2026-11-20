import test from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import {
  SURGICAL_REFINER_VERSION,
  parseRepairDirective,
  rankSlotReplacementCandidates,
  refineLookSurgically,
  refineLooksSurgicallyInParallel,
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

test('budgeted refineLookSurgically and swapSingleSlot pass 2-arg (item, currency) to strict FX converter', async () => {
  const seenCurrencies = [];
  const strictFxConverter = (item, currency) => {
    // Mirrors src/domain/budgetFx.ts referenceAmount calling to.trim()
    const target = currency.trim().toUpperCase();
    seenCurrencies.push(target);
    return item.currency === target ? item.price : item.price * 1.1;
  };

  const swapRes = await swapSingleSlot({
    look: baseLook,
    targetSlot: 'shoes',
    reason: 'wrong_shoes',
    catalogPool,
    context: { occasion: 'Work', budget: { amount: 900, currency: 'USD' } },
    excludedItemIds: ['shoes-deck-used-loafer'],
    priceToBudgetCurrency: strictFxConverter,
    hardGateValidator: () => ({ ok: true }),
  });
  assert.equal(swapRes.status, 'swapped');
  assert.ok(seenCurrencies.length > 0 && seenCurrencies.every((c) => c === 'USD'));

  const refineRes = await refineLookSurgically({
    look: baseLook,
    critique: {
      candidateId: baseLook.id,
      reject: true,
      editorialScore: 5.5,
      weakestSlot: 'shoes',
      defectCategory: 'formality_mismatch',
      repairDirective: 'Swap shoes for brown suede loafer',
      confidence: 0.92,
      referenceless: true,
    },
    catalogPool,
    context: { occasion: 'Work', budget: { amount: 900, currency: 'USD' } },
    excludedItemIds: ['shoes-deck-used-loafer'],
    priceToBudgetCurrency: strictFxConverter,
    hardGateValidator: () => ({ ok: true }),
    visualCritic: async ([candidate]) => [{
      candidateId: candidate.id,
      reject: false,
      editorialScore: 8.6,
      weakestSlot: null,
      defectCategory: 'none',
      confidence: 0.94,
      referenceless: true,
    }],
  });
  assert.equal(refineRes.status, 'refined_and_approved');
  assert.equal(refineRes.committed, true);
});

test('cropped-photo defect rescue swaps ONLY the cropped shirt cuff slot with a clean full-garment white dress shirt (no anti-clone penalty & skips duplicate broken photo URL)', async () => {
  const croppedTopLook = {
    id: 'look_cropped_cuff',
    title: 'Classic Tailoring With Cropped Shirt Photo',
    items: [
      { id: 'outer-navy-blazer', macroCategory: 'outerwear', subtype: 'blazer', color: 'navy', texture: 'wool', formality: 'smart', price: 240, currency: 'USD', imageUrl: 'https://static.zara.net/blazer.jpg' },
      { id: 'top-cropped-cuff-shirt', macroCategory: 'top', subtype: 'dress-shirt', color: 'white', texture: 'poplin', formality: 'formal', price: 85, currency: 'USD', imageUrl: 'https://static.zara.net/cropped-cuff.jpg' },
      { id: 'bottom-grey-trouser', macroCategory: 'bottom', subtype: 'tailored-trouser', color: 'grey', texture: 'wool', formality: 'smart', price: 140, currency: 'USD', imageUrl: 'https://static.zara.net/grey-trouser.jpg' },
      { id: 'shoes-black-oxford', macroCategory: 'shoes', subtype: 'oxford', color: 'black', texture: 'leather', formality: 'formal', price: 220, currency: 'USD', imageUrl: 'https://static.zara.net/black-oxford.jpg' },
    ],
  };
  const pool = [
    ...croppedTopLook.items,
    // Duplicate colorway pointing to the SAME broken cropped cuff image URL — MUST be skipped!
    { id: 'top-same-cropped-url', macroCategory: 'top', subtype: 'dress-shirt', color: 'white', texture: 'poplin', formality: 'formal', price: 85, currency: 'USD', imageUrl: 'https://static.zara.net/cropped-cuff.jpg', qualityScore: 0.99 },
    // Clean full-garment white poplin dress shirt — MUST rank #1 without near-clone penalty!
    { id: 'top-clean-white-dress-shirt', macroCategory: 'top', subtype: 'dress-shirt', color: 'white', texture: 'poplin', formality: 'formal', price: 95, currency: 'USD', imageUrl: 'https://static.zara.net/clean-white-shirt.jpg', qualityScore: 0.92, visualEvidenceVerified: true },
    { id: 'top-casual-green-polo', macroCategory: 'top', subtype: 'polo', color: 'green', texture: 'cotton', formality: 'casual', price: 70, currency: 'USD', imageUrl: 'https://static.zara.net/green-polo.jpg', qualityScore: 0.72 },
  ];

  const parsed = parseRepairDirective(
    'Replace cropped shirt cuff close-up with full-garment crisp white poplin dress shirt',
    'photo_quality_defect',
  );
  assert.equal(parsed.isPhotoDefect, true);

  const ranked = rankSlotReplacementCandidates({
    look: croppedTopLook,
    targetSlot: 'top',
    directive: 'Replace cropped shirt cuff close-up with full-garment crisp white poplin dress shirt',
    defectCategory: 'photo_quality_defect',
    catalogPool: pool,
    context: { occasion: 'Work' },
  });
  assert.equal(ranked[0].item.id, 'top-clean-white-dress-shirt');
  assert.ok(!ranked.some((r) => r.item.id === 'top-same-cropped-url'), 'must skip duplicate broken photo URL');

  const rescued = await refineLookSurgically({
    look: croppedTopLook,
    critique: {
      candidateId: croppedTopLook.id,
      reject: true,
      editorialScore: 4.8,
      weakestSlot: 'top',
      defectCategory: 'photo_quality_defect',
      repairDirective: 'Replace cropped shirt cuff close-up with full-garment crisp white poplin dress shirt',
      confidence: 0.95,
    },
    catalogPool: pool,
    context: { occasion: 'Work' },
    hardGateValidator: () => ({ ok: true }),
    visualCritic: async ([cand]) => [{
      candidateId: cand.id,
      reject: false,
      editorialScore: 8.9,
      weakestSlot: null,
      defectCategory: 'none',
      visualCohesion: 0.91,
      styleFit: 0.90,
      occasionFit: 0.92,
      formalityFit: 0.91,
      confidence: 0.95,
      referenceless: true,
    }],
  });

  assert.equal(rescued.status, 'refined_and_approved');
  assert.equal(rescued.committed, true);
  assert.equal(rescued.swappedSlot, 'top');
  assert.equal(rescued.selectedItemId, 'top-clean-white-dress-shirt');
  assert.deepEqual(
    rescued.look.items.filter((i) => i.macroCategory !== 'top').map((i) => i.id),
    ['outer-navy-blazer', 'bottom-grey-trouser', 'shoes-black-oxford'],
  );
});

test('refineLooksSurgicallyInParallel executes concurrent surgical rescues with disjoint replacement SKUs', async () => {
  const lookA = { ...baseLook, id: 'parallel_look_A' };
  const lookB = {
    id: 'parallel_look_B',
    title: 'Olive Cardigan Near-Miss',
    items: [
      { id: 'outer-olive-cardigan', macroCategory: 'outerwear', subtype: 'cardigan', color: 'olive', texture: 'merino', formality: 'smart', price: 180, currency: 'USD', imageUrl: 'https://static.zara.net/card.jpg' },
      { id: 'top-sky-shirt', macroCategory: 'top', subtype: 'shirt', color: 'blue', texture: 'cotton', formality: 'smart', price: 90, currency: 'USD', imageUrl: 'https://static.zara.net/sky.jpg' },
      { id: 'bottom-sand-chino', macroCategory: 'bottom', subtype: 'chino', color: 'beige', texture: 'cotton', formality: 'smart', price: 110, currency: 'USD', imageUrl: 'https://static.zara.net/chino.jpg' },
      { id: 'shoes-trainer-b', macroCategory: 'shoes', subtype: 'sport-sneaker', color: 'white', texture: 'smooth', formality: 'sport', price: 130, currency: 'USD', imageUrl: 'https://static.zara.net/trainer-b.jpg' },
    ],
  };

  let active = 0;
  let peak = 0;
  const results = await refineLooksSurgicallyInParallel({
    targets: [
      {
        look: lookA,
        critique: { candidateId: lookA.id, reject: true, editorialScore: 5.2, weakestSlot: 'shoes', defectCategory: 'footwear_mismatch', repairDirective: 'Swap sneaker for brown suede loafer', confidence: 0.9 },
      },
      {
        look: lookB,
        critique: { candidateId: lookB.id, reject: true, editorialScore: 5.5, weakestSlot: 'shoes', defectCategory: 'footwear_mismatch', repairDirective: 'Swap sneaker for leather derby or loafer', confidence: 0.9 },
      },
    ],
    catalogPool: [...catalogPool, ...lookB.items],
    context: { occasion: 'Work', budget: { amount: 1100, currency: 'USD' } },
    excludedItemIds: ['shoes-deck-used-loafer'],
    hardGateValidator: () => ({ ok: true }),
    topCandidatesPerRound: 2,
    parallelCandidates: true,
    visualCritic: async (batch) => {
      active++;
      peak = Math.max(peak, active);
      await new Promise((r) => setTimeout(r, 25));
      active--;
      return batch.map((cand) => ({
        candidateId: cand.id,
        reject: false,
        editorialScore: 8.6,
        weakestSlot: null,
        defectCategory: 'none',
        visualCohesion: 0.88,
        styleFit: 0.86,
        occasionFit: 0.89,
        formalityFit: 0.87,
        confidence: 0.94,
        referenceless: true,
      }));
    },
  });

  assert.equal(results.length, 2);
  assert.equal(peak, 2, 'both surgical rescues must run concurrently in parallel');
  assert.equal(results[0].committed, true);
  assert.equal(results[1].committed, true);
  assert.notEqual(
    results[0].selectedItemId,
    results[1].selectedItemId,
    'parallel rescues targeting the same slot must receive disjoint replacement SKUs',
  );
});

test('sku_collision surgical rescue replaces ONLY the colliding slot with a visually verified passing unused SKU', async () => {
  const res = await refineLookSurgically({
    look: { ...baseLook, id: 'sku_collision_look' },
    critique: {
      candidateId: 'sku_collision_look',
      reject: false,
      editorialScore: 7.8,
      weakestSlot: 'shoes',
      defectCategory: 'sku_collision',
      repairDirective: 'Replace duplicate shoes SKU with a harmonious unused catalog shoes so all 3 looks are 100% SKU-disjoint.',
      confidence: 0.92,
    },
    catalogPool,
    context: { occasion: 'Work', budget: { amount: 1100, currency: 'USD' } },
    excludedItemIds: ['shoes-deck-used-loafer'],
    hardGateValidator: () => ({ ok: true }),
    maxRounds: 1,
    topCandidatesPerRound: 2,
    parallelCandidates: true,
    visualCritic: async (batch) => batch.map((cand) => ({
      candidateId: cand.id,
      reject: false,
      editorialScore: 7.7, // Already passes quality floor cleanly despite 0.1 delta vs 7.8 baseline!
      weakestSlot: null,
      defectCategory: 'none',
      visualCohesion: 0.84,
      styleFit: 0.82,
      occasionFit: 0.85,
      formalityFit: 0.84,
      confidence: 0.92,
      referenceless: true,
    })),
  });
  assert.equal(res.committed, true, 'sku_collision swap passing the visual floor must commit');
  assert.equal(res.approved, true);
  assert.equal(res.swappedSlot, 'shoes');
  assert.notEqual(res.selectedItemId, 'shoes-chunky-sneaker');
  assert.notEqual(res.selectedItemId, 'shoes-deck-used-loafer');
});

test('refineLookSurgically uses Visual Swap Board (selectVisualSlotReplacement) when providerRouter + photoCache are supplied, promoting VLM visual pick before 1024x1024 verification', async () => {
  const jpg = await sharp({ create: { width: 120, height: 150, channels: 3, background: '#5E4B3C' } }).jpeg().toBuffer();
  const photoCache = createPhotoByteCache({ maxEntries: 32 });
  const fetchImpl = async () => new Response(jpg, { status: 200 });

  // Warm photoCache for the 3 locked pieces
  await renderEditorialLookBoard(baseLook.items.slice(0, 3), { photoCache, fetchImpl, requireAllPhotos: true });

  // Text ranker prefers shoes-brown-suede-loafer (#1), then shoes-budget-derby (#2) or shoes-black-oxford (#3).
  // Suppose the VLM sees the Swap Board photos and prefers R2 (shoes-budget-derby) visually!
  let visionCalls = 0;
  let capturedSwapBoardLen = 0;
  let criticCheckedId = null;

  const mockRouter = {
    isProviderConfigured: () => true,
    callVisionModel: async (req) => {
      visionCalls++;
      capturedSwapBoardLen = req.images?.[0]?.bytes?.length || 0;
      const meta = await sharp(req.images[0].bytes).metadata();
      assert.equal(meta.width, 1400, 'Visual Swap Board sent to VLM must be 1400x900');
      assert.equal(meta.height, 900);
      return {
        ok: true,
        provider: 'openrouter',
        model: 'google/gemini-2.5-flash',
        parsedJson: {
          selectedToken: 'R2',
          visualRationale: 'R2 burnished brown leather derby visually balances the camel coat and charcoal trouser drape cleanly.',
          confidence: 0.93,
        },
      };
    },
  };

  const result = await refineLookSurgically({
    look: baseLook,
    critique: {
      candidateId: baseLook.id,
      reject: true,
      editorialScore: 5.3,
      weakestSlot: 'shoes',
      defectCategory: 'footwear_mismatch',
      repairDirective: 'Replace chunky white sneaker with sleek dark brown leather or suede dress shoe',
      confidence: 0.92,
    },
    catalogPool,
    context: { occasion: 'Work', style: 'classic', budget: { amount: 1000, currency: 'USD' } },
    excludedItemIds: ['shoes-deck-used-loafer'],
    hardGateValidator: () => ({ ok: true }),
    providerRouter: mockRouter,
    photoCache,
    fetchImpl,
    topCandidatesPerRound: 1,
    boardRenderer: (items, opts) => renderEditorialLookBoard(items, { ...opts, photoCache, fetchImpl }),
    visualCritic: async ([cand]) => {
      assert.ok(cand.editorialBoardJpeg?.jpegBuffer?.length > 1000, 'must still verify resulting 1024x1024 board');
      const shoe = cand.products.find((p) => p.slot === 'shoes');
      criticCheckedId = shoe.id;
      return [{
        candidateId: cand.id,
        reject: false,
        editorialScore: 8.9,
        weakestSlot: null,
        defectCategory: 'none',
        visualCohesion: 0.91,
        styleFit: 0.90,
        occasionFit: 0.91,
        formalityFit: 0.90,
        confidence: 0.95,
        referenceless: true,
      }];
    },
  });

  assert.equal(visionCalls, 1, 'Visual Swap Board VLM selector must execute once');
  assert.ok(capturedSwapBoardLen > 1000);
  assert.equal(result.status, 'refined_and_approved');
  assert.equal(result.committed, true);
  assert.equal(result.visualSwapUsed, true);
  assert.equal(result.visuallySelected, true);
  assert.equal(result.visualSwapToken, 'R2');
  assert.match(result.visualSwapRationale, /burnished brown/i);
  assert.equal(result.selectedItemId, criticCheckedId);
  assert.notEqual(result.selectedItemId, 'shoes-brown-suede-loafer', 'VLM picked R2 over text-rank #1');
});

test('swapSingleSlot integrates visualReplacementSelector / providerRouter + photoCache and degrades cleanly on fallback', async () => {
  const jpg = await sharp({ create: { width: 120, height: 150, channels: 3, background: '#3A4A5A' } }).jpeg().toBuffer();
  const photoCache = createPhotoByteCache({ maxEntries: 32 });
  const fetchImpl = async () => new Response(jpg, { status: 200 });

  // 1. Injected visualReplacementSelector chooses shoes-black-oxford over text #1
  let selectorSawCandidates = [];
  const visualSwap = await swapSingleSlot({
    look: baseLook,
    targetSlot: 'shoes',
    reason: 'wrong_shoes more polished',
    catalogPool,
    context: { occasion: 'Work', budget: { amount: 1000, currency: 'USD' } },
    excludedItemIds: ['shoes-deck-used-loafer'],
    hardGateValidator: () => ({ ok: true }),
    photoCache,
    fetchImpl,
    boardRenderer: (items, opts) => renderEditorialLookBoard(items, { ...opts, photoCache, fetchImpl }),
    visualReplacementSelector: async ({ candidates }) => {
      selectorSawCandidates = candidates.map((c) => c.id);
      const chosen = candidates.find((c) => c.id === 'shoes-black-oxford');
      return {
        status: 'selected',
        selectedItem: chosen,
        token: 'R3',
        rationale: 'Crisp black leather oxford sharpens the charcoal trouser line.',
        swapBoard: { width: 1400, height: 900, ok: true },
      };
    },
    visualCritic: async ([cand]) => [{
      candidateId: cand.id,
      reject: false,
      editorialScore: 8.8,
      confidence: 0.94,
      referenceless: true,
    }],
  });

  assert.ok(selectorSawCandidates.includes('shoes-black-oxford'));
  assert.ok(!selectorSawCandidates.includes('shoes-overbudget-loafer'), 'hard budget gate runs BEFORE visual selection');
  assert.equal(visualSwap.status, 'swapped');
  assert.equal(visualSwap.newItem.id, 'shoes-black-oxford');
  assert.equal(visualSwap.visualVerified, true);
  assert.equal(visualSwap.visualSwapUsed, true);
  assert.equal(visualSwap.visuallySelected, true);
  assert.equal(visualSwap.visualSwapToken, 'R3');

  // 2. Graceful fallback when visual selector fails/abstains
  const fallbackSwap = await swapSingleSlot({
    look: baseLook,
    targetSlot: 'shoes',
    reason: 'over_budget',
    catalogPool,
    context: { occasion: 'Work', budget: { amount: 1000, currency: 'USD' } },
    excludedItemIds: ['shoes-deck-used-loafer'],
    hardGateValidator: () => ({ ok: true }),
    visualReplacementSelector: async () => ({ status: 'fallback', selectedItem: null, reason: 'vision_unavailable' }),
  });

  assert.equal(fallbackSwap.status, 'swapped');
  assert.equal(fallbackSwap.newItem.id, 'shoes-budget-derby');
  assert.equal(fallbackSwap.visualVerified, false);
});

test('Phase 10 Defect A: rankSlotReplacementCandidates enforces brand/color/silhouette diversity across R1..R6 and caps original color on different-tone directive', () => {
  const look = {
    id: 'look-work-navy',
    items: [
      { id: 't1', macroCategory: 'top', subtype: 'dress-shirt', color: 'white', brand: 'Jack & Jones', name: 'White Poplin Shirt', price: 70, currency: 'USD' },
      { id: 'o1', macroCategory: 'outerwear', subtype: 'blazer', color: 'navy', brand: 'Armani Exchange', name: 'Navy Tailored Blazer', price: 320, currency: 'USD' },
      { id: 'b1', macroCategory: 'bottom', subtype: 'tailored-trouser', color: 'navy', brand: 'Giorgio Armani', name: 'Navy Wool Trousers', price: 420, currency: 'USD' },
      { id: 's0', macroCategory: 'shoes', subtype: 'derby', color: 'black', brand: 'Grant Stone', name: 'Black Plain Toe Derby', price: 380, currency: 'USD' },
    ],
  };

  // Flood the pool with 6 nearly identical black ECCO derbies + diverse tones/brands/subtypes
  const catalogPool = [
    ...Array.from({ length: 6 }, (_, i) => ({
      id: `ecco-black-derby-${i + 1}`,
      macroCategory: 'shoes',
      subtype: 'derby',
      color: 'black',
      brand: 'ECCO',
      name: `ECCO Metropole Black Leather Derby #${i + 1}`,
      price: 180 + i * 5,
      currency: 'USD',
      imageUrl: `https://static.zara.net/ecco-black-${i + 1}.jpg`,
      qualityScore: 0.95,
    })),
    { id: 'bs-burgundy-loafer', macroCategory: 'shoes', subtype: 'loafer', color: 'burgundy', brand: 'Beckett Simonon', name: 'Eriksen Bordeaux Loafer', price: 249, currency: 'USD', imageUrl: 'https://static.zara.net/bs-burg.jpg', qualityScore: 0.9 },
    { id: 'je-brown-oxford', macroCategory: 'shoes', subtype: 'oxford', color: 'brown', brand: 'Jack Erwin', name: 'Joe Espresso Cap-Toe Oxford', price: 228, currency: 'USD', imageUrl: 'https://static.zara.net/je-brown.jpg', qualityScore: 0.9 },
    { id: 'cu-navy-loafer', macroCategory: 'shoes', subtype: 'loafer', color: 'navy', brand: 'Cobbler Union', name: 'Navy Suede Penny Loafer', price: 320, currency: 'USD', imageUrl: 'https://static.zara.net/cu-navy.jpg', qualityScore: 0.88 },
    { id: 'tb-tan-chelsea', macroCategory: 'shoes', subtype: 'chelsea', color: 'tan', brand: 'Thursday Boot Co.', name: 'Cavalier Tan Chelsea Boot', price: 199, currency: 'USD', imageUrl: 'https://static.zara.net/tb-tan.jpg', qualityScore: 0.87 },
    { id: 'bk-espresso-monk', macroCategory: 'shoes', subtype: 'monk', color: 'brown', brand: 'Barker', name: 'Dark Brown Single Monk Strap', price: 295, currency: 'USD', imageUrl: 'https://static.zara.net/bk-monk.jpg', qualityScore: 0.86 },
    { id: 'bs-black-oxford', macroCategory: 'shoes', subtype: 'oxford', color: 'black', brand: 'Beckett Simonon', name: 'Dean Black Cap-Toe Oxford', price: 239, currency: 'USD', imageUrl: 'https://static.zara.net/bs-blk.jpg', qualityScore: 0.92 },
  ];

  const top6 = rankSlotReplacementCandidates({
    look,
    targetSlot: 'shoes',
    directive: 'try a different leather dress shoe tone while keeping the tailored silhouette',
    catalogPool,
    context: { occasion: 'Work', style: 'Old Money' },
    limit: 6,
  });

  assert.equal(top6.length, 6);
  const brandColors = top6.map((e) => `${String(e.item.brand).toLowerCase()}|${String(e.item.color).toLowerCase()}`);
  assert.equal(new Set(brandColors).size, 6, `max 1 per (brand, color), got: ${brandColors.join(', ')}`);
  const colors = top6.map((e) => String(e.item.color).toLowerCase());
  assert.ok(new Set(colors).size >= 4, `expected >=4 distinct color tones across R1..R6, got: ${colors.join(', ')}`);
  const blackCount = colors.filter((c) => c === 'black').length;
  assert.ok(blackCount <= 1, `original black color capped at <=1 slot when different tone requested, got ${blackCount}`);
  const eccoCount = top6.filter((e) => e.item.brand === 'ECCO').length;
  assert.ok(eccoCount <= 1, `ECCO must not flood R1..R6, got ${eccoCount}`);
});
