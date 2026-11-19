import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildUnifiedTasteProfile,
  scoreItemTasteAffinity,
  scoreLookTasteAffinity,
  summarizeTasteForPrompt,
  TASTE_LEARNING_VERSION,
  TASTE_EMBEDDING_DIM,
  MAX_ITEM_TASTE_SCORE,
  MAX_LOOK_TASTE_SCORE,
  MAX_VISUAL_ITEM_AFFINITY,
  MAX_VISUAL_LOOK_AFFINITY,
} from './taste_learning.mjs';

const NOW = Date.UTC(2026, 10, 19, 12, 0, 0);
const daysAgo = (d) => NOW - d * 86_400_000;

function makeUnitVec(seedIdx, sign = 1) {
  const vec = new Array(TASTE_EMBEDDING_DIM).fill(0);
  vec[seedIdx % TASTE_EMBEDDING_DIM] = sign * 0.8;
  vec[(seedIdx + 1) % TASTE_EMBEDDING_DIM] = sign * 0.6;
  return vec;
}

function makeItem(id, slot, opts = {}) {
  return {
    id,
    slot,
    macroCategory: slot,
    subtype: opts.subtype || 'shirt',
    color: opts.color || 'navy',
    brand: opts.brand || 'Massimo Dutti',
    formality: opts.formality || 'smart_casual',
    volume: opts.volume || 'regular',
    price: opts.price || 150,
    imageSha256: opts.imageSha256 || null,
  };
}

test('empty or malformed inputs return an inert neutral profile', () => {
  const empty = buildUnifiedTasteProfile({ nowMs: NOW });
  assert.equal(empty.version, TASTE_LEARNING_VERSION);
  assert.equal(empty.active, false);
  const item = makeItem('sku-1', 'top');
  assert.equal(scoreItemTasteAffinity(item, 'top', empty).score, 0);
  assert.equal(scoreLookTasteAffinity([item], empty).score, 0);
  assert.equal(summarizeTasteForPrompt(empty).active, false);
  assert.equal(summarizeTasteForPrompt(empty).promptBlock, '');

  const hostile = buildUnifiedTasteProfile({
    explicitFeedback: [null, 'junk', { rating: 2, created_at: daysAgo(1) }, { rating: 1, created_at: 'bad' }],
    savedOutfits: [null, 42, { items: 'not-array' }],
    wearLogs: [null, { item_ids: null }],
    brandClicks: [null, { brand: 123 }],
    nowMs: NOW,
  });
  assert.equal(hostile.active, false);
});

test('wrong_shoes fixes the blunt itemSum/4 penalty bug: ONLY downweights the shoe without penalizing coat/top/trouser', () => {
  const top = makeItem('top-oxford', 'top', { subtype: 'shirt', color: 'white', brand: 'Suitsupply' });
  const coat = makeItem('coat-camel', 'outerwear', { subtype: 'coat', color: 'camel', brand: 'Loro Piana' });
  const trouser = makeItem('trouser-navy', 'bottom', { subtype: 'tailored-trouser', color: 'navy', brand: 'Incotex' });
  const badShoe = makeItem('shoe-chunky', 'shoes', { subtype: 'sneaker', color: 'green', brand: 'Generic' });
  const similarBadShoe = makeItem('shoe-chunky-2', 'shoes', { subtype: 'sneaker', color: 'green', brand: 'OtherBrand' });
  const goodLoafer = makeItem('shoe-loafer', 'shoes', { subtype: 'loafer', color: 'espresso', brand: 'Velasca' });
  const greenCoat = makeItem('coat-green', 'outerwear', { subtype: 'coat', color: 'green', brand: 'Barbour' });

  const catalogById = new Map(
    [top, coat, trouser, badShoe, similarBadShoe, goodLoafer, greenCoat].map((x) => [x.id, x]),
  );

  const profile = buildUnifiedTasteProfile({
    explicitFeedback: [{
      rating: -1,
      item_ids: [top.id, coat.id, trouser.id, badShoe.id],
      target_slot: 'shoes',
      feedback_tags: ['wrong_shoes'],
      created_at: daysAgo(1),
    }],
    catalogById,
    nowMs: NOW,
  });

  assert.equal(profile.active, true);

  // Coat, top, and trouser MUST NOT be penalized!
  assert.equal(scoreItemTasteAffinity(top, 'top', profile).score, 0);
  assert.equal(scoreItemTasteAffinity(coat, 'outerwear', profile).score, 0);
  assert.equal(scoreItemTasteAffinity(trouser, 'bottom', profile).score, 0);
  // Even a green COAT is NOT penalized just because a green sneaker was marked wrong_shoes!
  assert.equal(scoreItemTasteAffinity(greenCoat, 'outerwear', profile).score, 0);

  // The offending shoe IS strongly downweighted and marked targetedPenalty!
  const badShoeScore = scoreItemTasteAffinity(badShoe, 'shoes', profile);
  assert.equal(badShoeScore.targetedPenalty, true);
  assert.ok(badShoeScore.score <= -0.25, `expected strong penalty on wrong_shoes SKU, got ${badShoeScore.score}`);

  // Another sneaker of the same rejected shoe subtype/color gets soft slot-scoped transfer!
  const simShoeScore = scoreItemTasteAffinity(similarBadShoe, 'shoes', profile);
  assert.equal(simShoeScore.targetedPenalty, false);
  assert.ok(simShoeScore.score < -0.05, `expected soft slot transfer penalty, got ${simShoeScore.score}`);

  // A suede loafer is unaffected!
  assert.equal(scoreItemTasteAffinity(goodLoafer, 'shoes', profile).score, 0);

  // Whole-look scoring:
  // 1) Keeping top + coat + trouser and swapping ONLY to goodLoafer yields 0 penalty!
  const fixedLook = scoreLookTasteAffinity([top, coat, trouser, goodLoafer], profile);
  assert.equal(fixedLook.score, 0);
  assert.equal(fixedLook.targetedPenalties.length, 0);

  // 2) Re-introducing badShoe in a 4-item look applies full targeted penalty (not diluted 4x)!
  const badLook = scoreLookTasteAffinity([top, coat, trouser, badShoe], profile);
  assert.equal(badLook.targetedPenalties.length, 1);
  assert.ok(badLook.score <= -0.2, `expected full non-diluted penalty, got ${badLook.score}`);
});

test('MEASURABLE RECOMMENDATION SHIFT: fixed Smart-Casual Work scenario before vs after explicit + implicit feedback', () => {
  // Build 4 realistic 4-item candidate looks with fixed baseline editorial scores.
  const catalogItems = [
    // Look A (baseline #1 = 0.78): Casual hoodie + sporty sneaker (user will mark wrong_shoes + too_casual)
    makeItem('a-top', 'top', { subtype: 't-shirt', color: 'grey', brand: 'StreetCo', formality: 'casual', volume: 'oversized', price: 85 }),
    makeItem('a-outer', 'outerwear', { subtype: 'bomber', color: 'black', brand: 'StreetCo', formality: 'casual', volume: 'oversized', price: 240 }),
    makeItem('a-bottom', 'bottom', { subtype: 'jeans', color: 'blue', brand: 'StreetCo', formality: 'casual', volume: 'wide', price: 140 }),
    makeItem('a-shoes', 'shoes', { subtype: 'sneaker', color: 'red', brand: 'RunLab', formality: 'casual', volume: 'oversized', price: 160 }),

    // Look B (baseline #2 = 0.74): Refined tailored smart-casual (navy/camel/cream, suede loafer, Velasca/Suitsupply)
    makeItem('b-top', 'top', { subtype: 'fine-knit', color: 'cream', brand: 'Suitsupply', formality: 'smart_casual', volume: 'regular', price: 130 }),
    makeItem('b-outer', 'outerwear', { subtype: 'blazer', color: 'camel', brand: 'Suitsupply', formality: 'tailored', volume: 'regular', price: 390 }),
    makeItem('b-bottom', 'bottom', { subtype: 'tailored-trouser', color: 'navy', brand: 'Incotex', formality: 'tailored', volume: 'regular', price: 220 }),
    makeItem('b-shoes', 'shoes', { subtype: 'loafer', color: 'brown', brand: 'Velasca', formality: 'smart_casual', volume: 'regular', price: 290 }),

    // Look C (baseline #3 = 0.72): Ultra-formal black suit + patent oxford
    makeItem('c-top', 'top', { subtype: 'dress-shirt', color: 'white', brand: 'Brioni', formality: 'formal', volume: 'slim', price: 350 }),
    makeItem('c-outer', 'outerwear', { subtype: 'blazer', color: 'black', brand: 'Brioni', formality: 'formal', volume: 'slim', price: 1400 }),
    makeItem('c-bottom', 'bottom', { subtype: 'tailored-trouser', color: 'black', brand: 'Brioni', formality: 'formal', volume: 'slim', price: 550 }),
    makeItem('c-shoes', 'shoes', { subtype: 'oxford', color: 'black', brand: 'Churchs', formality: 'formal', volume: 'regular', price: 750 }),
  ];

  const catalogById = new Map(catalogItems.map((x) => [x.id, x]));
  const candidates = [
    { id: 'look-A-street', baseScore: 0.78, items: catalogItems.slice(0, 4) },
    { id: 'look-B-tailored-camel', baseScore: 0.74, items: catalogItems.slice(4, 8) },
    { id: 'look-C-ultra-formal', baseScore: 0.72, items: catalogItems.slice(8, 12) },
  ];

  const rankWith = (profile) => candidates
    .map((c) => {
      const taste = scoreLookTasteAffinity(c.items, profile);
      return { id: c.id, total: +(c.baseScore + taste.score).toFixed(4), tasteDelta: taste.score };
    })
    .sort((a, b) => b.total - a.total);

  // BEFORE feedback: empty profile -> baseline ranking puts look-A-street first
  const beforeProfile = buildUnifiedTasteProfile({ catalogById, nowMs: NOW });
  const beforeRanking = rankWith(beforeProfile);
  assert.equal(beforeRanking[0].id, 'look-A-street');
  assert.equal(beforeRanking[0].tasteDelta, 0);

  // AFTER explicit + implicit feedback:
  // - User thumbs-down look-A for `wrong_shoes` + `too_casual`
  // - User thumbs-up a camel+navy+cream palette (`love_palette`, `love_silhouette`)
  // - User saved an outfit with Suitsupply + Velasca loafers 3 days ago
  // - User wore tailored navy trousers + cream fine-knit 5 days ago
  // - User clicked Velasca + Incotex products 2 days ago
  const afterProfile = buildUnifiedTasteProfile({
    explicitFeedback: [
      {
        rating: -1,
        item_ids: ['a-top', 'a-outer', 'a-bottom', 'a-shoes'],
        target_slot: 'shoes',
        feedback_tags: ['wrong_shoes', 'too_casual'],
        created_at: daysAgo(1),
      },
      {
        rating: 1,
        item_ids: ['b-top', 'b-outer', 'b-bottom', 'b-shoes'],
        feedback_tags: ['love_palette', 'love_silhouette'],
        created_at: daysAgo(2),
      },
    ],
    savedOutfits: [
      { items: ['b-outer', 'b-shoes'], occasion: 'work', created_at: daysAgo(3) },
    ],
    wearLogs: [
      { item_ids: ['b-top', 'b-bottom'], date: new Date(daysAgo(5)).toISOString() },
    ],
    brandClicks: [
      { item_id: 'b-shoes', brand: 'Velasca', price: 290, clicked_at: daysAgo(2) },
      { item_id: 'b-bottom', brand: 'Incotex', price: 220, clicked_at: daysAgo(2) },
    ],
    catalogById,
    nowMs: NOW,
  });

  const afterRanking = rankWith(afterProfile);

  // Measurable shift: look-B-tailored-camel rises from #2 to #1, look-A drops to last!
  assert.equal(afterRanking[0].id, 'look-B-tailored-camel');
  assert.equal(afterRanking[afterRanking.length - 1].id, 'look-A-street');

  const lookBDelta = afterRanking.find((x) => x.id === 'look-B-tailored-camel').tasteDelta;
  const lookADelta = afterRanking.find((x) => x.id === 'look-A-street').tasteDelta;
  assert.ok(lookBDelta >= 0.12, `expected strong positive lift on Look B, got ${lookBDelta}`);
  assert.ok(lookADelta <= -0.15, `expected strong negative shift on Look A, got ${lookADelta}`);
  assert.ok(lookBDelta - lookADelta >= 0.3, `expected >=0.30 separation, got ${lookBDelta - lookADelta}`);
});

test('all 9 feedbackTags and swappedFromSku steer taste dimensions accurately', () => {
  const redTop = makeItem('t-red', 'top', { brand: 'LoudLabel', color: 'red', volume: 'oversized', formality: 'formal', price: 600 });
  const navyTop = makeItem('t-navy', 'top', { brand: 'QuietTailor', color: 'navy', volume: 'regular', formality: 'casual', price: 110 });
  const oldShoe = makeItem('s-old', 'shoes', { brand: 'MonkMaker', subtype: 'monk', color: 'tan', price: 450 });
  const newShoe = makeItem('s-new', 'shoes', { brand: 'Velasca', subtype: 'loafer', color: 'brown', price: 220 });
  const catalogById = new Map([redTop, navyTop, oldShoe, newShoe].map((x) => [x.id, x]));

  const profile = buildUnifiedTasteProfile({
    explicitFeedback: [
      {
        rating: -1,
        item_ids: ['t-red'],
        feedback_tags: ['wrong_color', 'not_my_fit', 'too_formal', 'over_budget'],
        created_at: daysAgo(0),
      },
      {
        rating: 1,
        item_ids: ['t-navy', 's-new'],
        target_slot: 'shoes',
        feedback_tags: ['slot_swapped', 'love_palette', 'love_silhouette'],
        swapped_from_sku: 's-old',
        created_at: daysAgo(0),
      },
    ],
    savedOutfits: [
      { items: [navyTop], created_at: daysAgo(1) },
      { items: [newShoe], created_at: daysAgo(2) },
      { items: [{ id: 'ref-3', price: 150, color: 'navy', slot: 'bottom' }], created_at: daysAgo(3) },
    ],
    catalogById,
    nowMs: NOW,
  });

  assert.ok(profile.formalityShift < 0, 'too_formal shifts formalityShift negative');
  assert.ok(profile.priceSensitivity > 0.3, 'over_budget raises priceSensitivity');

  const redScore = scoreItemTasteAffinity(redTop, 'top', profile);
  const navyScore = scoreItemTasteAffinity(navyTop, 'top', profile);
  const oldShoeScore = scoreItemTasteAffinity(oldShoe, 'shoes', profile);
  const newShoeScore = scoreItemTasteAffinity(newShoe, 'shoes', profile);

  assert.ok(redScore.score < -0.1, `expected red/oversized/formal/expensive penalized, got ${redScore.score}`);
  assert.ok(navyScore.score > 0.1, `expected navy/regular/casual/affordable boosted, got ${navyScore.score}`);
  assert.equal(oldShoeScore.targetedPenalty, true);
  assert.ok(oldShoeScore.score < -0.2);
  assert.ok(newShoeScore.score > 0.1);
});

test('implicit signals obey signal hierarchy and exponential recency decay', () => {
  const wornBrand = makeItem('w-1', 'top', { brand: 'Brunello', color: 'cream', subtype: 'fine-knit', formality: 'smart_casual', volume: 'regular' });
  const savedBrand = makeItem('s-1', 'outerwear', { brand: 'Zegna', color: 'stone', subtype: 'blazer', formality: 'tailored', volume: 'slim' });
  const clickedBrand = makeItem('c-1', 'bottom', { brand: 'Zara', color: 'black', subtype: 'chino', formality: 'casual', volume: 'wide' });
  const staleWorn = makeItem('stale-1', 'shoes', { brand: 'OldLabel', color: 'olive', subtype: 'boot', formality: 'formal', volume: 'oversized' });

  const catalogById = new Map([wornBrand, savedBrand, clickedBrand, staleWorn].map((x) => [x.id, x]));

  const profile = buildUnifiedTasteProfile({
    wearLogs: [
      { item_ids: [wornBrand.id], date: new Date(daysAgo(0)).toISOString() },
      { item_ids: [staleWorn.id], date: new Date(daysAgo(180)).toISOString() },
    ],
    savedOutfits: [
      { items: [savedBrand.id], created_at: daysAgo(0) },
    ],
    brandClicks: [
      { item_id: clickedBrand.id, brand: 'Zara', clicked_at: daysAgo(0) },
    ],
    catalogById,
    nowMs: NOW,
  });

  const wScore = scoreItemTasteAffinity(wornBrand, 'top', profile).score;
  const sScore = scoreItemTasteAffinity(savedBrand, 'outerwear', profile).score;
  const cScore = scoreItemTasteAffinity(clickedBrand, 'bottom', profile).score;
  const staleScore = scoreItemTasteAffinity(staleWorn, 'shoes', profile).score;

  assert.ok(wScore > sScore, `wearLog (${wScore}) should outweigh savedOutfit (${sScore})`);
  assert.ok(sScore > cScore, `savedOutfit (${sScore}) should outweigh brandClick (${cScore})`);
  assert.ok(cScore > 0, `brandClick (${cScore}) should still give positive lift`);
  assert.ok(wScore > staleScore * 2, `fresh wear (${wScore}) should strongly outweigh 180d-old wear (${staleScore})`);
});

test('768-d positive/negative visual centroids compute bounded cosine affinity per ADR and ignore corrupted vectors', () => {
  const likedSha = 'a'.repeat(64);
  const dislikedSha = 'b'.repeat(64);
  const matchLikedSha = 'c'.repeat(64);
  const matchDislikedSha = 'd'.repeat(64);
  const badDimSha = 'e'.repeat(64);

  const embeddingsByImageSha = new Map([
    [likedSha, makeUnitVec(10, 1)],
    [dislikedSha, makeUnitVec(200, 1)],
    [matchLikedSha, makeUnitVec(10, 1)], // cosine +1.0 with liked, 0 with disliked
    [matchDislikedSha, makeUnitVec(200, 1)], // cosine 0 with liked, +1.0 with disliked
    [badDimSha, [1, 2, 3]], // invalid dimension — must be safely ignored
  ]);

  const likedItem = makeItem('vis-like', 'outerwear', { imageSha256: likedSha, color: 'unknown', brand: '' });
  const dislikedItem = makeItem('vis-dislike', 'shoes', { imageSha256: dislikedSha, color: 'unknown', brand: '' });
  const unseenMatchPos = makeItem('vis-cand-pos', 'outerwear', { imageSha256: matchLikedSha, color: 'unknown', brand: '' });
  const unseenMatchNeg = makeItem('vis-cand-neg', 'shoes', { imageSha256: matchDislikedSha, color: 'unknown', brand: '' });
  const badVecItem = makeItem('vis-bad', 'top', { imageSha256: badDimSha, color: 'unknown', brand: '' });

  const catalogById = new Map([likedItem, dislikedItem, badVecItem].map((x) => [x.id, x]));

  const profile = buildUnifiedTasteProfile({
    explicitFeedback: [
      { rating: 1, item_ids: [likedItem.id], created_at: daysAgo(0) },
      { rating: -1, item_ids: [dislikedItem.id], target_slot: 'shoes', feedback_tags: ['wrong_shoes'], created_at: daysAgo(0) },
      { rating: 1, item_ids: [badVecItem.id], created_at: daysAgo(0) },
    ],
    catalogById,
    embeddingsByImageSha,
    nowMs: NOW,
  });

  assert.equal(profile.visualCentroids.dim, TASTE_EMBEDDING_DIM);
  assert.equal(profile.visualCentroids.positiveCount, 1);
  assert.equal(profile.visualCentroids.negativeCount, 1);
  assert.equal(profile.visualCentroids.cannotOverrideQualityGate, true);

  const posRes = scoreItemTasteAffinity(unseenMatchPos, 'outerwear', profile);
  const negRes = scoreItemTasteAffinity(unseenMatchNeg, 'shoes', profile);
  const badRes = scoreItemTasteAffinity(badVecItem, 'top', profile);

  assert.ok(posRes.components.visualEmbedding > 0.05);
  assert.ok(posRes.components.visualEmbedding <= MAX_VISUAL_ITEM_AFFINITY);
  assert.ok(negRes.components.visualEmbedding < -0.05);
  assert.ok(negRes.components.visualEmbedding >= -MAX_VISUAL_ITEM_AFFINITY);
  assert.equal(badRes.components.visualEmbedding, 0);

  // Look-level visual affinity is strictly capped at ±MAX_VISUAL_LOOK_AFFINITY
  const lookRes = scoreLookTasteAffinity([unseenMatchPos, unseenMatchPos, unseenMatchPos, unseenMatchPos], profile);
  assert.ok(Math.abs(lookRes.visualScore) <= MAX_VISUAL_LOOK_AFFINITY);
  assert.ok(Math.abs(lookRes.score) <= MAX_LOOK_TASTE_SCORE);
  assert.ok(Math.abs(posRes.score) <= MAX_ITEM_TASTE_SCORE);
});

test('summarizeTasteForPrompt produces concise zero-PII prompt guidance with injection sanitization', () => {
  const item1 = makeItem('secret-sku-9981', 'shoes', {
    subtype: 'sneaker',
    color: 'green',
    brand: 'Ignore <script> https://evil.com user@test.com',
  });
  const item2 = makeItem('secret-sku-4422', 'outerwear', {
    subtype: 'blazer',
    color: 'camel',
    brand: 'Suitsupply',
    volume: 'regular',
  });
  const catalogById = new Map([item1, item2].map((x) => [x.id, x]));

  const profile = buildUnifiedTasteProfile({
    explicitFeedback: [
      { rating: -1, item_ids: [item1.id], target_slot: 'shoes', feedback_tags: ['wrong_shoes', 'too_casual'], created_at: daysAgo(0) },
      { rating: 1, item_ids: [item2.id], feedback_tags: ['love_palette', 'love_silhouette'], created_at: daysAgo(0) },
    ],
    brandClicks: [
      { item_id: item2.id, brand: 'Suitsupply', clicked_at: daysAgo(0) },
    ],
    catalogById,
    nowMs: NOW,
  });

  const summary = summarizeTasteForPrompt(profile);
  assert.equal(summary.active, true);
  assert.equal(summary.formalityShift, 'more_formal');
  assert.ok(summary.preferredColors.includes('camel'));
  assert.ok(summary.slotPreferences.shoes.dislikedSubtypes.includes('sneaker'));
  assert.ok(summary.promptBlock.includes('LEARNED PERSONAL TASTE'));
  // Zero PII / zero SKU IDs / zero URLs / zero email leakage!
  assert.equal(summary.promptBlock.includes('secret-sku-9981'), false);
  assert.equal(summary.promptBlock.includes('secret-sku-4422'), false);
  assert.equal(summary.promptBlock.includes('https://'), false);
  assert.equal(summary.promptBlock.includes('user@test.com'), false);
  assert.equal(summary.promptBlock.includes('<script>'), false);
});
