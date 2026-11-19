import test from 'node:test';
import assert from 'node:assert/strict';
import {
  feedbackTasteModel, evaluateFeedbackItem, evaluateFeedbackLook,
  FEEDBACK_TASTE_VERSION, FEEDBACK_TASTE_LOOK_CAP, FEEDBACK_TASTE_HALF_LIFE_DAYS,
} from './feedback_taste.mjs';

const NOW = Date.UTC(2026, 10, 13, 12, 0, 0); // fixed "now" for determinism
const daysAgo = (days) => NOW - days * 86_400_000;
const item = (id, extra = {}) => ({ id, name: `item ${id}`, ...extra });

test('malformed, weak and missing evidence degrade to an inert model', () => {
  for (const input of [null, undefined, 'bad', 42, {}, []]) {
    const model = feedbackTasteModel(input, NOW);
    assert.equal(model.active, false);
    assert.equal(model.version, FEEDBACK_TASTE_VERSION);
    assert.deepEqual(evaluateFeedbackLook([item('a')], 'clean-lines', model),
      { adjustment: 0, matches: [], likedItems: 0, dislikedItems: 0, conceptId: 'clean-lines' });
    assert.deepEqual(evaluateFeedbackItem(item('a'), model), { score: 0, match: null });
  }
  // Rows without explicit ±1 ratings or a parseable timestamp never count.
  const weak = feedbackTasteModel([
    { rating: 0, item_ids: ['a'], created_at: daysAgo(1) },
    { rating: 2, item_ids: ['a'], created_at: daysAgo(1) },
    { rating: '1', item_ids: ['a'], created_at: daysAgo(1) },
    { rating: true, item_ids: ['a'], created_at: daysAgo(1) },
    { rating: 0.5, item_ids: ['a'], created_at: daysAgo(1) },
    { rating: 1, item_ids: ['a'] },
    { rating: 1, created_at: 'not-a-date' },
    { rating: null, item_ids: ['a'], created_at: daysAgo(1) },
    { item_ids: ['a'], created_at: daysAgo(1) },
    'junk', 17, null,
    // A usable rating with no linkable key (no item ids, empty/unknown
    // concept-direction) contributes nothing at all.
    { rating: 1, created_at: '2026-11-13T12:00:00Z', concept_id: '' },
    { rating: -1, created_at: '2026-11-13T12:00:00Z', concept_id: 'clean-lines', direction: 'disco' },
  ], NOW);
  assert.equal(weak.active, false);
  assert.equal(weak.rows, 0);
});

test('explicit likes and dislikes are the only signals; both key spellings work', () => {
  const model = feedbackTasteModel([
    { rating: 1, item_ids: ['sku-1'], concept_id: 'clean-lines', direction: 'signature', created_at: daysAgo(0) },
    { rating: -1, itemIds: ['sku-2'], conceptId: 'modern-tailoring', directionLabel: 'refresh', createdAt: daysAgo(0) },
  ], NOW);
  assert.equal(model.active, true);
  assert.equal(model.rows, 2);
  assert.equal(model.itemTally.get('sku-1'), 1);
  assert.equal(model.itemTally.get('sku-2'), -1);
  assert.equal(model.pairTally.get('clean-lines|signature'), 1);
  assert.equal(model.pairTally.get('modern-tailoring|refresh'), -1);
});

test('recency decays with the 30-day half-life and very old rows expire', () => {
  const fresh = feedbackTasteModel([{ rating: 1, item_ids: ['x'], created_at: daysAgo(0) }], NOW);
  const half = feedbackTasteModel([{ rating: 1, item_ids: ['x'], created_at: daysAgo(FEEDBACK_TASTE_HALF_LIFE_DAYS) }], NOW);
  const quarter = feedbackTasteModel([{ rating: 1, item_ids: ['x'], created_at: daysAgo(2 * FEEDBACK_TASTE_HALF_LIFE_DAYS) }], NOW);
  assert.equal(fresh.itemTally.get('x'), 1);
  assert.ok(Math.abs(half.itemTally.get('x') - 0.5) < 1e-9);
  assert.ok(Math.abs(quarter.itemTally.get('x') - 0.25) < 1e-9);
  // Beyond a year the evidence is effectively gone, not stale-biased.
  const expired = feedbackTasteModel([{ rating: 1, item_ids: ['x'], created_at: daysAgo(366) }], NOW);
  assert.equal(expired.active, false);
  // A future timestamp is fresh evidence, never negative-age amplification.
  const future = feedbackTasteModel([{ rating: 1, item_ids: ['x'], created_at: daysAgo(-5) }], NOW);
  assert.equal(future.itemTally.get('x'), 1);
  // A decayed dislike cannot outvote a fresh like of the same item.
  const mixed = feedbackTasteModel([
    { rating: -1, item_ids: ['x'], created_at: daysAgo(60) },
    { rating: 1, item_ids: ['x'], created_at: daysAgo(0) },
  ], NOW);
  assert.ok(mixed.itemTally.get('x') > 0);
});

test('net tallies saturate at ±1: repetition says strong, never stronger', () => {
  const many = feedbackTasteModel(
    Array.from({ length: 60 }, (_, i) => ({ rating: 1, item_ids: ['sku-a'], created_at: daysAgo(0) })), NOW);
  assert.equal(many.itemTally.get('sku-a'), 1);
  // A like and a dislike of the same item cancel instead of averaging up.
  const cancelled = feedbackTasteModel([
    { rating: 1, item_ids: ['sku-a'], created_at: daysAgo(0) },
    { rating: -1, item_ids: ['sku-a'], created_at: daysAgo(0) },
  ], NOW);
  assert.equal(cancelled.itemTally.get('sku-a'), undefined);
  assert.equal(cancelled.active, false);
  // Distinct looks count as distinct evidence toward the same saturation.
  const twoLooks = feedbackTasteModel([
    { rating: -1, item_ids: ['sku-a', 'sku-b'], created_at: daysAgo(0) },
    { rating: -1, item_ids: ['sku-a', 'sku-c'], created_at: daysAgo(0) },
  ], NOW);
  assert.equal(twoLooks.itemTally.get('sku-a'), -1);
  assert.equal(twoLooks.itemTally.get('sku-b'), -1);
  assert.equal(twoLooks.itemTally.get('sku-c'), -1);
});

test('look adjustments are capped at ±0.06 and mix item + concept-direction evidence', () => {
  const liked = feedbackTasteModel([
    { rating: 1, item_ids: ['a', 'b', 'c', 'd'], concept_id: 'clean-lines', direction: 'signature', created_at: daysAgo(0) },
  ], NOW);
  const fullPositive = evaluateFeedbackLook([item('a'), item('b'), item('c'), item('d')], 'clean-lines', liked);
  assert.equal(fullPositive.adjustment, FEEDBACK_TASTE_LOOK_CAP);
  assert.equal(fullPositive.likedItems, 4);
  assert.ok(fullPositive.matches.every((match) => match.type === 'feedback_item' && match.value === 'liked'));
  const disliked = feedbackTasteModel([
    { rating: -1, item_ids: ['a', 'b', 'c', 'd'], concept_id: 'clean-lines', direction: 'signature', created_at: daysAgo(0) },
  ], NOW);
  const fullNegative = evaluateFeedbackLook([item('a'), item('b'), item('c'), item('d')], 'clean-lines', disliked);
  assert.equal(fullNegative.adjustment, -FEEDBACK_TASTE_LOOK_CAP);
  assert.equal(fullNegative.dislikedItems, 4);
  // Even 60 saturated rows cannot push past the cap.
  const hostile = feedbackTasteModel(
    Array.from({ length: 60 }, () => ({ rating: 1, item_ids: ['a', 'b', 'c', 'd'], concept_id: 'clean-lines', direction: 'signature', created_at: daysAgo(0) })), NOW);
  assert.equal(evaluateFeedbackLook([item('a'), item('b'), item('c'), item('d')], 'clean-lines', hostile).adjustment,
    FEEDBACK_TASTE_LOOK_CAP);
});

test('one disliked item moves a four-piece look only by its mean share', () => {
  const model = feedbackTasteModel([
    { rating: -1, item_ids: ['bad-shoe'], created_at: daysAgo(0) },
  ], NOW);
  const look = evaluateFeedbackLook(
    [item('top-1'), item('bottom-1'), item('bad-shoe'), item('outer-1')], 'clean-lines', model);
  assert.ok(Math.abs(look.adjustment - (-0.045 / 4)) < 1e-9);
  assert.equal(look.dislikedItems, 1);
  // The same shoe in a three-piece look has proportionally more say.
  const tighter = evaluateFeedbackLook([item('top-1'), item('bottom-1'), item('bad-shoe')], 'clean-lines', model);
  assert.ok(Math.abs(tighter.adjustment - (-0.045 / 3)) < 1e-9);
});

test('concept-direction transfer: signature full, refresh/discovery half, alternative quarter', () => {
  const pair = (direction) => feedbackTasteModel(
    [{ rating: 1, concept_id: 'clean-lines', direction, created_at: daysAgo(0) }], NOW);
  const look = [item('x1'), item('x2')];
  const of = (model) => evaluateFeedbackLook(look, 'clean-lines', model).adjustment;
  const signature = of(pair('signature'));
  assert.ok(Math.abs(signature - 0.03) < 1e-9);
  assert.ok(Math.abs(of(pair('refresh')) - 0.015) < 1e-9);
  assert.ok(Math.abs(of(pair('discovery')) - 0.015) < 1e-9);
  assert.ok(Math.abs(of(pair('alternative')) - 0.0075) < 1e-9);
  // A different concept earns nothing from this concept's feedback.
  assert.equal(of(pair('signature')) > 0, true);
  assert.equal(evaluateFeedbackLook(look, 'modern-tailoring', pair('signature')).adjustment, 0);
  // Unknown direction labels contribute nothing at all.
  assert.equal(feedbackTasteModel(
    [{ rating: 1, concept_id: 'clean-lines', direction: 'signature?', created_at: daysAgo(0) }], NOW).active, false);
});

test('unmatched items and unknown concepts are exactly neutral', () => {
  const model = feedbackTasteModel([
    { rating: -1, item_ids: ['gone-sku'], concept_id: 'clean-lines', direction: 'signature', created_at: daysAgo(0) },
  ], NOW);
  const untouched = evaluateFeedbackLook([item('new-1'), item('new-2'), item('new-3')], 'clean-lines', model);
  assert.equal(untouched.adjustment, -0.03); // concept transfer only
  assert.equal(evaluateFeedbackItem(item('new-1'), model).score, 0);
  assert.equal(evaluateFeedbackItem(item(null), model).score, 0);
  assert.equal(evaluateFeedbackItem(null, model).score, 0);
  assert.equal(evaluateFeedbackLook([], 'clean-lines', model).adjustment, 0);
});

test('hostile payloads never crash and never fabricate evidence', () => {
  const hostile = feedbackTasteModel([
    { rating: 1, item_ids: [null, 42, {}, '', '   ', 'ok-sku'.repeat(100)], created_at: daysAgo(0) },
    { rating: 1, item_ids: 'not-an-array', created_at: daysAgo(0) },
    { rating: -1, item_ids: Array.from({ length: 500 }, (_, i) => `x${i}`), created_at: daysAgo(0) },
    { rating: 1, concept_id: 99, direction: {}, created_at: daysAgo(0) },
    { rating: 1, item_ids: ['y'], created_at: Number.NaN },
    { rating: 1, item_ids: ['y'], created_at: ['array'] },
  ], NOW);
  assert.equal(hostile.itemTally.get('ok-sku'.repeat(100).slice(0, 128).trim()), 1);
  assert.equal(hostile.itemTally.get('x0'), -1);
  for (const junk of [null, undefined, 42, 'x', [], {}]) {
    assert.doesNotThrow(() => feedbackTasteModel([{ rating: 1, item_ids: ['a'], created_at: daysAgo(0) }], junk));
  }
  assert.doesNotThrow(() => evaluateFeedbackLook([{ id: { nested: true } }, 7, null], { id: 'x' }, hostile));
  assert.doesNotThrow(() => evaluateFeedbackItem({ id: ['array'] }, hostile));
});

test('id matching is exact: case, whitespace and prefixes are different items', () => {
  const model = feedbackTasteModel([{ rating: 1, item_ids: ['SKU-1'], created_at: daysAgo(0) }], NOW);
  assert.equal(model.itemTally.has('SKU-1'), true);
  assert.equal(evaluateFeedbackItem(item('sku-1'), model).score, 0);
  assert.equal(evaluateFeedbackItem(item(' SKU-1 '), model).score, 1);
});

test('the model is deterministic for a fixed input set and now', () => {
  const rows = [
    { rating: 1, item_ids: ['a', 'b'], concept_id: 'clean-lines', direction: 'signature', created_at: daysAgo(2) },
    { rating: -1, item_ids: ['c'], concept_id: 'modern-tailoring', direction: 'discovery', created_at: daysAgo(9) },
    { rating: 1, item_ids: ['a'], concept_id: 'clean-lines', direction: 'refresh', created_at: daysAgo(31) },
  ];
  const first = feedbackTasteModel(rows, NOW);
  const second = feedbackTasteModel([...rows].reverse(), NOW);
  const serial = (model) => JSON.stringify(
    [...model.itemTally, ...model.pairTally].sort((a, b) => String(a[0]).localeCompare(String(b[0]))));
  assert.equal(serial(first), serial(second));
  const lookFirst = evaluateFeedbackLook([item('a'), item('c')], 'clean-lines', first);
  const lookSecond = evaluateFeedbackLook([item('a'), item('c')], 'clean-lines', second);
  assert.equal(lookFirst.adjustment, lookSecond.adjustment);
});

test('the bounded read cap keeps 60+ rows from growing the influence', () => {
  const rows = Array.from({ length: 200 }, (_, index) => ({
    rating: index % 2 ? 1 : -1,
    item_ids: [`item-${index}`],
    created_at: daysAgo(0),
  }));
  const model = feedbackTasteModel(rows, NOW);
  assert.equal(model.rows, 60);
  assert.equal(model.itemTally.size, 60);
  // Rows arrive newest-first from the indexed read; the cap keeps the head
  // of that array, so the oldest rows beyond 60 are dropped.
  assert.equal(model.itemTally.has('item-59'), true);
  assert.equal(model.itemTally.has('item-60'), false);
  assert.equal(model.itemTally.has('item-199'), false);
});


test('cross-SKU fashion taste: a liked combination transfers without the same product IDs', () => {
  const features = [
    ['shirt', 'regular', 'cream', 'linen', 'plain'],
    ['blazer', 'relaxed', 'brown', 'wool', 'plain'],
    ['trousers', 'regular', 'taupe', 'wool', 'plain'],
    ['derby', 'regular', 'brown', 'leather', 'plain'],
  ];
  const model = feedbackTasteModel([{
    rating: 1,
    item_ids: ['old-1', 'old-2', 'old-3', 'old-4'],
    created_at: daysAgo(0),
    look_signature: { version: 1, direction: 'signature', features },
  }], NOW);
  const newItems = [item('new-1'), item('new-2'), item('new-3'), item('new-4')];
  const result = evaluateFeedbackLook(newItems, 'unseen-concept', model, features);
  assert.ok(result.adjustment > 0 && result.adjustment <= 0.02);
  assert.equal(result.likedItems, 0, 'no legacy SKU matched');
  assert.deepEqual(result.matches, [{ type: 'feedback_style', value: 'liked' }]);
  assert.equal(evaluateFeedbackLook(newItems, 'unseen-concept', model).adjustment, 0,
    'old callers without semantic vector retain byte-identical behavior');
});

test('cross-SKU dislike transfers softly, cancels with a like and obeys recency', () => {
  const features = [
    ['shirt', 'regular', 'cream', 'linen', 'plain'],
    null,
    ['trousers', 'regular', 'taupe', 'wool', 'plain'],
    ['derby', 'regular', 'brown', 'leather', 'plain'],
  ];
  const row = (rating, created_at) => ({
    rating, created_at, look_signature: { version: 1, features },
  });
  const look = [item('x'), item('y'), item('z')];
  const dislike = feedbackTasteModel([row(-1, daysAgo(0))], NOW);
  assert.ok(evaluateFeedbackLook(look, '', dislike, features).adjustment < 0);
  const cancelled = feedbackTasteModel([row(1, daysAgo(0)), row(-1, daysAgo(0))], NOW);
  assert.equal(evaluateFeedbackLook(look, '', cancelled, features).adjustment, 0);
  const old = feedbackTasteModel([row(1, daysAgo(30))], NOW);
  assert.ok(Math.abs(evaluateFeedbackLook(look, '', old, features).adjustment - 0.01) < 1e-9);
  assert.ok(Math.abs(evaluateFeedbackLook(look, '', dislike, features).adjustment) <= 0.02);
});

test('cross-SKU evidence never follows an unrelated silhouette, invalid or sparse signature', () => {
  const good = [
    ['shirt', 'regular', 'cream', 'linen', 'plain'],
    ['blazer', 'relaxed', 'brown', 'wool', 'plain'],
    ['trousers', 'regular', 'taupe', 'wool', 'plain'],
    ['derby', 'regular', 'brown', 'leather', 'plain'],
  ];
  const veryDifferent = [
    ['hoodie', 'oversized', 'red', 'jersey', 'graphic'],
    good[1],
    ['jeans', 'baggy', 'blue', 'denim', 'washed'],
    good[3],
  ];
  const model = feedbackTasteModel([{
    rating: 1, created_at: daysAgo(0),
    lookSignature: { version: 1, features: good },
  }], NOW);
  assert.equal(evaluateFeedbackLook([item('a'), item('b')], '', model, veryDifferent).adjustment, 0);
  assert.equal(evaluateFeedbackLook([item('a'), item('b')], '', model,
    Array(4).fill(['unknown', 'unknown', 'unknown', 'unknown', 'unknown'])).adjustment, 0);
  for (const bad of [
    { version: 2, features: good }, { version: 1, features: [good[0]] },
    { version: 1, features: Array(4).fill({ dangerous: true }) },
    { version: 1, features: [good[0], null, null, null] },
  ]) {
    const invalid = feedbackTasteModel([{ rating: 1, created_at: daysAgo(0), look_signature: bad }], NOW);
    assert.equal(invalid.active, false);
    assert.equal(invalid.rows, 0);
  }
});

test('feature-only feedback remains below the existing look quality cap', () => {
  const features = [
    ['shirt', 'regular', 'cream', 'linen', 'plain'],
    null,
    ['trousers', 'regular', 'taupe', 'wool', 'plain'],
    ['derby', 'regular', 'brown', 'leather', 'plain'],
  ];
  const rows = Array.from({ length: 100 }, () => ({
    rating: 1, created_at: daysAgo(0), look_signature: { version: 1, features },
  }));
  const model = feedbackTasteModel(rows, NOW);
  assert.equal(model.rows, 60);
  const result = evaluateFeedbackLook([item('a'), item('b'), item('c')], '', model, features);
  assert.equal(result.adjustment, 0.02);
  assert.ok(result.adjustment < FEEDBACK_TASTE_LOOK_CAP);
});

test('slot-level attribution (wrong_shoes / targetSlot) penalizes ONLY the shoe without penalizing top/coat/bottom', () => {
  const features = [
    ['shirt', 'regular', 'white', 'cotton', 'plain'],
    ['coat', 'relaxed', 'camel', 'wool', 'plain'],
    ['trousers', 'regular', 'navy', 'wool', 'plain'],
    ['sneaker', 'chunky', 'neon', 'synthetic', 'graphic'],
  ];
  const model = feedbackTasteModel([{
    rating: -1,
    item_ids: ['top-1', 'coat-1', 'bottom-1', 'shoe-bad'],
    target_slot: 'shoes',
    feedback_tags: ['wrong_shoes'],
    concept_id: 'clean-lines',
    direction: 'signature',
    look_signature: { version: 1, direction: 'signature', features },
    created_at: daysAgo(0),
  }], NOW);

  assert.equal(model.active, true);
  assert.equal(model.tagTally.get('wrong_shoes'), 1);
  // Innocent sibling items are completely unpenalized.
  assert.equal(evaluateFeedbackItem(item('top-1'), model).score, 0);
  assert.equal(evaluateFeedbackItem(item('coat-1'), model).score, 0);
  assert.equal(evaluateFeedbackItem(item('bottom-1'), model).score, 0);
  // The offending shoe carries the full targeted negative signal.
  assert.equal(evaluateFeedbackItem(item('shoe-bad'), model).score, -1);
  assert.equal(model.slotTally.get('shoes|shoe-bad'), -1);

  // Reusing the great top, coat, and trousers with a DIFFERENT shoe has zero penalty!
  const repairedFeatures = [
    features[0],
    features[1],
    features[2],
    ['loafer', 'regular', 'brown', 'suede', 'plain'],
  ];
  const repairedLook = evaluateFeedbackLook(
    [item('top-1'), item('coat-1'), item('bottom-1'), item('shoe-good')],
    'clean-lines',
    model,
    repairedFeatures,
  );
  assert.equal(repairedLook.adjustment, 0);
  assert.equal(repairedLook.dislikedItems, 0);

  // Re-introducing the bad shoe gets full targeted penalty (not diluted by /4) and stays within cap.
  const badLook = evaluateFeedbackLook(
    [item('top-2'), item('coat-2'), item('bottom-2'), item('shoe-bad')],
    'clean-lines',
    model,
    features,
  );
  assert.equal(badLook.dislikedItems, 1);
  assert.ok(badLook.adjustment <= -0.045, `expected strong targeted penalty, got ${badLook.adjustment}`);
  assert.ok(badLook.adjustment >= -FEEDBACK_TASTE_LOOK_CAP);
});

test('swappedFromSku penalizes only the swapped-away SKU even on a positive rating for the updated look', () => {
  const model = feedbackTasteModel([{
    rating: 1,
    itemIds: ['top-1', 'coat-1', 'bottom-1', 'shoe-new'],
    targetSlot: 'shoes',
    feedbackTags: ['slot_swapped', 'love_palette'],
    swappedFromSku: 'shoe-old',
    createdAt: daysAgo(0),
  }], NOW);

  assert.equal(model.active, true);
  assert.equal(evaluateFeedbackItem(item('shoe-old'), model).score, -1);
  assert.equal(model.swappedFromTally.get('shoe-old'), -1);
  // Kept pieces and the new shoe receive positive credit.
  assert.equal(evaluateFeedbackItem(item('top-1'), model).score, 1);
  assert.equal(evaluateFeedbackItem(item('shoe-new'), model).score, 1);
  assert.equal(model.tagTally.get('slot_swapped'), 1);
  assert.equal(model.tagTally.get('love_palette'), 1);
});

test('wrong_shoes tag alone implies shoes slot even when targetSlot is omitted', () => {
  const model = feedbackTasteModel([{
    rating: -1,
    item_ids: ['top-x', 'coat-x', 'trouser-x', 'bad-boot'],
    feedback_tags: ['wrong_shoes', 'too_casual', 'unknown_junk_tag'],
    created_at: daysAgo(0),
  }], NOW);

  assert.equal(evaluateFeedbackItem(item('top-x'), model).score, 0);
  assert.equal(evaluateFeedbackItem(item('coat-x'), model).score, 0);
  assert.equal(evaluateFeedbackItem(item('trouser-x'), model).score, 0);
  assert.equal(evaluateFeedbackItem(item('bad-boot'), model).score, -1);
  assert.equal(model.tagTally.get('wrong_shoes'), 1);
  assert.equal(model.tagTally.get('too_casual'), 1);
  assert.equal(model.tagTally.has('unknown_junk_tag'), false);
});

test('slot-level feature evidence transfers softly to another shoe with matching subtype+color without penalizing coats', () => {
  const features = [
    ['shirt', 'regular', 'white', 'cotton', 'plain'],
    ['blazer', 'regular', 'navy', 'wool', 'plain'],
    ['trousers', 'regular', 'grey', 'wool', 'plain'],
    ['sneaker', 'oversized', 'white', 'leather', 'plain'],
  ];
  const model = feedbackTasteModel([{
    rating: -1,
    item_ids: ['top-1', 'blazer-1', 'trouser-1', 'sneaker-1'],
    targetSlot: 'shoes',
    feedbackTags: ['wrong_shoes'],
    look_signature: { version: 1, direction: 'signature', features },
    created_at: daysAgo(0),
  }], NOW);

  // Another look with a different oversized white sneaker gets negative cross-SKU slot transfer
  const anotherWhiteSneakerLook = [
    ['polo', 'regular', 'cream', 'knit', 'plain'],
    ['coat', 'relaxed', 'camel', 'wool', 'plain'],
    ['chino', 'regular', 'beige', 'cotton', 'plain'],
    ['sneaker', 'oversized', 'white', 'leather', 'plain'],
  ];
  const resSneaker = evaluateFeedbackLook(
    [item('t2'), item('o2'), item('b2'), item('s2')],
    '',
    model,
    anotherWhiteSneakerLook,
  );
  assert.ok(resSneaker.adjustment < 0, `expected negative cross-SKU slot transfer, got ${resSneaker.adjustment}`);

  // A look reusing the exact same blazer+shirt+trousers but with suede loafers has 0 penalty!
  const loaferLook = [
    features[0],
    features[1],
    features[2],
    ['loafer', 'regular', 'espresso', 'suede', 'plain'],
  ];
  const resLoafer = evaluateFeedbackLook(
    [item('t3'), item('o3'), item('b3'), item('s3')],
    '',
    model,
    loaferLook,
  );
  assert.equal(resLoafer.adjustment, 0);
});
