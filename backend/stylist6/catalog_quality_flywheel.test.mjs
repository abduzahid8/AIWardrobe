import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createQualityFlywheel,
  recordVisualCriticOutcome,
  computeCatalogQualityScore,
  computeWilsonLowerBound,
  isSkuQuarantined,
  getQuarantineStatus,
  buildQualityLookupMap,
  DEFAULT_NEUTRAL_QUALITY_SCORE,
} from './catalog_quality_flywheel.mjs';

test('new unseen SKUs start neutral at ~0.65 and are never quarantined', () => {
  assert.equal(computeCatalogQualityScore({}), DEFAULT_NEUTRAL_QUALITY_SCORE);
  assert.equal(computeCatalogQualityScore(null), DEFAULT_NEUTRAL_QUALITY_SCORE);
  assert.equal(
    computeCatalogQualityScore({ approvals: 0, rejections: 0, photoDefects: 0, categoryMismatches: 0 }),
    0.65,
  );
  assert.equal(isSkuQuarantined({}), false);
  assert.equal(isSkuQuarantined({ approvals: 0, rejections: 0 }), false);
});

test('Bayesian/Wilson shrinkage rewards consistent approvals and downranks low-approval SKUs smoothly', () => {
  const neutral = computeCatalogQualityScore({ approvals: 0, rejections: 0 });
  const oneApprove = computeCatalogQualityScore({ approvals: 1, rejections: 0, avgEditorialScore: 8.5 });
  const tenApproves = computeCatalogQualityScore({ approvals: 10, rejections: 0, avgEditorialScore: 8.8 });
  const mixed = computeCatalogQualityScore({ approvals: 3, rejections: 2, avgEditorialScore: 6.5 });
  const poor = computeCatalogQualityScore({ approvals: 1, rejections: 4, avgEditorialScore: 4.0 });

  assert.ok(tenApproves > oneApprove, `10/10 (${tenApproves}) should beat 1/1 (${oneApprove})`);
  assert.ok(oneApprove > neutral, `1/1 (${oneApprove}) should beat neutral (${neutral})`);
  assert.ok(neutral > mixed, `neutral (${neutral}) should beat 3/5 (${mixed})`);
  assert.ok(mixed > poor, `3/5 (${mixed}) should beat 1/5 (${poor})`);
  assert.ok(tenApproves <= 1 && poor >= 0);
});

test('quarantine rules enforce exact thresholds for photoDefects, categoryMismatches, and Wilson < 0.25 after >= 5 reviews', () => {
  // 1. Chronic photo defects (>= 3)
  assert.equal(isSkuQuarantined({ photoDefects: 2 }), false);
  assert.equal(isSkuQuarantined({ photoDefects: 3 }), true);
  assert.equal(getQuarantineStatus({ photoDefects: 3 }).reason, 'chronic_photo_defects');

  // 2. Category mismatches (>= 2)
  assert.equal(isSkuQuarantined({ categoryMismatches: 1 }), false);
  assert.equal(isSkuQuarantined({ categoryMismatches: 2 }), true);
  assert.equal(getQuarantineStatus({ categoryMismatches: 2 }).reason, 'category_mismatch');

  // 3. Wilson lower bound < 0.25 ONLY after >= 5 reviews
  assert.equal(isSkuQuarantined({ approvals: 0, rejections: 4 }), false, 'only 4 reviews — wait for 5');
  assert.equal(isSkuQuarantined({ approvals: 0, rejections: 5 }), true);
  assert.equal(isSkuQuarantined({ approvals: 1, rejections: 4 }), true);
  assert.equal(isSkuQuarantined({ approvals: 2, rejections: 3 }), true);
  assert.ok(computeWilsonLowerBound(2, 5) < 0.25);
  assert.equal(getQuarantineStatus({ approvals: 1, rejections: 4 }).reason, 'low_wilson_approval');

  // 3/5 approvals has Wilson > 0.25 -> survives!
  assert.ok(computeWilsonLowerBound(3, 5) >= 0.25);
  assert.equal(isSkuQuarantined({ approvals: 3, rejections: 2 }), false);
});

test('recordVisualCriticOutcome attributes weakestSlot rejections/defects surgically without harming innocent items', async () => {
  const store = new Map();
  const items = [
    { id: 'top-good', slot: 'top' },
    { id: 'coat-good', slot: 'outerwear' },
    { id: 'bottom-good', slot: 'bottom' },
    { id: 'shoe-broken-img', slot: 'shoes' },
  ];

  // Record 3 photo-defect rejections where weakestSlot === 'shoes'
  for (let i = 0; i < 3; i++) {
    await recordVisualCriticOutcome(store, {
      approved: false,
      items,
      weakestSlot: 'shoes',
      defectCategory: 'broken_photo',
      editorialScore: 3.5,
    });
  }

  // Innocent sibling items were NEVER penalized!
  assert.equal(store.has('top-good'), false);
  assert.equal(store.has('coat-good'), false);
  assert.equal(store.has('bottom-good'), false);

  // Broken shoe SKU is now quarantined with chronic_photo_defects and heavily downranked
  const badShoe = store.get('shoe-broken-img');
  assert.equal(badShoe.photoDefects, 3);
  assert.equal(badShoe.rejections, 3);
  assert.equal(badShoe.quarantined, true);
  assert.equal(badShoe.quarantineReason, 'chronic_photo_defects');
  assert.ok(badShoe.qualityScore < 0.25);
});

test('recordVisualCriticOutcome handles approvals, category mismatches, and batch arrays cleanly', async () => {
  const store = new Map();
  await recordVisualCriticOutcome(store, [
    {
      approved: true,
      items: [
        { id: 'top-1', slot: 'top' },
        { id: 'coat-1', slot: 'outerwear' },
        { id: 'bottom-1', slot: 'bottom' },
        { id: 'shoe-1', slot: 'shoes' },
      ],
      editorialScore: 8.8,
    },
    {
      approved: false,
      items: [{ id: 'swim-short-99', slot: 'bottom' }],
      weakestSlot: 'bottom',
      defectCategory: 'category_mismatch',
    },
    {
      approved: false,
      items: [{ id: 'swim-short-99', slot: 'bottom' }],
      weakestSlot: 'bottom',
      defectCategory: 'swimwear_mismatch',
    },
  ]);

  assert.equal(store.get('top-1').approvals, 1);
  assert.equal(store.get('top-1').avgEditorialScore, 8.8);
  assert.ok(store.get('top-1').qualityScore > DEFAULT_NEUTRAL_QUALITY_SCORE);

  assert.equal(store.get('swim-short-99').categoryMismatches, 2);
  assert.equal(store.get('swim-short-99').quarantined, true);
  assert.equal(store.get('swim-short-99').quarantineReason, 'category_mismatch');
});

test('buildQualityLookupMap normalizes Postgres snake_case rows and camelCase objects identically', () => {
  const map = buildQualityLookupMap([
    {
      product_id: 'sku-db-healthy',
      approvals: 8,
      rejections: 1,
      photo_defects: 0,
      category_mismatches: 0,
      editorial_score_sum: 72,
      editorial_score_count: 9,
    },
    {
      productId: 'sku-js-bad',
      approvals: 1,
      rejections: 5,
      photoDefects: 0,
      categoryMismatches: 0,
    },
  ]);

  assert.equal(map.size, 2);
  const healthy = map.get('sku-db-healthy');
  assert.equal(healthy.quarantined, false);
  assert.equal(healthy.avgEditorialScore, 8);
  assert.ok(healthy.qualityScore > 0.72);

  const bad = map.get('sku-js-bad');
  assert.equal(bad.quarantined, true);
  assert.equal(bad.quarantineReason, 'low_wilson_approval');
});

test('createQualityFlywheel works with Supabase-style client adapter and reports aggregate stats', async () => {
  const dbRows = new Map();
  const mockSupabase = {
    from(table) {
      assert.equal(table, 'catalog_visual_quality');
      return {
        select() {
          return {
            async in(col, ids) {
              assert.equal(col, 'product_id');
              return { data: ids.map((id) => dbRows.get(id)).filter(Boolean), error: null };
            },
            then(resolve) {
              resolve({ data: [...dbRows.values()], error: null });
            },
          };
        },
        async upsert(rows) {
          for (const row of rows) dbRows.set(row.product_id, row);
          return { error: null };
        },
      };
    },
  };

  const flywheel = createQualityFlywheel(mockSupabase);

  // Unseen SKU defaults to neutral 0.65
  const unseen = await flywheel.getQuality('unseen-sku');
  assert.equal(unseen.qualityScore, 0.65);
  assert.equal(await flywheel.isQuarantined('unseen-sku'), false);

  await flywheel.recordOutcome({
    approved: true,
    itemIds: ['good-a', 'good-b'],
    editorialScore: 9.0,
  });
  for (let i = 0; i < 5; i++) {
    await flywheel.recordOutcome({
      approved: false,
      itemIds: ['bad-c'],
      editorialScore: 3.0,
    });
  }

  assert.equal(await flywheel.isQuarantined('good-a'), false);
  assert.equal(await flywheel.isQuarantined('bad-c'), true);

  const quarantined = await flywheel.listQuarantined();
  assert.equal(quarantined.length, 1);
  assert.equal(quarantined[0].productId, 'bad-c');

  const summary = await flywheel.stats();
  assert.equal(summary.trackedSkus, 3);
  assert.equal(summary.quarantinedSkus, 1);
  assert.equal(summary.healthySkus, 2);
});
