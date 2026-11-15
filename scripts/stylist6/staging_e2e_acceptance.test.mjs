import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildAcceptanceMatrix,
  captureAcceptanceRecord,
  validateDryMatrix,
} from './staging_e2e_acceptance.mjs';

test('E2E acceptance matrix covers catalog, wardrobe, mixed, anchor, budget, EN/RU/UZ, and both app surfaces', () => {
  const checked = validateDryMatrix({ anchorItemId: 'shop_top-1' });
  assert.equal(checked.length, 8);
  const matrix = buildAcceptanceMatrix({ anchorItemId: 'shop_top-1' });
  const ids = new Set(matrix.map((s) => s.id));
  for (const required of [
    'ai_maker_catalog_en',
    'ai_maker_wardrobe_en',
    'ai_maker_mixed_en',
    'ai_maker_locked_anchor_mixed',
    'ai_maker_budget_uzs',
    'ai_maker_catalog_ru',
    'ai_maker_catalog_uz',
    'daily_ai_outfit_top_level',
  ]) {
    assert.ok(ids.has(required), `Missing scenario ${required}`);
  }
  const daily = checked.find((c) => c.id === 'daily_ai_outfit_top_level');
  assert.equal(daily.normalized.style, 'business_casual');
  assert.equal(daily.normalized.enableVisual, true);
});

test('captureAcceptanceRecord detects engine mismatches, missing anchors, budget overruns, and locale issues', () => {
  const matrix = buildAcceptanceMatrix({ anchorItemId: 'w-top-1', budgetAmountUzs: 500_000 });
  const anchoredScenario = matrix.find((s) => s.id === 'ai_maker_locked_anchor_mixed');
  const ruScenario = matrix.find((s) => s.id === 'ai_maker_catalog_ru');
  const budgetScenario = matrix.find((s) => s.id === 'ai_maker_budget_uzs');

  const validOutfits = (anchorId, price = 100_000, desc = 'Tailored look') => [1, 2, 3].map((n) => ({
    id: `outfit_${n}`,
    selectionSource: 'ai',
    engineVersion: 'stylist-6-cloud-staging-1',
    description: desc,
    fashionDiagnostics: { visualVerdict: 'JUDGED', referenceEvaluationMode: 'product_only' },
    items: [
      { id: anchorId || `top-${n}`, type: 'top', isShopItem: anchorId ? false : true, price, currency: 'UZS', imageUrl: 'https://img.example/t.jpg', shopUrl: 'https://shop.example/t' },
      { id: `bot-${n}`, type: 'bottom', isShopItem: true, price, currency: 'UZS', imageUrl: 'https://img.example/b.jpg', shopUrl: 'https://shop.example/b' },
      { id: `shoe-${n}`, type: 'shoes', isShopItem: true, price, currency: 'UZS', imageUrl: 'https://img.example/s.jpg', shopUrl: 'https://shop.example/s' },
    ],
  }));

  const okAnchor = captureAcceptanceRecord(anchoredScenario, 200, {
    success: true,
    generationVersion: 'stylist-6-cloud-staging-1',
    serverInventory: true,
    language: 'en',
    visualStatus: 'NVIDIA_NOT_RUN',
    outfits: validOutfits('w-top-1'),
  }, { latencyMs: 420, quotaBefore: 0, quotaAfter: 1 });
  assert.equal(okAnchor.pass, true);
  assert.equal(okAnchor.quotaOutcome.delta, 1);

  const missingAnchor = captureAcceptanceRecord(anchoredScenario, 200, {
    success: true,
    generationVersion: 'stylist-6-cloud-staging-1',
    outfits: validOutfits(null),
  });
  assert.equal(missingAnchor.pass, false);
  assert.ok(missingAnchor.failures.some((f) => f.includes('missing_anchor')));

  const badRu = captureAcceptanceRecord(ruScenario, 200, {
    success: true,
    generationVersion: 'stylist-6-cloud-staging-1',
    language: 'ru',
    outfits: validOutfits(null, 100_000, 'English leaked into RU'),
  });
  assert.equal(badRu.pass, false);
  assert.ok(badRu.failures.includes('missing_cyrillic_in_ru_copy'));

  const overBudget = captureAcceptanceRecord(budgetScenario, 200, {
    success: true,
    generationVersion: 'stylist-6-cloud-staging-1',
    language: 'en',
    outfits: validOutfits(null, 250_000, 'Classic work look'),
  });
  assert.equal(overBudget.pass, false);
  assert.ok(overBudget.failures.some((f) => f.includes('budget_exceeded_750000')));
});
