import {
  findBetterV2,
  maximumWardrobeGain,
  sameLookLessMoney,
  upgradeThisLook,
  detectMissingPieces,
} from '../../src/services/purchase/wardrobeTools';
import { deriveFitProfile } from '../../src/services/purchase/fitEvidence';
import { row, shoe, owned, context, fit, now } from './fixtures';
const varied = [
  ...owned,
  row('shirt2', 'Light blue cotton oxford shirt', 'upper_body', 0),
  row('pants2', 'Beige chino trousers', 'lower_body', 0),
  row('layer2', 'Beige lightweight linen blazer', 'upper_body', 0),
];
test('Find Better provides mode-specific rankings, evidence and tradeoffs instead of one shared score', () => {
  const cheaper = row('cheap', 'Brown leather penny loafers', 'shoes', 60),
    costly = row('costly', 'Brown leather penny loafers', 'shoes', 600),
    wrong = row('wrong', 'White neon athletic running sneakers', 'shoes', 20);
  const result = findBetterV2(shoe, [costly, wrong, cheaper], varied, 'cheaper', context);
  expect(result.map((r) => r.item.id)).toEqual(['cheap']);
  expect(result[0].why.join(' ')).toContain('40.00 USD');
  expect(result[0].tradeoffs).toContain('Fit evidence is limited');
  expect(findBetterV2(shoe, [cheaper], varied, 'fit', context)).toEqual([]);
  const profile = deriveFitProfile([fit('one', { productId: cheaper.id })], now);
  expect(
    findBetterV2(shoe, [{ ...cheaper, availableSizes: ['42'] }], varied, 'fit', {
      ...context,
      fitProfile: profile,
    })
  ).toHaveLength(1);
  expect(findBetterV2(shoe, [], varied, 'overall', context)).toEqual([]);
});
test('budget-first gain prioritizes an actual missing slot, reports uncertain fit, and respects native caps', () => {
  const top = row('extra-top', 'Another white cotton oxford shirt', 'upper_body', 60);
  const result = maximumWardrobeGain(
    [top, shoe, { ...shoe, id: 'over-budget', price: 1000 }],
    varied,
    context
  );
  expect(result.candidates[0].candidate.id).toBe(shoe.id);
  expect(result.candidates[0].fit.confidence).toBe('low');
  expect(result.candidates.some((r) => r.candidate.price > 500)).toBe(false);
  expect(maximumWardrobeGain([shoe], varied, {}).candidates).toEqual([]);
});
test('Same Look Less Money changes a bounded subset, preserves anchors/style/palette and reaches a budget', () => {
  const outfit = owned
    .map((item) => ({ ...item, price: 100, currency: 'USD' }))
    .concat({ ...shoe, currency: 'USD' });
  const cheap = row('cheap', 'Brown leather penny loafers', 'shoes', 60),
    bad = row('bad', 'White athletic running sneakers', 'shoes', 1);
  const result = sameLookLessMoney(
    outfit,
    [cheap, bad],
    [],
    { ...context, lockedIds: ['shirt'] },
    { budget: 380 }
  );
  expect(result.savings).toBe(40);
  expect(result.total).toBe(360);
  expect(result.items.find((item) => item.id === 'shirt')).toBeDefined();
  expect(result.replacements).toHaveLength(1);
  expect(sameLookLessMoney(outfit, [bad], [], context).replacements).toEqual([]);
  expect(
    sameLookLessMoney(outfit, [cheap], [], { ...context, lockedIds: ['shoe'] }).replacements
  ).toEqual([]);
  expect(sameLookLessMoney(outfit, [cheap], [], context, { budget: 100 }).replacements).toEqual([]);
});
test('Upgrade chooses evidence-based single replacements and gap detection returns validated counts only', () => {
  const outfit = [...owned, shoe].map((item) => ({ ...item, price: 100, currency: 'USD' }));
  const alternatives = [
    row('blue', 'Light blue cotton oxford shirt', 'upper_body', 60),
    row('cheap', 'Brown leather penny loafers', 'shoes', 60),
  ];
  const upgrade = upgradeThisLook(outfit, alternatives, varied, context);
  expect(upgrade.length).toBeLessThanOrEqual(3);
  for (const result of upgrade) {
    expect(result.from.id).not.toBe(result.to.id);
    expect(result.evidence).toHaveLength(3);
  }
  const missing = detectMissingPieces(varied, [shoe], context);
  expect(missing.bottlenecks).toContain('shoes');
  expect(missing.suggestions[0].unlocks.total).toBeGreaterThan(0);
  expect(
    findBetterV2(shoe, alternatives, varied, 'overall', { ...context, sourceMode: 'wardrobe' })
  ).toEqual([]);
});

test('wardrobe, lowest-duplicate, missing-piece and closest-style modes have distinct supported outcomes', () => {
  const substitute = row('owned-shoe', 'Regular brown leather penny loafers');
  const derby = row('derby', 'Brown leather derby shoes', 'shoes', 80),
    near = row('near', 'Brown leather penny loafer shoes', 'shoes', 80);
  expect(
    findBetterV2(shoe, [derby], [...varied, substitute], 'lowest-duplicate', context)[0].item.id
  ).toBe('derby');
  expect(
    findBetterV2(shoe, [derby], [...varied, substitute], 'wardrobe', context)[0].analysis.wardrobe
      .usefulOutfitUnlocks.total
  ).toBeGreaterThan(0);
  expect(
    findBetterV2(shoe, [derby], varied, 'missing-piece', context)[0].analysis.wardrobe
      .missingPieceValue
  ).toBe('high');
  expect(findBetterV2(shoe, [derby, near], varied, 'closest-style', context)[0].item.id).toBe(
    'near'
  );
});

test('a cheaper single-piece upgrade can win through stronger fit evidence rather than higher price', () => {
  const newShoe = {
    ...row('better-fit', 'Brown leather derby shoes', 'shoes', 70),
    availableSizes: ['42'],
  };
  const profile = deriveFitProfile([fit('exact-new', { productId: newShoe.id })], now);
  const outfit = [...owned, { ...shoe, price: 200 }];
  const result = upgradeThisLook(outfit, [newShoe], varied, { ...context, fitProfile: profile });
  expect(result[0].from.id).toBe(shoe.id);
  expect(result[0].to.id).toBe('better-fit');
  expect(result[0].priceDelta).toBe(-130);
  const saved = sameLookLessMoney(
    [...owned.map((item) => ({ ...item, price: 100, currency: 'USD' })), shoe],
    [row('cheaper', 'Brown leather penny loafers', 'shoes', 60)],
    [],
    context
  );
  expect(saved.pieceRoles).toHaveLength(4);
  expect(saved.confidence).toBe('low');
});

test('a native MYR budget ranks actual gain without inventing cross-currency rates', () => {
  const loafers = { ...shoe, price: 430, currency: 'MYR' },
    shirt = {
      ...row('extra-shirt', 'Another white cotton oxford shirt', 'upper_body', 220),
      currency: 'MYR',
    };
  const result = maximumWardrobeGain([shirt, loafers], varied, {
    ...context,
    budget: { amount: 500, currency: 'MYR' },
  });
  expect(result.candidates[0].candidate.id).toBe(shoe.id);
  expect(result.candidates[0].purchase.withinBudget).toBe(true);
});

test('gain ranking abstains without validated gains and contextual gaps include more than the first occasion', () => {
  expect(maximumWardrobeGain([shoe], [], context).candidates).toEqual([]);
  const athletic = row('running', 'Athletic running performance sneakers', 'shoes', 0);
  const wardrobe = [...varied, athletic];
  const gaps = detectMissingPieces(wardrobe, [shoe], {
    ...context,
    occasion: 'Casual',
    contexts: [
      { id: 'work', occasion: 'Work' },
      { id: 'casual', occasion: 'Casual' },
    ],
  });
  expect(gaps.contextCoverage).toHaveLength(2);
  expect(gaps.missingFrequency.shoes).toBeGreaterThan(0);
});
