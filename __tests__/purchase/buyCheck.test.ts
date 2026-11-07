import { buyCheck, duplicateRisk, usefulUnlocksV2 } from '../../src/services/purchase/buyCheck';
import { deriveFitProfile } from '../../src/services/purchase/fitEvidence';
import { row, shoe, owned, context, fit, now } from './fixtures';
const varied = [
  ...owned,
  row('shirt-blue', 'Light blue cotton oxford shirt', 'upper_body', 0),
  row('pants-beige', 'Beige chino trousers', 'lower_body', 0),
  row('layer-beige', 'Beige lightweight linen blazer', 'upper_body', 0),
];
test('high wardrobe value and useful size evidence support buying while unknown fit remains explicit', () => {
  const candidate = { ...shoe, availableSizes: ['42'] },
    profile = deriveFitProfile([fit('r1'), fit('r2'), fit('r3')], now);
  const result = buyCheck(candidate, varied, {
    ...context,
    budget: undefined,
    fitProfile: profile,
  });
  expect(result.wardrobe.usefulOutfitUnlocks.total).toBeGreaterThanOrEqual(3);
  expect(result.fit.recommendedSize).toBe('42');
  expect(result.verdict).toBe('BUY');
  const unknown = buyCheck(candidate, varied, context);
  expect(unknown.verdict).toBe('CONSIDER');
  expect(unknown.fit.confidence).toBe('low');
  expect(unknown.purchase.priceAssessment).toBeUndefined();
});
test('high duplicate and low incremental value support skipping, while intentional multiples remain a user choice', () => {
  const duplicate = row('owned-shoe', 'Regular brown leather penny loafers');
  const result = buyCheck(shoe, [...owned, duplicate], context);
  expect(result.wardrobe.duplicateRisk.level).toBe('high');
  expect(result.verdict).toBe('SKIP');
  expect(
    buyCheck(shoe, [...owned, duplicate], { ...context, allowMultiples: true }).verdict
  ).not.toBe('SKIP');
  expect(duplicateRisk(shoe, [duplicate]).substitutes[0].evidence).toContain('Same garment role');
});
test('insufficient wardrobe/fit evidence, explicit budget conflict and known fit with no wardrobe value stay distinguishable', () => {
  expect(buyCheck(shoe, [], {}).verdict).toBe('INSUFFICIENT_EVIDENCE');
  expect(
    buyCheck(shoe, owned, { ...context, budget: { amount: 1, currency: 'USD' } }).verdict
  ).toBe('SKIP');
  const profile = deriveFitProfile([fit('exact', { productId: shoe.id })], now);
  const low = buyCheck({ ...shoe, availableSizes: ['42'] }, [], { fitProfile: profile });
  expect(low.fit.confidence).toBe('high');
  expect(low.verdict).toBe('CONSIDER');
});
test('unlock v2 removes trivial color substitutions and rejects climate/activity-invalid looks within bounded work', () => {
  const navy = { ...shoe, id: 'navy', name: 'Navy leather penny loafers' };
  const black = { ...shoe, id: 'black', name: 'Black leather penny loafers' };
  expect(usefulUnlocksV2(navy, [...owned, black], context).total).toBe(0);
  expect(usefulUnlocksV2(shoe, varied, { ...context, occasion: 'Workout' }).total).toBe(0);
  expect(usefulUnlocksV2(shoe, varied, { ...context, weather: { temp: -15 } }).total).toBe(0);
  const bounded = usefulUnlocksV2(shoe, varied, {
    ...context,
    contexts: Array.from({ length: 10 }, (_, i) => ({ id: String(i), occasion: 'Work' })),
  });
  expect(bounded.evaluations).toBeLessThanOrEqual(416);
  expect(bounded.truncated).toBe(true);
});

test('known checkout charges participate in budget decisions without assuming absent charges are zero', () => {
  const { offer } = require('./fixtures');
  const { knownCharge } = require('../../src/services/purchase/priceIntelligence');
  const stamp = new Date().toISOString(),
    charges = {
      shipping: knownCharge(30, 'USD', 'checkout', stamp),
      tax: knownCharge(0, 'USD', 'checkout', stamp),
      fees: knownCharge(0, 'USD', 'checkout', stamp),
    };
  const result = buyCheck(
    shoe,
    owned,
    { ...context, budget: { amount: 110, currency: 'USD' } },
    { offers: [offer('known', { ...charges, lastSeenAt: stamp })] }
  );
  expect(result.verdict).toBe('SKIP');
  expect(result.purchase).toMatchObject({ withinBudget: false, budgetBasis: 'checkout' });
  const unknown = buyCheck(shoe, owned, context, {
    offers: [offer('unknown', { lastSeenAt: stamp })],
  });
  expect(unknown.purchase.budgetBasis).toBe('item');
  expect(unknown.purchase.offers![0].truePrice.total).toBeUndefined();
});
test('one validated look is not multiplied into multiple outfits merely by attaching multiple contexts', () => {
  const result = usefulUnlocksV2(shoe, owned, {
    ...context,
    contexts: [
      { id: 'work1', occasion: 'Work' },
      { id: 'work2', occasion: 'Work' },
    ],
  });
  expect(result.total).toBe(1);
  expect(result.contextBreakdown).toEqual({ work1: 1, work2: 1 });
});

test('high historical prices and known unavailable size are explicit purchase tradeoffs', () => {
  const { offer } = require('./fixtures'),
    stamp = new Date().toISOString();
  const varied = [
    ...owned,
    row('blue', 'Light blue cotton oxford shirt', 'upper_body', 0),
    row('beige', 'Beige chino trousers', 'lower_body', 0),
  ];
  const profile = deriveFitProfile([fit('r1'), fit('r2'), fit('r3')], now);
  const history = Array.from({ length: 16 }, (_, i) => ({
    offerId: 'market',
    currency: 'USD',
    price: 100,
    observedAt: new Date(Date.now() - (80 - i * 5) * 86400000).toISOString(),
  }));
  const expensive = buyCheck(
    { ...shoe, price: 140, availableSizes: ['42'] },
    varied,
    { ...context, fitProfile: profile },
    { offers: [offer('market', { currentPrice: 140, lastSeenAt: stamp })], history }
  );
  expect(expensive.verdict).not.toBe('BUY');
  expect(expensive.explanation.join(' ')).toContain('high relative');
  const unavailable = buyCheck(
    { ...shoe, availableSizes: ['42'] },
    varied,
    { ...context, fitProfile: profile, selectedSize: '42' },
    { offers: [offer('market', { lastSeenAt: stamp, sizes: { '42': false } })] }
  );
  expect(unavailable.verdict).not.toBe('BUY');
  expect(unavailable.explanation.join(' ')).toContain('out of stock');
});

test('multiple real offers are compared before the budget verdict, not by arbitrary input order', () => {
  const { offer } = require('./fixtures'),
    { knownCharge } = require('../../src/services/purchase/priceIntelligence');
  const stamp = new Date().toISOString();
  const zero = knownCharge(0, 'USD', 'merchant checkout', stamp),
    extras = { shipping: zero, tax: zero, fees: zero, inStock: true, lastSeenAt: stamp };
  const result = buyCheck(
    { ...shoe, price: 200 },
    owned,
    { ...context, budget: { amount: 90, currency: 'USD' } },
    {
      offers: [
        offer('expensive', { ...extras, currentPrice: 200 }),
        offer('affordable', { ...extras, currentPrice: 80 }),
      ],
    }
  );
  expect(result.purchase.withinBudget).toBe(true);
  expect(result.verdict).not.toBe('SKIP');
  expect(result.purchase.offers![0].offer.id).toBe('affordable');
});
