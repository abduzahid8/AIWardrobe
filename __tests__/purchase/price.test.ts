import {
  truePrice,
  assessPrice,
  knownCharge,
  LocalPriceRepository,
  compareMerchantOffers,
  sameOfferProduct,
  evaluatePriceWatch,
} from '../../src/services/purchase/priceIntelligence';
import { offer, now } from './fixtures';
const charge = (amount: number) =>
  knownCharge(amount, 'USD', 'merchant checkout', new Date(now).toISOString());
const history = Array.from({ length: 16 }, (_, i) => ({
  offerId: 'offer',
  currency: 'USD',
  price: 100,
  observedAt: new Date(now - (80 - i * 5) * 86400000).toISOString(),
}));
test('price history states distinguish insufficient, below median, above history, current minimum and waiting without forecasts', () => {
  expect(assessPrice(offer(), [], { now }).state).toBe('INSUFFICIENT_HISTORY');
  expect(
    assessPrice(offer('offer', { currentPrice: 80 }), history, { now, windowDays: 90 })
  ).toMatchObject({ state: 'GOOD_PRICE', median: 100, belowMedianPercent: 20 });
  expect(assessPrice(offer('offer', { currentPrice: 125 }), history, { now }).state).toBe(
    'HIGH_PRICE'
  );
  expect(
    assessPrice(offer('offer', { currentPrice: 125 }), history, { now, waitIfHigh: true }).state
  ).toBe('WAIT');
  expect(assessPrice(offer('offer', { currentPrice: 100 }), history, { now }).minimum).toBe(100);
  expect(assessPrice(offer('offer', { currentPrice: 105 }), history, { now }).state).toBe(
    'NORMAL_PRICE'
  );
});
test('unknown shipping/tax/fees remain unknown while confirmed zero fees and discounts compose totals', () => {
  expect(truePrice(offer(), now)).toMatchObject({
    total: undefined,
    knownSubtotal: 100,
    unknownComponents: ['shipping', 'tax', 'fees'],
  });
  expect(
    truePrice(
      offer('offer', { shipping: charge(10), tax: charge(5), fees: charge(0), coupon: charge(20) }),
      now
    ).total
  ).toBe(95);
  expect(
    truePrice(offer('offer', { shipping: { ...charge(0), currency: 'MYR' } }), now)
      .unknownComponents
  ).toContain('shipping');
});
test('same-day same-price observations do not grow history and stale snapshots cannot overwrite latest prices', () => {
  const repository = new LocalPriceRepository();
  repository.observe(offer());
  repository.observe(offer('offer', { lastSeenAt: new Date(now + 1000).toISOString() }));
  expect(repository.getHistory('offer')).toHaveLength(1);
  repository.observe(
    offer('offer', { currentPrice: 80, lastSeenAt: new Date(now + 2000).toISOString() })
  );
  expect(repository.getHistory('offer')).toHaveLength(2);
  repository.observe(
    offer('offer', { currentPrice: 99, lastSeenAt: new Date(now - 1000).toISOString() })
  );
  expect(repository.getOffers('shoe')[0].currentPrice).toBe(80);
});
test('multi-merchant comparison requires authoritative relation and real comparable totals', () => {
  const reference = offer('a', {
    shipping: charge(0),
    tax: charge(0),
    fees: charge(0),
    inStock: true,
  });
  const cheaper = offer('b', {
    merchant: 'Other',
    currentPrice: 80,
    shipping: charge(5),
    tax: charge(0),
    fees: charge(0),
    inStock: true,
  });
  const uncertain = offer('c', { currentPrice: 50 });
  const unrelated = offer('d', { variantId: 'different' });
  expect(sameOfferProduct(reference, unrelated)).toBe(false);
  const result = compareMerchantOffers(reference, [reference, uncertain, unrelated, cheaper], {
    now,
    currency: 'USD',
  });
  expect(result).toHaveLength(3);
  expect(result[0].offer.id).toBe('b');
  expect(result[2].comparableTotal).toBeUndefined();
  const gtin = { kind: 'gtin' as const, value: '1234567890123' };
  expect(
    sameOfferProduct(
      { ...reference, canonicalIdentifier: gtin },
      { ...unrelated, canonicalIdentifier: gtin }
    )
  ).toBe(true);
});
test('price watches are local contracts with fresh matching selected-size stock only', () => {
  const watch = {
    id: 'watch',
    userId: 'u',
    productId: 'shoe',
    currency: 'USD',
    threshold: 90,
    status: 'active' as const,
    createdAt: new Date(now).toISOString(),
    selectedSize: '42',
  };
  expect(
    evaluatePriceWatch(watch, [offer('a', { currentPrice: 80, inStock: true })], now).triggered
  ).toBe(false);
  expect(
    evaluatePriceWatch(watch, [offer('a', { currentPrice: 80, sizes: { '42': true } })], now)
      .triggered
  ).toBe(true);
  expect(
    evaluatePriceWatch(
      { ...watch, status: 'paused' },
      [offer('a', { currentPrice: 80, sizes: { '42': true } })],
      now
    ).triggered
  ).toBe(false);
});

test('real daily observations of a stable price provide history without redundant same-day rows', () => {
  const repository = new LocalPriceRepository();
  for (let day = 40; day >= 0; day--)
    repository.observe(
      offer('stable', { lastSeenAt: new Date(now - day * 86400000).toISOString() })
    );
  repository.observe(offer('stable', { lastSeenAt: new Date(now + 1000).toISOString() }));
  expect(repository.getHistory('stable')).toHaveLength(41);
  const assessment = assessPrice(
    offer('stable', { currentPrice: 80 }),
    repository.getHistory('stable'),
    { now }
  );
  expect(assessment.state).toBe('GOOD_PRICE');
  expect(assessment.median).toBe(100);
  const intraday = [
    ...history,
    {
      offerId: 'offer',
      currency: 'USD',
      price: 50,
      observedAt: new Date(now - 3600000).toISOString(),
    },
    {
      offerId: 'offer',
      currency: 'USD',
      price: 100,
      observedAt: new Date(now - 1800000).toISOString(),
    },
  ];
  expect(assessPrice(offer(), intraday, { now }).minimum).toBe(50);
});

test('stale merchant quotes cannot be labelled a current good price', () => {
  const stale = offer('offer', {
    currentPrice: 80,
    lastSeenAt: new Date(now - 10 * 86400000).toISOString(),
  });
  expect(assessPrice(stale, history, { now }).state).toBe('INSUFFICIENT_HISTORY');
});

test('a cached/manual estimate cannot masquerade as fresh merchant history', () => {
  const estimate = offer('estimate', { quoteKind: 'estimate' });
  expect(assessPrice(estimate, history, { now })).toMatchObject({
    state: 'INSUFFICIENT_HISTORY',
    quoteFresh: false,
  });
  expect(() => new LocalPriceRepository().observe(estimate)).toThrow('not a market observation');
});
