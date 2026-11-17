import type { ShopCatalogItem } from '../../src/types/shop';
import type { FitEvidence, MerchantOffer, PurchaseContext } from '../../src/types/purchase';
export const now = Date.parse('2026-11-02T12:00:00Z');
// Date-sensitive evidence/history fixtures keep a fixed wall clock. Real timers
// and performance clocks remain active for the rendered async flows/benchmarks.
beforeAll(() =>
  jest.useFakeTimers({
    now,
    doNotFake: [
      'setTimeout',
      'clearTimeout',
      'setInterval',
      'clearInterval',
      'setImmediate',
      'clearImmediate',
      'nextTick',
      'queueMicrotask',
      'performance',
      'hrtime',
      'requestAnimationFrame',
      'cancelAnimationFrame',
      'requestIdleCallback',
      'cancelIdleCallback',
    ],
  })
);
afterAll(() => jest.useRealTimers());
export const row = (
  id: string,
  name: string,
  type: ShopCatalogItem['garmentType'] = 'shoes',
  price = 100
): ShopCatalogItem => ({
  id,
  name,
  brand: 'Brand',
  garmentType: type,
  price,
  currency: 'USD',
  imageUrl: `https://test.invalid/${id}.jpg`,
  sourceUrl: `https://test.invalid/products/${id}`,
  source: 'product-only',
});
export const shoe = row('shoe', 'Brown leather penny loafers');
export const owned = [
  row('jacket', 'Navy lightweight cotton blazer', 'upper_body', 0),
  row('shirt', 'White cotton oxford shirt', 'upper_body', 0),
  row('pants', 'Navy chino trousers', 'lower_body', 0),
];
export const context: PurchaseContext = {
  style: 'classic',
  occasion: 'Work',
  weather: { temp: 20 },
  budget: { amount: 500, currency: 'USD' },
};
export const fit = (id: string, patch: Partial<FitEvidence> = {}): FitEvidence => ({
  id,
  userId: 'user-a',
  productId: id,
  brand: 'Brand',
  category: 'shoes',
  sizeLabel: '42',
  source: 'owned',
  overallFit: 'perfect',
  confidence: 0.9,
  kept: true,
  createdAt: new Date(now).toISOString(),
  ...patch,
});
export const offer = (id = 'offer', patch: Partial<MerchantOffer> = {}): MerchantOffer => ({
  id,
  variantId: 'shoe',
  merchant: 'Merchant',
  url: 'https://test.invalid/products/shoe',
  currency: 'USD',
  currentPrice: 100,
  lastSeenAt: new Date(now).toISOString(),
  merchantConfidence: 'medium',
  ...patch,
});
