/** @jest-environment node */
const mockFrom = jest.fn();
jest.mock('../../lib/supabase', () => ({
  supabase: { from: (...args: unknown[]) => mockFrom(...args) },
}));
import { createPriceReadRepository } from '../../src/services/purchase/priceRepository';
import { LocalPriceRepository } from '../../src/services/purchase/priceIntelligence';
import { offer } from './fixtures';
test('remote history is dormant without explicit activation and local observations remain available', async () => {
  const local = new LocalPriceRepository();
  local.observe(offer());
  const repository = createPriceReadRepository(local);
  expect((await repository.read('shoe')).source).toBe('local');
  expect(mockFrom).not.toHaveBeenCalled();
});
test('missing or failed remote schema uses local fallback without writes', async () => {
  const q: any = {};
  for (const method of ['select', 'eq', 'limit']) q[method] = jest.fn(() => q);
  q.abortSignal = jest.fn(async () => ({ data: null, error: { code: '42P01' } }));
  mockFrom.mockReturnValue(q);
  const local = new LocalPriceRepository();
  local.observe(offer());
  const result = await createPriceReadRepository(local, true).read('shoe');
  expect(result.source).toBe('local');
  expect(result.offers).toHaveLength(1);
  expect(q.limit).toHaveBeenCalledWith(20);
});

test('activated read adapter coalesces requests, bounds history and keeps absent cost provenance unknown', async () => {
  const { now } = require('./fixtures');
  const timestamp = new Date(now).toISOString();
  const queries: any[] = [];
  mockFrom.mockImplementation((table: string) => {
    const q: any = {};
    for (const method of ['select', 'eq', 'order', 'limit']) q[method] = jest.fn(() => q);
    q.abortSignal = jest.fn(async () =>
      table === 'shop_offers'
        ? {
            data: [
              {
                id: 'remote-offer',
                variant_id: 'shoe',
                merchant_id: 'merchant',
                url: 'https://merchant.test/product',
                currency: 'USD',
                current_price: '80',
                last_seen_at: timestamp,
                shipping_price: 0,
                tax_price: 0,
                fees: 0,
                component_provenance: {
                  tax: { source: 'merchant', confidence: 'high', observedAt: timestamp },
                },
                shop_merchants: { name: 'Merchant', confidence: 'high' },
              },
            ],
            error: null,
          }
        : {
            data: [
              { offer_id: 'remote-offer', observed_at: timestamp, price: '80', currency: 'USD' },
            ],
            error: null,
          }
    );
    queries.push(q);
    return q;
  });
  const repository = createPriceReadRepository(new LocalPriceRepository(), true);
  const [a, b] = await Promise.all([repository.read('shoe'), repository.read('shoe')]);
  expect(a).toBe(b);
  expect(a.source).toBe('remote');
  expect(a.offers[0].shipping).toBeUndefined();
  expect(a.offers[0].tax!.amount).toBe(0);
  expect(a.history[0].price).toBe(80);
  expect(queries[1].limit).toHaveBeenCalledWith(90);
});
