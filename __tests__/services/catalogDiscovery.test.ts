/** @jest-environment node */
const mockFrom = jest.fn(),
  mockRpc = jest.fn();
jest.mock('../../lib/supabase', () => ({
  supabase: {
    from: (...args: unknown[]) => mockFrom(...args),
    rpc: (...args: unknown[]) => mockRpc(...args),
  },
}));
let service: typeof import('../../src/services/catalogDiscovery');
const rows = Array.from({ length: 900 }, (_, i) => ({
  id: `id-${i}`,
  name: `Model ${i}`,
  brand: `Brand ${i % 9}`,
  description: 'product-only',
  source: 'product-only',
  garment_type: i % 3 ? 'upper_body' : 'shoes',
}));
const windows: number[][] = [];
function fallback() {
  let count = false,
    start = 0,
    end = 99;
  const q: any = {};
  q.select = jest.fn((_fields: string, opts?: { head?: boolean }) => {
    count = !!opts?.head;
    return q;
  });
  for (const method of ['eq', 'or', 'order', 'lte']) q[method] = jest.fn(() => q);
  q.range = jest.fn((a: number, b: number) => {
    start = a;
    end = b;
    windows.push([a, b]);
    return q;
  });
  q.abortSignal = jest.fn(async () =>
    count ? { count: rows.length, error: null } : { data: rows.slice(start, end + 1), error: null }
  );
  return q;
}
beforeEach(() => {
  jest.resetModules();
  mockFrom.mockReset();
  mockRpc.mockReset();
  windows.length = 0;
  service = require('../../src/services/catalogDiscovery');
  mockFrom.mockImplementation(fallback);
  mockRpc.mockReturnValue({
    abortSignal: async () => ({ data: null, error: { code: 'PGRST202' } }),
  });
});
test('counted discovery reaches interior and later rows with bounded windows and stable non-overlapping pagination', async () => {
  const a = await service.loadCatalogDiscovery({ seed: 'late-row-proof', limit: 40 });
  const b = await service.loadCatalogDiscovery({ seed: 'late-row-proof', limit: 40, page: 1 });
  expect(a.rows).toHaveLength(40);
  expect(b.rows).toHaveLength(40);
  expect(new Set([...a.rows, ...b.rows].map((r) => r.id)).size).toBe(80);
  const varied = await Promise.all(
    Array.from({ length: 10 }, (_, i) =>
      service.loadCatalogDiscovery({ seed: `discovery-${i}`, limit: 40 })
    )
  );
  expect(varied.flatMap((page) => page.rows).some((r) => Number(r.id.slice(3)) > 600)).toBe(true);
  expect(windows.every(([start, end]) => end - start < 40)).toBe(true);
  expect(new Set(a.rows.map((r) => r.brand)).size).toBe(9);
  expect(a.rows.some((r) => r.garment_type === 'shoes')).toBe(true);
  expect(mockRpc).toHaveBeenCalledTimes(1);
});
test('full bounded traversal wraps without dropping or duplicating variants; different seeds expose different rows', async () => {
  const pages = await Promise.all(
    Array.from({ length: 9 }, (_, page) =>
      service.loadCatalogDiscovery({ seed: 'traversal', limit: 100, page })
    )
  );
  expect(new Set(pages.flatMap((p) => p.rows.map((r) => r.id))).size).toBe(900);
  expect(pages[8].hasMore).toBe(false);
  expect(
    (await service.loadCatalogDiscovery({ seed: 'different', limit: 100 })).rows.map((r) => r.id)
  ).not.toEqual(pages[0].rows.map((r) => r.id));
});
test('balanced RPC preserves filters and concrete IDs; concurrent readers share the request', async () => {
  mockRpc.mockReturnValue({
    abortSignal: async () => ({ data: rows.slice(750, 800), error: null }),
  });
  const query = {
    seed: 'rpc',
    slot: 'shoes' as const,
    limit: 50,
    currency: 'USD',
    maxPrice: 80,
    search: 'loafer',
  };
  const [a, b] = await Promise.all([
    service.loadCatalogDiscovery(query),
    service.loadCatalogDiscovery(query),
  ]);
  expect(a).toBe(b);
  expect(a.strategy).toBe('balanced-rpc');
  expect(a.rows[0].id).toBe('id-750');
  expect(mockRpc).toHaveBeenCalledWith(
    'catalog_discovery_candidates',
    expect.objectContaining({
      p_slot: 'shoes',
      p_currency: 'USD',
      p_max_price: 80,
      p_search: 'loafer',
      p_limit: 50,
    })
  );
  expect(mockFrom).not.toHaveBeenCalled();
});
test('auth errors remain errors, unsafe provenance is rejected, and cancellation is per reader', async () => {
  mockRpc.mockReturnValueOnce({
    abortSignal: async () => ({ error: { code: '42501' }, data: null }),
  });
  await expect(service.loadCatalogDiscovery({ seed: 'denied' })).rejects.toMatchObject({
    code: '42501',
  });
  mockRpc.mockReturnValue({
    abortSignal: async () => ({
      data: [{ ...rows[0], description: 'photo shows a model', source: 'product-only' }],
      error: null,
    }),
  });
  expect((await service.loadCatalogDiscovery({ seed: 'provenance' })).rows).toEqual([]);
  const controller = new AbortController();
  controller.abort();
  await expect(service.loadCatalogDiscovery({}, controller.signal)).rejects.toMatchObject({
    name: 'AbortError',
  });
});

test('cancelling a reader leaves the shared transport and other reader alive', async () => {
  let finish!: (value: unknown) => void;
  let transport: AbortSignal | undefined;
  mockRpc.mockReturnValue({
    abortSignal: (signal: AbortSignal) => {
      transport = signal;
      return new Promise((resolve) => {
        finish = resolve;
      });
    },
  });
  const query = { seed: 'shared-cancel', limit: 10 };
  const shared = service.loadCatalogDiscovery(query);
  const controller = new AbortController();
  const reader = service.loadCatalogDiscovery(query, controller.signal);
  controller.abort();
  await expect(reader).rejects.toMatchObject({ name: 'AbortError' });
  expect(transport?.aborted).toBe(false);
  finish({ data: rows.slice(0, 10), error: null });
  expect((await shared).rows).toHaveLength(10);
  expect(mockRpc).toHaveBeenCalledTimes(1);
});
