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
test('a missing discovery RPC fails closed without a legacy catalog scan', async () => {
  // The counted table-scan fallback was removed with the Stylist 3.0
  // retrieval redesign: a missing RPC capability is an explicit retrieval
  // failure, never a silent legacy scan (service header comment).
  await expect(service.loadCatalogDiscovery({ seed: 'late-row-proof', limit: 40 }))
    .rejects.toMatchObject({ code: 'catalog_lookup_unavailable', stage: 'catalog_retrieval' });
  expect(mockFrom).not.toHaveBeenCalled();
});
test('discovery forwards request identity and paging to the RPC and caches stable pages', async () => {
  let call = 0;
  mockRpc.mockImplementation(() => {
    call += 1;
    return {
      abortSignal: async () => ({ data: rows.slice(call === 1 ? 0 : 40, call === 1 ? 40 : 80), error: null }),
    };
  });
  const a = await service.loadCatalogDiscovery({ seed: 'late-row-proof', limit: 40 });
  const b = await service.loadCatalogDiscovery({ seed: 'late-row-proof', limit: 40, page: 1 });
  expect(a.rows).toHaveLength(40);
  expect(b.rows).toHaveLength(40);
  expect(new Set([...a.rows, ...b.rows].map((r) => r.id)).size).toBe(80);
  expect(mockRpc).toHaveBeenCalledWith(
    'catalog_discovery_candidates',
    expect.objectContaining({ p_seed: 'late-row-proof', p_limit: 40, p_offset: 40 }),
  );
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
test('auth errors stay retrieval failures, unsafe provenance is rejected, and cancellation is per reader', async () => {
  mockRpc.mockReturnValueOnce({
    abortSignal: async () => ({ error: { code: '42501' }, data: null }),
  });
  // The client normalizes every RPC failure (including RLS denials) to the
  // client-safe retrieval code; raw database error codes never leak to the
  // discovery surface.
  await expect(service.loadCatalogDiscovery({ seed: 'denied' })).rejects.toMatchObject({
    code: 'catalog_lookup_unavailable',
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
