/** @jest-environment node */

const mockFrom = jest.fn(),
  mockRpc = jest.fn();
jest.mock('../../lib/supabase', () => ({
  supabase: {
    from: (...args: unknown[]) => mockFrom(...args),
    rpc: (...args: unknown[]) => mockRpc(...args),
  },
}));

// The inventory read goes through the shared bounded discovery RPC (four
// balanced slot windows), never a legacy table scan — a missing RPC
// capability is an explicit retrieval failure, not a silent fallback.
const row = { id: 'live-shirt', name: 'Cotton shirt', description: 'product-only fixture', source: 'product-only', garment_type: 'upper_body', image_url: 'https://shop.example/shirt.jpg' };
const response = { data: [row], error: null };
let service: typeof import('../../src/services/stylistCatalogInventory');

function rpcPages(result: Promise<unknown>) {
  mockRpc.mockImplementation(() => ({ abortSignal: () => result }));
}

beforeEach(() => {
  jest.resetModules();
  mockFrom.mockReset();
  mockRpc.mockReset();
  service = require('../../src/services/stylistCatalogInventory');
});
afterEach(() => { jest.restoreAllMocks(); });

test('Home prewarming and screen mounting share one balanced fetch and a synchronous cache', async () => {
  let finish!: (value: typeof response) => void;
  rpcPages(new Promise(resolve => { finish = resolve; }));
  expect(service.getCachedStylistCatalogInventory()).toBeUndefined();
  const prewarm = service.loadStylistCatalogInventory();
  const screen = service.loadStylistCatalogInventory();
  expect(mockRpc).toHaveBeenCalledTimes(4);
  finish(response);
  const rows = await prewarm;
  expect(await screen).toBe(rows);
  expect(rows).toEqual([row]);
  expect(service.getCachedStylistCatalogInventory()).toBe(rows);
  expect(await service.loadStylistCatalogInventory()).toBe(rows);
  expect(mockRpc).toHaveBeenCalledTimes(4);
  expect(mockFrom).not.toHaveBeenCalled();
  expect(mockRpc).toHaveBeenCalledWith(
    'catalog_discovery_candidates',
    expect.objectContaining({ p_slot: 'top', p_limit: 100 }),
  );
});

test('expired inventory is hidden until live verification finishes again', async () => {
  const now = jest.spyOn(Date, 'now').mockReturnValue(1000);
  rpcPages(Promise.resolve(response));
  await service.loadStylistCatalogInventory();
  now.mockReturnValue(1000 + 5 * 60 * 1000);
  expect(service.getCachedStylistCatalogInventory()).toBeUndefined();
  await service.loadStylistCatalogInventory();
  expect(mockRpc).toHaveBeenCalledTimes(8);
});

test('closing one reader does not cancel a shared prewarm or poison its cache', async () => {
  let finish!: (value: typeof response) => void;
  rpcPages(new Promise(resolve => { finish = resolve; }));
  const controller = new AbortController();
  const prewarm = service.loadStylistCatalogInventory();
  const screen = service.loadStylistCatalogInventory(controller.signal);
  controller.abort();
  await expect(screen).rejects.toMatchObject({ name: 'AbortError' });
  finish(response);
  expect(await prewarm).toEqual([row]);
  expect(service.getCachedStylistCatalogInventory()).toEqual([row]);
  expect(mockRpc).toHaveBeenCalledTimes(4);
});

test('failed checks are retried and do not become verified empty inventory', async () => {
  rpcPages(Promise.resolve({ data: null, error: { code: 'PGRST202' } }));
  await expect(service.loadStylistCatalogInventory()).rejects.toMatchObject({ code: 'catalog_lookup_unavailable' });
  expect(service.getCachedStylistCatalogInventory()).toBeUndefined();
  rpcPages(Promise.resolve(response));
  expect(await service.loadStylistCatalogInventory()).toEqual([row]);
  expect(mockRpc).toHaveBeenCalledTimes(8);
});

test('verified empty inventory stays empty until an explicit refresh checks again', async () => {
  rpcPages(Promise.resolve({ data: [], error: null }));
  await service.loadStylistCatalogInventory();
  expect(service.getCachedStylistCatalogInventory()).toEqual([]);
  // A verified-empty discovery window is still recent evidence: the plain
  // reload serves it for the TTL instead of re-asking the RPC.
  rpcPages(Promise.resolve(response));
  expect(await service.loadStylistCatalogInventory()).toEqual([]);
  // An explicit refresh rechecks the live discovery and picks up new rows.
  expect(await service.loadStylistCatalogInventory(undefined, true)).toEqual([row]);
  expect(mockRpc).toHaveBeenCalledTimes(8);
});

test('an explicit refresh rechecks the underlying discovery instead of serving its cached window', async () => {
  rpcPages(Promise.resolve(response));
  await service.loadStylistCatalogInventory();
  const replacement = { ...row, id: 'replacement' };
  rpcPages(Promise.resolve({ data: [replacement], error: null }));
  expect(await service.loadStylistCatalogInventory(undefined, true)).toEqual([replacement]);
  expect(mockRpc).toHaveBeenCalledTimes(8);
});
