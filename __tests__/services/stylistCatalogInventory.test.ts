/** @jest-environment node */

const mockFrom = jest.fn();
jest.mock('../../lib/supabase', () => ({ supabase: { from: (...args: unknown[]) => mockFrom(...args) } }));

const row = { id: 'live-shirt', name: 'Cotton shirt', garment_type: 'upper_body', image_url: 'https://shop.example/shirt.jpg' };
const response = { data: [row], error: null };
let service: typeof import('../../src/services/stylistCatalogInventory');

function pages(result: Promise<unknown>) {
  mockFrom.mockImplementation(() => {
    const query: Record<string, jest.Mock> = {};
    for (const method of ['select', 'eq', 'or', 'order', 'limit']) query[method] = jest.fn(() => query);
    query.abortSignal = jest.fn(() => result);
    return query;
  });
}

beforeEach(() => {
  jest.resetModules();
  mockFrom.mockReset();
  service = require('../../src/services/stylistCatalogInventory');
});
afterEach(() => { jest.restoreAllMocks(); });

test('Home prewarming and screen mounting share one balanced fetch and a synchronous cache', async () => {
  let finish!: (value: typeof response) => void;
  pages(new Promise(resolve => { finish = resolve; }));
  expect(service.getCachedStylistCatalogInventory()).toBeUndefined();
  const prewarm = service.loadStylistCatalogInventory();
  const screen = service.loadStylistCatalogInventory();
  expect(mockFrom).toHaveBeenCalledTimes(4);
  finish(response);
  const rows = await prewarm;
  expect(await screen).toBe(rows);
  expect(rows).toEqual([row]);
  expect(service.getCachedStylistCatalogInventory()).toBe(rows);
  expect(await service.loadStylistCatalogInventory()).toBe(rows);
  expect(mockFrom).toHaveBeenCalledTimes(4);
});

test('expired inventory is hidden until live verification finishes again', async () => {
  const now = jest.spyOn(Date, 'now').mockReturnValue(1000);
  pages(Promise.resolve(response));
  await service.loadStylistCatalogInventory();
  now.mockReturnValue(1000 + 5 * 60 * 1000);
  expect(service.getCachedStylistCatalogInventory()).toBeUndefined();
  await service.loadStylistCatalogInventory();
  expect(mockFrom).toHaveBeenCalledTimes(8);
});

test('closing one reader does not cancel a shared prewarm or poison its cache', async () => {
  let finish!: (value: typeof response) => void;
  pages(new Promise(resolve => { finish = resolve; }));
  const controller = new AbortController();
  const prewarm = service.loadStylistCatalogInventory();
  const screen = service.loadStylistCatalogInventory(controller.signal);
  controller.abort();
  await expect(screen).rejects.toMatchObject({ name: 'AbortError' });
  finish(response);
  expect(await prewarm).toEqual([row]);
  expect(service.getCachedStylistCatalogInventory()).toEqual([row]);
  expect(mockFrom).toHaveBeenCalledTimes(4);
});

test('failed checks are retried and do not become verified empty inventory', async () => {
  pages(Promise.resolve({ data: null, error: new Error('offline') }));
  await expect(service.loadStylistCatalogInventory()).rejects.toThrow('offline');
  expect(service.getCachedStylistCatalogInventory()).toBeUndefined();
  pages(Promise.resolve(response));
  expect(await service.loadStylistCatalogInventory()).toEqual([row]);
  expect(mockFrom).toHaveBeenCalledTimes(8);
});

test('verified empty inventory stays empty until an explicit refresh checks again', async () => {
  pages(Promise.resolve({ data: [], error: null }));
  await service.loadStylistCatalogInventory();
  expect(service.getCachedStylistCatalogInventory()).toEqual([]);
  pages(Promise.resolve(response));
  expect(await service.loadStylistCatalogInventory()).toEqual([]);
  expect(await service.loadStylistCatalogInventory(undefined, true)).toEqual([row]);
  expect(mockFrom).toHaveBeenCalledTimes(8);
});
