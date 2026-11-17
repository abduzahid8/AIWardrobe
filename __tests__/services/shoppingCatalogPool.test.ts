/** @jest-environment node */
const mockFrom = jest.fn();
const mockRpc = jest.fn();
jest.mock('../../lib/supabase', () => ({
  supabase: {
    from: (...args: unknown[]) => mockFrom(...args),
    rpc: (...args: unknown[]) => mockRpc(...args),
  },
}));
jest.mock('@react-native-async-storage/async-storage', () => ({
  getItem: jest.fn(async () => null),
  setItem: jest.fn(async () => undefined),
}));
import { fetchShopPoolForStyle, shoppingService } from '../../src/services/shoppingService';
const rows = [
  'Macharles Loafer Black',
  'Macharles Loafer Brown',
  'Leather Sneaker White',
  'Chelsea Boot Brown',
].map((name, i) => ({
  id: 'row-' + i,
  name,
  brand: 'Brand',
  price: 100,
  currency: 'USD',
  image_url: `https://img.test/${i}.jpg`,
  garment_type: 'shoes',
  source: 'product-only',
  source_url: `https://merchant.test/products/${i}`,
}));
function query() {
  const q: any = {
    then: (resolve: any) => Promise.resolve({ data: rows, error: null }).then(resolve),
  };
  for (const method of ['select', 'eq', 'or', 'order', 'limit', 'range', 'abortSignal'])
    q[method] = jest.fn(() => q);
  return q;
}
// Shipped engine reads the catalog through the shared bounded discovery RPC
// (catalog_discovery_candidates) instead of raw shop_catalog paging.
function rpcPage(pageRows: unknown[], window = 30) {
  const abortSignal = jest.fn(() => Promise.resolve({ data: pageRows, error: null }));
  return { abortSignal, builder: { abortSignal } };
}
beforeEach(() => {
  mockFrom.mockReset();
  mockRpc.mockReset();
});
test('real bounded completion pool spreads footwear roles and keeps concrete prefixed row IDs', async () => {
  const page = rpcPage(rows);
  mockRpc.mockReturnValue(page.builder);
  const result = await fetchShopPoolForStyle(['shoes'], 'casual', 3);
  expect(result).toHaveLength(3);
  expect(result.filter((i) => /loafer/i.test(i.name))).toHaveLength(1);
  expect(result.every((i) => i.id.startsWith('shop_row-'))).toBe(true);
  // The bound moved from .range(0,29) into the RPC window: one 30-row page
  // per slot, starting at offset 0.
  expect(mockRpc).toHaveBeenCalledWith(
    'catalog_discovery_candidates',
    expect.objectContaining({ p_slot: 'shoes', p_limit: 30, p_offset: 0 }),
  );
  expect(page.abortSignal).toHaveBeenCalled();
});
test('legacy search adapter returns real destinations and no fabricated stock or ratings', async () => {
  mockRpc.mockReturnValue(rpcPage(rows).builder);
  const result = await shoppingService.searchProducts({ query: 'Loafer', category: 'shoes' });
  expect(result[0].id).toBe('row-0');
  expect(result[0].productUrl).toBe(rows[0].source_url);
  expect(result[0].inStock).toBeUndefined();
  expect(result[0].rating).toBeUndefined();
});

test('native-currency cheap representatives survive the actual completion-pool trim',async()=>{
 const affordable={...rows[3],id:'budget-boot',price:5};
 const q=query();q.then=(resolve:any)=>Promise.resolve({data:[...rows,affordable],error:null}).then(resolve);
 mockFrom.mockReturnValue(q);
 const discovery=require('../../src/services/catalogDiscovery');
 const load=jest.spyOn(discovery,'loadCatalogDiscovery').mockResolvedValueOnce({rows:[...rows,affordable],hasMore:false,strategy:'balanced-rpc'});
 const picks=await fetchShopPoolForStyle(['shoes'],'classic',2);
 expect(picks.some(item=>item.id==='shop_budget-boot')).toBe(true);load.mockRestore();
});
