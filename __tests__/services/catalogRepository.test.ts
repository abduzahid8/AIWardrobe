/** @jest-environment node */
const mockFrom = jest.fn();
jest.mock('../../lib/supabase', () => ({
  supabase: { from: (...args: unknown[]) => mockFrom(...args) },
}));
import {
  createFixtureCatalogRepository,
  createSupabaseCatalogRepository,
} from '../../src/services/catalogRepository';
import type { ShopCatalogItem } from '../../src/types/shop';
const fixture: ShopCatalogItem = {
  id: 'offline',
  name: 'Oxford Shirt Black',
  brand: 'Brand',
  price: 50,
  currency: 'USD',
  imageUrl: 'offline.jpg',
  garmentType: 'upper_body',
};
function query(result: unknown) {
  const q: any = { then: (resolve: any) => Promise.resolve(result).then(resolve) };
  for (const method of [
    'select',
    'eq',
    'or',
    'gte',
    'lte',
    'order',
    'range',
    'limit',
    'abortSignal',
  ])
    q[method] = jest.fn(() => q);
  return q;
}
beforeEach(() => mockFrom.mockReset());
test('live search filters before bounded paging, deduplicates concurrent reads and retains exact IDs', async () => {
  const q = query({
    data: [
      {
        id: 'live',
        name: 'Black Shirt',
        brand: 'Brand',
        price: 60,
        currency: 'USD',
        garment_type: 'upper_body',
        description: 'product-only',
        image_url: 'live.jpg',
        source_url: 'https://merchant.test/products/one',
      },
    ],
    error: null,
  });
  mockFrom.mockReturnValue(q);
  const repository = createSupabaseCatalogRepository(
    createFixtureCatalogRepository(() => [fixture])
  );
  const params = {
    query: 'shirt,)unsafe',
    discovery: false,
    category: 'tops',
    priceMax: 100,
    currency: 'USD',
    limit: 10,
    page: 2,
  };
  const [a, b] = await Promise.all([repository.search(params), repository.search(params)]);
  expect(a).toBe(b);
  expect(a.items[0].id).toBe('live');
  expect(mockFrom).toHaveBeenCalledTimes(1);
  expect(q.range).toHaveBeenCalledWith(20, 29);
  expect(q.eq).toHaveBeenCalledWith('garment_type', 'upper_body');
  expect(q.lte).toHaveBeenCalledWith('price', 100);
  expect(q.or.mock.calls[0][0]).toContain('name.ilike."%shirt,)unsafe%"');
  expect(a.items[0]).not.toHaveProperty('inStock');
});
test('offline search uses bundled rows, preserves filters and does not cache a failed live check', async () => {
  mockFrom.mockReturnValue(query({ data: null, error: new Error('offline') }));
  const repository = createSupabaseCatalogRepository(
    createFixtureCatalogRepository(() => [fixture])
  );
  expect(
    (await repository.search({ query: 'Oxford', discovery: false, currency: 'USD', priceMax: 60 })).items
  ).toEqual([fixture]);
  expect((await repository.search({ query: 'Oxford', discovery: false, currency: 'EUR' })).items).toEqual([]);
  await repository.search({ query: 'Oxford', discovery: false, currency: 'USD', priceMax: 60 });
  expect(mockFrom).toHaveBeenCalledTimes(3);
});
test('verified empty and provenance-filtered results do not turn into fake inventory', async () => {
  mockFrom.mockReturnValue(
    query({
      data: [{ id: 'person-photo', name: 'Shirt', description: 'model wearing shirt' }],
      error: null,
    })
  );
  const repository = createSupabaseCatalogRepository(
    createFixtureCatalogRepository(() => [fixture])
  );
  expect((await repository.search({ query: '' })).items).toEqual([]);
  expect(await repository.getVariant('person-photo')).toBeNull();
});

test('color requests cannot admit a different garment color mentioned only in styling copy',async()=>{
 mockFrom.mockReturnValue(query({data:[{id:'black',name:'Black Shirt',brand:'Brand',price:50,currency:'USD',garment_type:'upper_body',image_url:'image.jpg',description:'product-only; wear with blue trousers'}],error:null}));
 const repository=createSupabaseCatalogRepository(createFixtureCatalogRepository(()=>[]));
 expect((await repository.search({query:'shirt',color:'blue'})).items).toEqual([]);
});
