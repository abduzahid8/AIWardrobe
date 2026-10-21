import { getShopDestination, getShopSearchUrl } from '../../src/utils/shopDestination';

describe('getShopDestination', () => {
  it('keeps the catalog retailer URL when one is available', () => {
    expect(
      getShopDestination({
        brand: 'Zara',
        name: 'Linen Shirt',
        shopUrl: 'https://www.zara.com/us/en/linen-shirt-p0123.html',
      })
    ).toBe('https://www.zara.com/us/en/linen-shirt-p0123.html');
  });

  it('normalizes a retailer URL whose feed omitted https', () => {
    expect(
      getShopDestination({
        brand: 'Zara',
        name: 'Linen Shirt',
        shopUrl: 'www.zara.com/us/en/linen-shirt-p0123.html',
      })
    ).toBe('https://www.zara.com/us/en/linen-shirt-p0123.html');
  });

  it('provides a product-specific shopping destination when a catalog URL is absent', () => {
    const destination = getShopDestination({ brand: 'Massimo Dutti', name: 'Wool Trousers' });

    expect(destination).toBe(getShopSearchUrl({ brand: 'Massimo Dutti', name: 'Wool Trousers' }));
    expect(destination).toContain('tbm=shop');
    expect(destination).toContain('Massimo%20Dutti%20Wool%20Trousers');
  });
});
