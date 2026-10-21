import { getShopDestination } from '../../src/utils/shopDestination';

// Shop URLs come from catalog feeds / edge enrichment — untrusted input.
// Only http(s) retailer links may reach Linking.openURL; every other
// scheme must degrade to a Google Shopping search, never pass through.
describe('getShopDestination hostile URLs', () => {
  it.each([
    'javascript:alert(1)',
    'data:text/html,<script>alert(1)</script>',
    'tel:+998000000000',
    'intent://evil#Intent;scheme=https;end',
    'com.aiwardrobe://open',
    '//evil.example/phish',
    'not a url at all',
  ])('never passes hostile shopUrl through: %s', (shopUrl) => {
    const dest = getShopDestination({ brand: 'Zara', name: 'Wool Coat', shopUrl });
    expect(dest.startsWith('https://www.google.com/search?tbm=shop')).toBe(true);
    expect(dest).not.toContain('evil');
    expect(dest).not.toContain('javascript:');
  });

  it('still passes legitimate retailer URLs through unchanged', () => {
    expect(
      getShopDestination({ name: 'Coat', shopUrl: 'https://shop.example.com/coat' }),
    ).toBe('https://shop.example.com/coat');
    expect(
      getShopDestination({ name: 'Coat', shopUrl: 'http://shop.example.com/coat' }),
    ).toBe('http://shop.example.com/coat');
  });
});
