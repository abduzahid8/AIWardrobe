import {
  merchantAdapters,
  parseIncomingProduct,
  resolvePurchaseInput,
} from '../../src/services/purchase/merchantProducts';
import {
  purchaseShareLink,
  decodePurchaseShare,
  isPurchaseShareLink,
  parseShoppingIntent,
} from '../../src/services/purchase/shareReceiving';
import { row, now } from './fixtures';
const url = 'https://www.zara.com/uk/en/leather-loafers-p12345678.html';
test('supported merchant fixtures parse structured product/offer data offline without invented fees or measurements', async () => {
  const snapshot = {
    url,
    observedAt: new Date(now).toISOString(),
    structuredData: {
      '@type': 'Product',
      name: 'Brown Leather Loafers',
      brand: { name: 'Zara' },
      sku: '12345678',
      image: 'https://img.test/shoe.jpg',
      offers: {
        price: '89',
        priceCurrency: 'GBP',
        availability: 'https://schema.org/InStock',
        url,
      },
    },
  };
  const result = await resolvePurchaseInput(url, { snapshot });
  expect(result.status).toBe('parsed');
  expect(result.candidate?.price).toBe(89);
  expect(result.offers[0].inStock).toBe(true);
  expect(result.offers[0].shipping).toBeUndefined();
  expect(
    merchantAdapters.find((adapter) => adapter.id === 'zara')!.parseSizeChart(snapshot)
  ).toEqual([]);
});
test('exact catalog IDs remain authoritative, family resolution requires a concrete selection, and offline unknown merchants degrade gracefully', async () => {
  const existing = { ...row('real-row', 'Brown Loafer'), sourceUrl: url };
  expect((await resolvePurchaseInput(url, { catalog: [existing] })).candidate?.id).toBe('real-row');
  const familyUrl = 'https://www.jackjones.com/en-gb/product/12306276_2048/regular-fit-chinos';
  const variant = {
    ...row('concrete', 'Regular Fit Chino trousers', 'lower_body'),
    brand: 'JACK & JONES',
    sourceUrl: familyUrl,
  };
  const family = await resolvePurchaseInput(familyUrl.replace('_2048', '_5104'), {
    catalog: [variant],
  });
  expect(family.status).toBe('family');
  expect(family.candidate).toBeUndefined();
  expect(family.variants[0].id).toBe('concrete');
  expect(
    (
      await resolvePurchaseInput('https://unknown.test/item', {
        lookupUrl: async () => {
          throw Error('offline');
        },
      })
    ).status
  ).toBe('partial');
});
test('malformed/private/scheme URLs and incomplete snapshots cannot become trusted products', async () => {
  for (const input of [
    'https://',
    'http://127.0.0.1/item',
    'http://2130706433/',
    'file:///private/data',
    'javascript:alert(1)',
  ])
    expect(parseIncomingProduct(input).error).toBeDefined();
  const incomplete = await resolvePurchaseInput(url, {
    snapshot: {
      url,
      observedAt: new Date(now).toISOString(),
      html: '<script>not product data</script>',
    },
  });
  expect(incomplete.status).toBe('partial');
});
test('receiving text, URLs and budget intents use bounded deep links and never need an external scheduler', () => {
  const shared = purchaseShareLink('Check this ' + url);
  expect(isPurchaseShareLink(shared)).toBe(true);
  expect(decodePurchaseShare(shared).input).toBe(url);
  expect(decodePurchaseShare('com.aiwardrobe://buy-check?input=')).toHaveProperty('error');
  expect(parseShoppingIntent('I have RM500. What should I buy?')).toMatchObject({
    mode: 'budget',
    budget: { amount: 500, currency: 'MYR' },
  });
  expect(parseShoppingIntent('Find brown loafers under RM400')).toMatchObject({
    mode: 'find',
    budget: { amount: 400, currency: 'MYR' },
  });
});

test('a cold share waits for authentication/navigation and cannot carry across account changes', async () => {
  const { authEvents } = require('../../store/authEvents');
  const {
    queuePurchaseLink,
    takePurchaseInput,
    purchaseAwareLinking,
  } = require('../../src/services/purchase/shareQueue');
  authEvents.emitLogout();
  queuePurchaseLink(purchaseShareLink(url));
  authEvents.emitLogin('first-owner');
  expect(takePurchaseInput('first-owner')).toBe(url);
  queuePurchaseLink(purchaseShareLink(url));
  authEvents.emitLogout();
  authEvents.emitLogin('other-owner');
  expect(takePurchaseInput('other-owner')).toBeUndefined();
  const flush = jest.fn(),
    listener = jest.fn();
  let send!: (url: string) => void;
  const linking = purchaseAwareLinking(
    {
      getInitialURL: async () => purchaseShareLink(url),
      subscribe: (callback: (url: string) => void) => {
        send = callback;
        return () => undefined;
      },
    },
    flush
  );
  expect(await linking.getInitialURL()).toBeNull();
  expect(flush).toHaveBeenCalled();
  linking.subscribe(listener);
  send('com.aiwardrobe://reset-password');
  expect(listener).toHaveBeenCalledWith('com.aiwardrobe://reset-password');
});

test('incoming product links retain concrete variant selectors while removing authentication/session secrets', () => {
  const value = parseIncomingProduct(
    'https://merchant.test/product?color=blue&sessionid=private&access_token=secret&utm_source=x'
  );
  expect(value.url).toContain('color=blue');
  expect(value.url).not.toContain('private');
  expect(value.url).not.toContain('secret');
});

test('structured product groups never become physical candidates and retain each variant link', async () => {
  const parent = 'https://www.uniqlo.com/products/model',
    blue = parent + '?variant=blue',
    white = parent + '?variant=white';
  const data = {
    '@type': 'ProductGroup',
    name: 'Oxford Shirt',
    brand: { name: 'Uniqlo' },
    hasVariant: [
      {
        '@type': 'Product',
        sku: 'blue-m',
        name: 'Blue Oxford Shirt',
        url: blue,
        color: 'Blue',
        size: 'M',
        offers: {
          price: 40,
          priceCurrency: 'USD',
          url: blue,
          availability: 'https://schema.org/InStock',
        },
      },
      {
        '@type': 'Product',
        sku: 'white-m',
        name: 'White Oxford Shirt',
        url: white,
        color: 'White',
        size: 'M',
        offers: {
          price: 40,
          priceCurrency: 'USD',
          url: white,
          availability: 'https://schema.org/InStock',
        },
      },
    ],
  };
  const group = await resolvePurchaseInput(parent, {
    snapshot: { url: parent, observedAt: new Date(now).toISOString(), structuredData: data },
  });
  expect(group.status).toBe('family');
  expect(group.candidate).toBeUndefined();
  expect(group.variants).toHaveLength(2);
  expect(group.variants[1].sourceUrl).toBe(white);
  expect(group.variants[1].brand).toBe('Uniqlo');
  expect(group.offers[1].sizes?.M).toBe(true);
  const exact = await resolvePurchaseInput(white, {
    snapshot: { url: white, observedAt: new Date(now).toISOString(), structuredData: data },
  });
  expect(exact.candidate?.id).toContain('white-m');
  expect(exact.candidate?.sourceUrl).toBe(white);
});
